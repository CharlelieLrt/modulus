# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; chosen before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator

# Make the sibling `dataset/` package importable when running as
# `python utils/plot_3d.py` from the recipe root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import TCADMapsDataset  # noqa: E402


def render_frame(
    ax_temp,
    ax_pot,
    positions: np.ndarray,
    temperature: np.ndarray,
    potential: np.ndarray,
    temp_range: tuple[float, float],
    pot_range: tuple[float, float],
    thickness_nm: float,
    colormap: str = "plasma",
    title: str = "",
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """Clear and redraw temperature and potential 3D scatter plots for one frame.

    Parameters
    ----------
    ax_temp, ax_pot : Axes3D
        The two matplotlib 3D axes to draw on.
    positions : (N, 3) array in meters
    temperature : (N,) array in K
    potential : (N,) array in V
    temp_range, pot_range : fixed (vmin, vmax) for the colorbars
    thickness_nm : device thickness in nm (for box aspect ratio)
    elev, azim : view angles in degrees (matplotlib defaults: 30, -60)
    """
    for ax in (ax_temp, ax_pot):
        ax.cla()
        ax.view_init(elev=elev, azim=azim)

    # Convert from meters to nm for legible axis labels
    x_nm = positions[:, 0] * 1e9
    y_nm = positions[:, 1] * 1e9
    z_nm = positions[:, 2] * 1e9

    xy_span = max(np.ptp(x_nm), np.ptp(y_nm))  # ~15 nm

    panels = [
        (ax_temp, temperature, temp_range, "Temperature (K)"),
        (ax_pot, potential, pot_range, "Potential (V)"),
    ]
    for ax, values, (vmin, vmax), cbar_label in panels:
        sc = ax.scatter(
            x_nm,
            y_nm,
            z_nm,
            c=values,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            s=30,
            depthshade=True,
        )
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")

        # Cube-like aspect ratio so the thin Z dimension doesn't get squashed
        ax.set_box_aspect([1.0, 1.0, 1.0])

        # Remove grid and background panes; keep only the main axis lines
        ax.grid(False)
        for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane_axis.pane.fill = False
            pane_axis.pane.set_edgecolor("none")

        # Ticks showing actual domain extent
        for axis_vals, setter in [
            (x_nm, ax.set_xticks),
            (y_nm, ax.set_yticks),
        ]:
            ticks = np.linspace(axis_vals.min(), axis_vals.max(), 5)
            setter(np.round(ticks, 1))
        z_step = thickness_nm / 5.0
        z_ticks = np.arange(0, thickness_nm + z_step * 0.5, z_step)
        ax.set_zticks(np.round(z_ticks, 2))

        # Colorbar: create once on the first frame, never update (range is fixed).
        if not hasattr(ax, "_tcad_cbar"):
            # Independent ScalarMappable so the colorbar is not tied to the scatter.
            sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.5, pad=0.12, aspect=20)
            cbar.set_label(cbar_label, fontsize=9)
            # Explicit ticks and labels suppress matplotlib's auto-offset notation (+3e2)
            span = vmax - vmin
            decimals = (
                max(2, int(np.ceil(-np.log10(span + 1e-30))) + 1) if span > 0 else 2
            )
            ticks = np.linspace(vmin, vmax, 6)
            cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
            cbar.ax.yaxis.set_major_formatter(
                FixedFormatter([f"{v:.{decimals}f}" for v in ticks])
            )
            cbar.ax.yaxis.offsetText.set_visible(False)
            ax._tcad_cbar = True  # mark as created

    if title:
        ax_temp.set_title(f"Temperature  {title}", fontsize=10)
        ax_pot.set_title(f"Potential  {title}", fontsize=10)


def animate_simulation(
    data_dir: str | Path,
    thickness_str: str,
    sim_id: int,
    output: str = "animation.gif",
    fps: int = 5,
    colormap: str = "plasma",
    ts_start: int = 0,
    ts_end: int | None = None,
    dpi: int = 100,
    clip_temp: float = 2.0,
    clip_pot: float = 35.0,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """Create and save a 3D scatter animation for one simulation.

    Parameters
    ----------
    data_dir : path to maps_only/data/
    thickness_str : "2nm", "3nm", or "4nm"
    sim_id : simulation index
    output : output path (.gif uses Pillow writer, .mp4 uses ffmpeg)
    fps : animation frame rate
    colormap : matplotlib colormap name
    ts_start : first timestep to include (inclusive)
    ts_end : last timestep to include (exclusive); None = all
    dpi : figure resolution
    clip_temp, clip_pot : clip this percent from each end of the temperature /
        potential distributions when computing the colorbar range (e.g. 2.0 →
        use [P2, P98] instead of [min, max]). Higher values = more saturation,
        making mid-range variations more visible. Use 0 for no clipping.
    """
    # Use n_steps=1 so each sample corresponds to exactly one timestep.
    dataset = TCADMapsDataset(data_dir, n_steps=1)
    indices = dataset.get_sim_indices(thickness_str, sim_id)

    # Apply ts_start / ts_end filter
    indices = [i for i in indices if ts_start <= dataset._samples[i][2]]
    if ts_end is not None:
        indices = [i for i in indices if dataset._samples[i][2] < ts_end]
    if not indices:
        raise ValueError(
            f"No samples remain after ts_start={ts_start}, ts_end={ts_end} filter."
        )

    # Load all frames upfront so we can compute fixed color ranges
    print(f"Loading {len(indices)} frames …")
    positions_np: np.ndarray | None = None
    frames: list[tuple[np.ndarray, np.ndarray, float]] = []

    for idx in indices:
        sample, _ = dataset[idx]
        if positions_np is None:
            positions_np = sample["positions"].numpy()
        # variables shape: (1, 2, N) with n_steps=1
        temp = sample["variables"][0, 0].numpy()
        pot = sample["variables"][0, 1].numpy()
        time_s = sample["time"][0].item()
        frames.append((temp, pot, time_s))

    # Use percentile clipping so rare extreme values don't flatten the rest
    temp_all = np.concatenate([f[0] for f in frames])
    pot_all = np.concatenate([f[1] for f in frames])
    temp_range = (
        float(np.percentile(temp_all, clip_temp)),
        float(np.percentile(temp_all, 100.0 - clip_temp)),
    )
    pot_range = (
        float(np.percentile(pot_all, clip_pot)),
        float(np.percentile(pot_all, 100.0 - clip_pot)),
    )
    thickness_m = sample["thickness"][0].item()
    thickness_nm = thickness_m * 1e9

    print(
        f"Temperature range: [{temp_range[0]:.2f}, {temp_range[1]:.2f}] K\n"
        f"Potential range:   [{pot_range[0]:.4f}, {pot_range[1]:.4f}] V\n"
        f"Thickness: {thickness_nm} nm"
    )

    fig, (ax_temp, ax_pot) = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "3d"},
        figsize=(14, 6),
    )
    fig.suptitle(
        f"TCAD simulation — thickness {thickness_nm:.0f} nm, sim {sim_id}",
        fontsize=11,
    )
    fig.tight_layout(pad=2.0)

    def _update(frame_idx: int) -> None:
        temp, pot, time_s = frames[frame_idx]
        render_frame(
            ax_temp,
            ax_pot,
            positions=positions_np,
            temperature=temp,
            potential=pot,
            temp_range=temp_range,
            pot_range=pot_range,
            thickness_nm=thickness_nm,
            colormap=colormap,
            title=f"t = {time_s:.2e} s",
            elev=elev,
            azim=azim,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=1000 // fps,
        repeat=False,
    )

    output_path = Path(output)
    writer = "pillow" if output_path.suffix.lower() == ".gif" else "ffmpeg"
    print(f"Saving animation to {output_path} …")
    anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    print("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate TCAD temperature and potential fields as 3D scatter plots."
    )
    p.add_argument("--data-dir", required=True, help="Path to maps_only/data/")
    p.add_argument(
        "--thickness",
        required=True,
        choices=["2nm", "3nm", "4nm"],
        help="Device thickness",
    )
    p.add_argument("--sim-id", required=True, type=int, help="Simulation index")
    p.add_argument(
        "--output",
        default="animation.gif",
        help="Output file (.gif or .mp4) [default: animation.gif]",
    )
    p.add_argument("--fps", type=int, default=5, help="Frames per second [default: 5]")
    p.add_argument(
        "--colormap", default="plasma", help="Matplotlib colormap [default: plasma]"
    )
    p.add_argument(
        "--ts-start", type=int, default=0, help="First timestep to include [default: 0]"
    )
    p.add_argument(
        "--ts-end",
        type=int,
        default=None,
        help="Last timestep (exclusive) [default: all]",
    )
    p.add_argument("--dpi", type=int, default=100, help="Figure DPI [default: 100]")
    p.add_argument(
        "--clip-temp",
        type=float,
        default=2.0,
        help="Clip this percent from each end of the temperature distribution "
        "when computing its colorbar range (0 = use min/max) [default: 2.0]",
    )
    p.add_argument(
        "--clip-pot",
        type=float,
        default=35.0,
        help="Same for potential. Default is much higher than temperature "
        "because the boundary-bias values repeat a lot and small clips "
        "are no-ops [default: 35.0]",
    )
    p.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="3D view elevation angle in degrees [default: 30]",
    )
    p.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D view azimuth angle in degrees; add ~180 to see the opposite "
        "face of the cube [default: -60]",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    animate_simulation(
        data_dir=args.data_dir,
        thickness_str=args.thickness,
        sim_id=args.sim_id,
        output=args.output,
        fps=args.fps,
        colormap=args.colormap,
        ts_start=args.ts_start,
        ts_end=args.ts_end,
        dpi=args.dpi,
        clip_temp=args.clip_temp,
        clip_pot=args.clip_pot,
        elev=args.elev,
        azim=args.azim,
    )
