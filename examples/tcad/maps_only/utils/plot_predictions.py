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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; chosen before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_VARIABLE_LABELS = {
    "temperature": ("Temperature (K)", "MSE T (K$^2$)"),
    "potential": ("Potential (V)", "MSE V (V$^2$)"),
}


def _attach_shared_colorbar(
    fig,
    axes,
    value_range: tuple[float, float],
    colormap: str,
    label: str,
) -> None:
    """One colorbar shared between the two top-row scatter axes, fully frozen.

    The colorbar's data range, ticks, and tick labels are all pinned at
    creation time and must not move during the animation.
    """
    vmin, vmax = value_range
    sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=vmin, vmax=vmax))
    # Seed the ScalarMappable's array with [vmin, vmax] so the colorbar's
    # internal data range is pinned to exactly these endpoints — set_array([])
    # leaves it implicit and matplotlib can re-derive it on redraw.
    sm.set_array(np.array([vmin, vmax], dtype=float))

    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=25)
    cbar.set_label(label, fontsize=9)

    span = vmax - vmin
    decimals = max(2, int(np.ceil(-np.log10(span + 1e-30))) + 1) if span > 0 else 2
    ticks = np.linspace(vmin, vmax, 6)
    # High-level cbar API: set_ticks/set_ticklabels go through the colorbar's
    # own machinery and survive re-layouts more reliably than poking yaxis.
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.{decimals}f}" for v in ticks])
    cbar.ax.yaxis.offsetText.set_visible(False)
    # Final clamp on the colorbar axes' visible range — defends against any
    # autoscale that could be triggered by per-frame redraws of the scatters.
    cbar.ax.set_ylim(vmin, vmax)


def _configure_scatter_axes(
    ax,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    thickness_nm: float,
    elev: float,
    azim: float,
) -> None:
    """One-off cosmetic configuration for a 3D scatter subplot."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.grid(False)
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.pane.fill = False
        pane_axis.pane.set_edgecolor("none")
    for axis_vals, setter in [(x_nm, ax.set_xticks), (y_nm, ax.set_yticks)]:
        ticks = np.linspace(axis_vals.min(), axis_vals.max(), 5)
        setter(np.round(ticks, 1))
    z_step = thickness_nm / 5.0
    z_ticks = np.arange(0, thickness_nm + z_step * 0.5, z_step)
    ax.set_zticks(np.round(z_ticks, 2))


def animate_prediction(
    file: str | Path,
    variable: str,
    output: str = "prediction.gif",
    fps: int = 5,
    colormap: str = "plasma",
    ts_start: int = 0,
    ts_end: int | None = None,
    dpi: int = 100,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """Animate ground-truth vs prediction for one rollout, with synchronized MSE.

    The shared colorbar is pinned to the **ground-truth min/max** over the
    selected timestep window, so the prediction's overshoots or undershoots
    saturate against the physical reference.

    Parameters
    ----------
    file : path to a ``.pth`` file produced by ``generate.py``
    variable : ``"temperature"`` or ``"potential"`` — chooses which field is
        rendered on the top-row scatter plots. Both MSE traces are always shown
        on the bottom-row line plot regardless of this choice.
    output : output path (.gif uses Pillow writer, .mp4 uses ffmpeg)
    fps : animation frame rate
    colormap : matplotlib colormap name
    ts_start : first timestep to include (inclusive)
    ts_end : last timestep to include (exclusive); None = all
    dpi : figure resolution
    elev, azim : view angles in degrees for the 3D scatter plots
    """
    if variable not in _VARIABLE_LABELS:
        raise ValueError(
            f"variable must be one of {list(_VARIABLE_LABELS)}, got {variable!r}"
        )

    file = Path(file)
    print(f"Loading rollout: {file}")
    blob = torch.load(file, map_location="cpu", weights_only=False)

    positions = blob["positions"].numpy()  # (N, 3) meters
    times = blob["time"].numpy()  # (T,) seconds
    pred = blob[f"{variable}_pred"].numpy()  # (T, N)
    gt = blob[f"{variable}_gt"].numpy()  # (T, N)
    mse_T = blob["mse_temperature"].numpy()  # (T,)
    mse_V = blob["mse_potential"].numpy()  # (T,)
    thickness_m = float(blob["thickness"][0].item())
    thickness_nm = thickness_m * 1e9
    thickness_str = blob["thickness_str"]
    sim_id = int(blob["sim_id"])

    T_total = times.shape[0]
    if ts_end is None:
        ts_end = T_total
    ts_end = min(ts_end, T_total)
    if ts_start >= ts_end:
        raise ValueError(
            f"Empty range after filter: ts_start={ts_start} ts_end={ts_end}"
        )

    sel = slice(ts_start, ts_end)
    times_sel = times[sel]
    pred_sel = pred[sel]
    gt_sel = gt[sel]

    # Color range = exact GT min / max over the whole rollout. Predictions may
    # overshoot or undershoot — the shared colorbar saturates outside the GT
    # band, which is the desired physical reference.
    value_range = (float(gt_sel.min()), float(gt_sel.max()))
    cbar_label, _ = _VARIABLE_LABELS[variable]
    variable_title = variable.title()
    print(
        f"{variable_title} GT min/max: "
        f"[{value_range[0]:.4f}, {value_range[1]:.4f}]\n"
        f"Thickness: {thickness_nm:.1f} nm | sim_id: {sim_id} | "
        f"frames: {times_sel.shape[0]}"
    )

    # 2x2 grid: top row = scatter pair (each column), bottom row spans both.
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.05)
    ax_gt = fig.add_subplot(gs[0, 0], projection="3d")
    ax_pred = fig.add_subplot(gs[0, 1], projection="3d")
    ax_mse = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"TCAD prediction vs ground truth — thickness {thickness_str}, sim {sim_id}",
        fontsize=12,
    )

    # ---- 3D scatter artists created ONCE; only the color array updates per
    # frame. This keeps the shared colorbar's Normalize (and therefore its
    # ticks) completely frozen across the rollout. ----
    x_nm = positions[:, 0] * 1e9
    y_nm = positions[:, 1] * 1e9
    z_nm = positions[:, 2] * 1e9
    vmin, vmax = value_range

    scatter_gt = ax_gt.scatter(
        x_nm,
        y_nm,
        z_nm,
        c=gt_sel[0],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        s=30,
        depthshade=True,
    )
    scatter_pred = ax_pred.scatter(
        x_nm,
        y_nm,
        z_nm,
        c=pred_sel[0],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        s=30,
        depthshade=True,
    )
    _configure_scatter_axes(ax_gt, x_nm, y_nm, thickness_nm, elev, azim)
    _configure_scatter_axes(ax_pred, x_nm, y_nm, thickness_nm, elev, azim)

    t0 = float(times_sel[0])
    title_gt = ax_gt.set_title(
        f"{variable_title}  Ground truth — t={t0:.2e} s", fontsize=10
    )
    title_pred = ax_pred.set_title(
        f"{variable_title}  Prediction — t={t0:.2e} s", fontsize=10
    )

    # ---- Static MSE traces (drawn once; only the cursor moves) ----
    color_T = "tab:red"
    color_V = "tab:blue"
    ax_mse.plot(times, mse_T, color=color_T, label="MSE T (K$^2$)", lw=1.5)
    ax_mse.set_xlabel("Time (s)")
    ax_mse.set_ylabel("MSE T (K$^2$)", color=color_T)
    ax_mse.tick_params(axis="y", labelcolor=color_T)
    ax_mse.set_xlim(times.min(), times.max())
    ax_mse.grid(True, which="both", linestyle=":", alpha=0.5)

    ax_mse_v = ax_mse.twinx()
    ax_mse_v.plot(times, mse_V, color=color_V, label="MSE V (V$^2$)", lw=1.5)
    ax_mse_v.set_ylabel("MSE V (V$^2$)", color=color_V)
    ax_mse_v.tick_params(axis="y", labelcolor=color_V)

    (cursor_T,) = ax_mse.plot(
        [times_sel[0]], [mse_T[ts_start]], "o", color=color_T, markersize=8, zorder=5
    )
    (cursor_V,) = ax_mse_v.plot(
        [times_sel[0]], [mse_V[ts_start]], "o", color=color_V, markersize=8, zorder=5
    )
    cursor_line = ax_mse.axvline(
        times_sel[0], color="k", linestyle="--", lw=1.0, alpha=0.6
    )

    lines = [ax_mse.get_lines()[0], ax_mse_v.get_lines()[0]]
    ax_mse.legend(lines, [line.get_label() for line in lines], loc="upper left")

    _attach_shared_colorbar(fig, [ax_gt, ax_pred], value_range, colormap, cbar_label)

    def _update(frame_idx: int) -> None:
        global_idx = ts_start + frame_idx
        t = float(times_sel[frame_idx])
        scatter_gt.set_array(gt_sel[frame_idx])
        scatter_pred.set_array(pred_sel[frame_idx])
        title_gt.set_text(f"{variable_title}  Ground truth — t={t:.2e} s")
        title_pred.set_text(f"{variable_title}  Prediction — t={t:.2e} s")
        cursor_T.set_data([times[global_idx]], [mse_T[global_idx]])
        cursor_V.set_data([times[global_idx]], [mse_V[global_idx]])
        cursor_line.set_xdata([times[global_idx], times[global_idx]])

    anim = FuncAnimation(
        fig,
        _update,
        frames=times_sel.shape[0],
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
        description="Animate a rollout produced by generate.py: ground truth "
        "vs prediction (3D scatter), with synchronized MSE traces below."
    )
    p.add_argument(
        "--file",
        required=True,
        help="Path to the .pth file produced by generate.py",
    )
    p.add_argument(
        "--variable",
        required=True,
        choices=sorted(_VARIABLE_LABELS),
        help="Which field to render on the top-row scatter plots",
    )
    p.add_argument(
        "--output",
        default="prediction.gif",
        help="Output file (.gif or .mp4) [default: prediction.gif]",
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
    animate_prediction(
        file=args.file,
        variable=args.variable,
        output=args.output,
        fps=args.fps,
        colormap=args.colormap,
        ts_start=args.ts_start,
        ts_end=args.ts_end,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
    )
