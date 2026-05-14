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

"""Render the diffusion perf benchmark results as grouped bar plots.

Reads the YAML files in ``results/`` for a given GPU type and produces:
  * One PNG per (benchmark, QoI) pair: 3 benchmarks (training, inference,
    inference+DPS) x 3 QoIs (throughput, peak memory, MFU) = 9 plots.
  * One PNG comparing user-facing lines of code (LoC).

Visual style follows the PhysicsNeMo brand:
  * Pure-PyTorch baseline      gray fill
  * PhysicsNeMo (no opts)      NVIDIA-green outline, white fill
  * PhysicsNeMo + full opts    NVIDIA-green outline, green diagonal hash
  * PhysicsNeMo + MD + opts    NVIDIA-green outline, green fill
  * OOM run                    dark bold "X" marker

Run with the device whose results are in ``results/``::

    python -m examples.diffusion_perf.plot --device L40s
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch

# NVIDIA brand green sampled from the public DoMINO blog reference plot.
NVIDIA_GREEN = "#76B900"
BASELINE_GRAY = "#888888"
BASELINE_EDGE = "#444444"
OOM_COLOR = "#111111"

_RESULTS_DEFAULT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Bar styles per implementation
# ---------------------------------------------------------------------------

STYLES: dict[str, dict] = {
    "baseline": dict(
        facecolor=BASELINE_GRAY,
        edgecolor=BASELINE_EDGE,
        hatch=None,
        label="pure PyTorch (baseline)",
    ),
    "physicsnemo": dict(
        facecolor="white",
        edgecolor=NVIDIA_GREEN,
        hatch=None,
        label="PhysicsNeMo",
    ),
    "physicsnemo+opt": dict(
        facecolor="white",
        edgecolor=NVIDIA_GREEN,
        hatch="///",
        label="PhysicsNeMo + opt",
    ),
    "MD+opt": dict(
        facecolor=NVIDIA_GREEN,
        edgecolor=NVIDIA_GREEN,
        hatch=None,
        label="PhysicsNeMo + multi-diffusion + opt",
    ),
}
SETTING_ORDER = ["baseline", "physicsnemo", "physicsnemo+opt", "MD+opt"]


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

FULL_OPTS = ["amp_bf16", "apex_gn", "compile"]

BENCHMARKS = {
    "training": dict(
        title="Training (DDP)",
        batch_size_key="batch_size_train",
        settings={
            "baseline": ("train_baseline", []),
            "physicsnemo": ("train_physicsnemo", []),
            "physicsnemo+opt": ("train_physicsnemo", FULL_OPTS),
            "MD+opt": ("train_physicsnemo_multidiffusion", FULL_OPTS),
        },
    ),
    "inference": dict(
        title="Inference (no guidance)",
        batch_size_key="batch_size_infer",
        settings={
            "baseline": ("generate_baseline", []),
            "physicsnemo": ("generate_physicsnemo", []),
            "physicsnemo+opt": ("generate_physicsnemo", FULL_OPTS),
            "MD+opt": ("generate_physicsnemo_multidiffusion", FULL_OPTS),
        },
    ),
    "inference_dps": dict(
        title="Inference + DPS guidance",
        batch_size_key="batch_size_infer",
        settings={
            "baseline": ("generate_dps_baseline", []),
            "physicsnemo": ("generate_dps_physicsnemo", []),
            "physicsnemo+opt": ("generate_dps_physicsnemo", FULL_OPTS),
            "MD+opt": ("generate_dps_physicsnemo_multidiffusion", FULL_OPTS),
        },
    ),
}

QOIS = {
    "throughput": dict(
        ylabel="samples / s / GPU  (global resolution)",
        extractor=lambda r: r["results"].get("samples_per_sec_per_gpu_median"),
        log_y=True,
    ),
    "peak_memory": dict(
        ylabel="peak memory  (GB / GPU)",
        extractor=lambda r: r["results"].get("peak_memory_allocated_gb_max_rank"),
        log_y=False,
    ),
    "mfu": dict(
        ylabel="MFU  (% of BF16 peak)",
        extractor=lambda r: (
            r["results"]["mfu"] * 100.0 if r["results"].get("mfu") is not None else None
        ),
        log_y=False,
    ),
}


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _load_runs(device: str, results_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(results_dir.glob("*.yaml")):
        if p.name.startswith("_") or p.name == "summary.yaml":
            continue
        d = yaml.safe_load(p.read_text())
        if d is None:
            continue
        if d.get("device", {}).get("name") != device:
            continue
        d["_p"] = p.name
        runs.append(d)
    return runs


def _runs_for_setting(runs, function, opts, batch_size):
    """Return ``{domain: result_dict}`` for matching runs at a given B."""
    out: dict[int, dict] = {}
    for r in runs:
        if r["function"] != function:
            continue
        if sorted(r["config"].get("optimizations") or []) != sorted(opts):
            continue
        if r["config"].get("batch_size_per_rank") != batch_size:
            continue
        d = r["config"]["domain"][0]
        # Prefer most recent timestamp if duplicates exist
        existing = out.get(d)
        if existing is None or r["timestamp"] >= existing["timestamp"]:
            out[d] = r
    return out


def _batch_size_for(bench, batch_size_train, batch_size_infer):
    if bench["batch_size_key"] == "batch_size_train":
        return batch_size_train
    return batch_size_infer


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _draw_oom_marker(ax, x_center, *, log_y: bool):
    """Draw a thick black X at ~5% of the visible y range above the axis bottom."""
    # We place the marker in a hybrid transform: x in data coords, y in axes
    # fraction. That keeps it visible regardless of y-scale or rescaling.
    trans = ax.get_xaxis_transform()
    ax.plot(
        [x_center],
        [0.04],
        marker="X",
        markersize=14,
        markeredgewidth=2.5,
        markerfacecolor=OOM_COLOR,
        markeredgecolor=OOM_COLOR,
        linestyle="None",
        transform=trans,
        zorder=10,
        clip_on=False,
    )


def _plot_qoi(
    runs,
    *,
    bench_key: str,
    qoi_key: str,
    batch_size_train: int,
    batch_size_infer: int,
    out_dir: Path,
) -> Path | None:
    bench = BENCHMARKS[bench_key]
    qoi = QOIS[qoi_key]
    bs = _batch_size_for(bench, batch_size_train, batch_size_infer)

    # Gather all domains observed across settings.
    setting_runs: dict[str, dict[int, dict]] = {}
    all_domains: set[int] = set()
    for key in SETTING_ORDER:
        fn, opts = bench["settings"][key]
        s = _runs_for_setting(runs, fn, opts, bs)
        setting_runs[key] = s
        all_domains.update(s.keys())
    if not all_domains:
        print(f"  [{bench_key}/{qoi_key}] no data; skipping")
        return None

    domains = sorted(all_domains)
    x = np.arange(len(domains), dtype=float)
    n_set = len(SETTING_ORDER)
    bar_w = 0.8 / n_set

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, key in enumerate(SETTING_ORDER):
        style = STYLES[key]
        srun = setting_runs[key]
        offsets = x + (i - (n_set - 1) / 2) * bar_w
        # Track OOM frontier: once a setting OOMs at domain d, every larger
        # domain is treated as OOM (matches run_sweep's stop-on-first-OOM).
        first_oom = None
        for d in domains:
            r = srun.get(d)
            if r is not None and r["results"]["status"] == "oom":
                first_oom = d
                break
        for j, d in enumerate(domains):
            xb = float(offsets[j])
            r = srun.get(d)
            ok = (
                r is not None
                and r["results"]["status"] == "ok"
                and qoi["extractor"](r) is not None
                and qoi["extractor"](r) > 0
            )
            if ok:
                v = qoi["extractor"](r)
                ax.bar(
                    xb,
                    v,
                    width=bar_w,
                    facecolor=style["facecolor"],
                    edgecolor=style["edgecolor"],
                    hatch=style["hatch"],
                    linewidth=1.6,
                )
            else:
                # OOM if either the run itself OOMed OR the setting has already
                # gone over the cliff at a smaller domain.
                is_oom = (
                    r is not None and r["results"]["status"] in ("oom", "error")
                ) or (first_oom is not None and d >= first_oom)
                if is_oom:
                    _draw_oom_marker(ax, xb, log_y=qoi["log_y"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in domains])
    ax.set_xlabel("global domain edge (pixels)")
    ax.set_ylabel(qoi["ylabel"])
    if qoi["log_y"]:
        ax.set_yscale("log")
    ax.set_title(f"{bench['title']}  —  {qoi['ylabel'].split('(')[0].strip()}")
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")

    # Legend: bar swatches + OOM marker
    handles = [
        Patch(
            facecolor=STYLES[k]["facecolor"],
            edgecolor=STYLES[k]["edgecolor"],
            hatch=STYLES[k]["hatch"],
            label=STYLES[k]["label"],
            linewidth=1.6,
        )
        for k in SETTING_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color=OOM_COLOR,
            markersize=12,
            markeredgewidth=2.5,
            linestyle="None",
            label="OOM",
        )
    )
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    out = out_dir / f"{bench_key}__{qoi_key}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_loc(runs, *, out_dir: Path) -> Path | None:
    """Relative LoC (baseline = 100%). 3 benchmark groups x 2 bars (no MD)."""
    LOC_PAIRS = [
        ("training", "train_baseline", "train_physicsnemo"),
        ("inference", "generate_baseline", "generate_physicsnemo"),
        ("inference + DPS", "generate_dps_baseline", "generate_dps_physicsnemo"),
    ]
    seen: dict[str, int] = {}
    for r in runs:
        fn = r["function"]
        marked = r.get("loc", {}).get("marked_lines")
        if marked is None:
            continue
        if fn not in seen:
            seen[fn] = int(marked)

    rows: list[tuple[str, int, int]] = []
    for label, b_fn, p_fn in LOC_PAIRS:
        b = seen.get(b_fn)
        p = seen.get(p_fn)
        if b is None or p is None or b <= 0:
            print(f"  [loc] missing LoC for {label}: baseline={b} physicsnemo={p}")
            continue
        rows.append((label, b, p))
    if not rows:
        print("  [loc] no LoC data; skipping")
        return None

    x = np.arange(len(rows))
    bar_w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    base_vals = [100.0 for _ in rows]
    pnm_vals = [r[2] / r[1] * 100.0 for r in rows]
    base_abs = [r[1] for r in rows]
    pnm_abs = [r[2] for r in rows]

    b_bars = ax.bar(
        x - bar_w / 2,
        base_vals,
        width=bar_w,
        facecolor=BASELINE_GRAY,
        edgecolor=BASELINE_EDGE,
        linewidth=1.6,
        label="pure PyTorch (baseline)",
    )
    p_bars = ax.bar(
        x + bar_w / 2,
        pnm_vals,
        width=bar_w,
        facecolor=NVIDIA_GREEN,
        edgecolor=NVIDIA_GREEN,
        linewidth=1.6,
        label="PhysicsNeMo",
    )

    for bar, n in zip(b_bars, base_abs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{n} lines",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, n in zip(p_bars, pnm_abs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{n} lines",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("user-facing lines of code  (relative, baseline = 100%)")
    ax.set_title("Lines of code: pure PyTorch vs PhysicsNeMo")
    ax.set_ylim(0, max(120.0, max(pnm_vals) + 20))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out = out_dir / "loc_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="L40s",
        help="Device name to filter by (matches device.name in result YAMLs)",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=4,
        help="Per-rank training batch size to filter by",
    )
    parser.add_argument(
        "--batch-size-infer",
        type=int,
        default=1,
        help="Per-rank inference batch size to filter by",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <results-dir>/plots/<device>)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing per-run YAML files "
        "(default: examples/diffusion_perf/results/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or _RESULTS_DEFAULT
    out_dir = args.out_dir or (results_dir / "plots" / args.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(args.device, results_dir)
    print(f"Loaded {len(runs)} runs for device={args.device}")
    if not runs:
        return

    written: list[Path] = []
    for bench_key in BENCHMARKS:
        for qoi_key in QOIS:
            out = _plot_qoi(
                runs,
                bench_key=bench_key,
                qoi_key=qoi_key,
                batch_size_train=args.batch_size_train,
                batch_size_infer=args.batch_size_infer,
                out_dir=out_dir,
            )
            if out is not None:
                written.append(out)
    loc_out = _plot_loc(runs, out_dir=out_dir)
    if loc_out is not None:
        written.append(loc_out)

    print(f"\nWrote {len(written)} plot(s) to {out_dir}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
