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
  * Pure-PyTorch baseline                  solid gray
  * ``physicsnemo.diffusion``              NVIDIA-green outline, dotted fill
  * ``physicsnemo.diffusion`` + opt        NVIDIA-green outline, diagonal-hatch
  * ``physicsnemo.diffusion`` + MD + opt   NVIDIA-green solid fill
  * OOM run                                styled stub-bar + dark "X" overlay

Run with the device whose results are in ``results/``::

    python -m examples.diffusion_perf.plot --device L40s
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch, Polygon

# Heavier hatch lines so the dotted / diagonal patterns are clearly
# readable at the bar sizes we use.
mpl.rcParams["hatch.linewidth"] = 1.8

# NVIDIA brand green sampled from the public DoMINO blog reference plot.
NVIDIA_GREEN = "#76B900"
BASELINE_GRAY = "#888888"
BASELINE_EDGE = "#444444"
OOM_X_COLOR = "#1f1f1f"

# Inline code-style label that renders ``physicsnemo.diffusion`` in a
# monospace font, so the legend/title makes clear we are comparing against
# the framework's diffusion toolkit specifically.
PNM = r"$\mathtt{physicsnemo.diffusion}$"

# Fallback decimal-GB memory capacity by device label. Used only when the
# loaded YAML's ``total_memory_gb`` is missing or obviously a GiB-typed
# marketing number (e.g. "L40s: 48.0" should be 48 GiB = 51.54 GB decimal).
# Match the convention used by ``torch.cuda.max_memory_allocated() / 1e9``.
_DEVICE_TOTAL_MEM_GB_DECIMAL = {
    "L40s": 48.0 * 1024**3 / 1e9,  # 51.54
    "H100-SXM-80GB": 80.0 * 1024**3 / 1e9,  # 85.90
    "H100-PCIe-80GB": 80.0 * 1024**3 / 1e9,
    "A100-SXM-80GB": 80.0 * 1024**3 / 1e9,
    "A100-SXM-40GB": 40.0 * 1024**3 / 1e9,  # 42.95
    "B100": 192.0 * 1024**3 / 1e9,  # 206.16
}

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
        hatch="...",
        label=PNM,
    ),
    "physicsnemo+opt": dict(
        facecolor="white",
        edgecolor=NVIDIA_GREEN,
        hatch="///",
        label=PNM + " + opt",
    ),
    "MD+opt": dict(
        facecolor=NVIDIA_GREEN,
        edgecolor=NVIDIA_GREEN,
        hatch=None,
        label=PNM + " + multi-diffusion + opt",
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
        short_title="throughput",
        extractor=lambda r: r["results"].get("samples_per_sec_per_gpu_median"),
        format=lambda v: f"{v:.2g}",
        log_y=True,
    ),
    "peak_memory": dict(
        ylabel="peak memory  (GB / GPU)",
        short_title="peak GPU memory",
        extractor=lambda r: r["results"].get("peak_memory_allocated_gb_max_rank"),
        format=lambda v: f"{v:.1f}",
        log_y=True,
    ),
    "mfu": dict(
        ylabel="MFU  (% of BF16 peak)",
        short_title="MFU",
        extractor=lambda r: (
            r["results"]["mfu"] * 100.0 if r["results"].get("mfu") is not None else None
        ),
        format=lambda v: f"{v:.1f}",
        log_y=False,
    ),
}


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _load_runs(device: str, results_dir: Path) -> list[dict]:
    """Load every per-run YAML in ``results_dir`` filtered to ``device``.

    Filters out calibration probe runs at non-power-of-2 domains (the
    canonical sweep is on powers of 2; calibration probes leak per-run
    YAMLs at intermediate sizes that should not appear in the bars).
    """
    runs = []
    for p in sorted(results_dir.glob("*.yaml")):
        if p.name.startswith("_") or p.name == "summary.yaml":
            continue
        d = yaml.safe_load(p.read_text())
        if d is None:
            continue
        if d.get("device", {}).get("name") != device:
            continue
        dom = (d.get("config", {}).get("domain") or [None])[0]
        if not (isinstance(dom, int) and dom > 0 and (dom & (dom - 1)) == 0):
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
        existing = out.get(d)
        if existing is None or r["timestamp"] >= existing["timestamp"]:
            out[d] = r
    return out


def _batch_size_for(bench, batch_size_train, batch_size_infer):
    if bench["batch_size_key"] == "batch_size_train":
        return batch_size_train
    return batch_size_infer


def _config_summary(runs, results_dir: Path | None = None):
    """Pull (device, model_class, params, channels, total_memory_gb,
    max_domain) from the first framework run available so we use the actual
    measured backbone, plus optionally read MAX_DOMAIN from
    ``_max_domain.yaml`` for use in plot titles (multi-diffusion patch
    cap)."""
    device = model_class = params = channels = total_memory_gb = None
    for r in runs:
        fn = r["function"]
        if fn.startswith(
            ("train_baseline", "generate_baseline", "generate_dps_baseline")
        ):
            continue
        params = r.get("backbone", {}).get("params")
        model_class = r.get("backbone", {}).get("class")
        device = r.get("device", {}).get("name")
        channels = r.get("config", {}).get("channels")
        total_memory_gb = r.get("device", {}).get("total_memory_gb")
        break
    else:
        r = runs[0]
        device = r.get("device", {}).get("name")
        model_class = r.get("backbone", {}).get("class")
        params = r.get("backbone", {}).get("params")
        channels = r.get("config", {}).get("channels")
        total_memory_gb = r.get("device", {}).get("total_memory_gb")

    # Older YAMLs hardcoded total_memory_gb to the marketing GiB number
    # (e.g. L40s: 48.0). Prefer the decimal-GB value so the displayed
    # capacity line is consistent with the decimal-GB peak_memory column.
    if device in _DEVICE_TOTAL_MEM_GB_DECIMAL:
        total_memory_gb = _DEVICE_TOTAL_MEM_GB_DECIMAL[device]

    max_domain = None
    if results_dir is not None:
        cal = results_dir / "_max_domain.yaml"
        if cal.exists():
            try:
                max_domain = int(yaml.safe_load(cal.read_text())["max_domain"])
            except Exception:
                max_domain = None
    return device, model_class, params, channels, total_memory_gb, max_domain


def _world_size_for_bench(runs, bench, batch_size):
    """Number of ranks observed in any matching benchmark run."""
    fn_set = {fn for fn, _ in bench["settings"].values()}
    for r in runs:
        if (
            r["function"] in fn_set
            and r["config"].get("batch_size_per_rank") == batch_size
        ):
            ws = r.get("world_size")
            if ws:
                return int(ws)
    return None


# ---------------------------------------------------------------------------
# OOM marker
# ---------------------------------------------------------------------------


def _draw_oom_marker(ax, x_center: float, bar_w: float, style: dict):
    """Render OOM as a plain bold dark ``X`` marker at the bottom of the
    plot in (data-x, axes-y) blended coordinates. Same glyph for every
    setting; the missing bar plus the cross is enough to indicate that
    the configuration did not run."""
    del bar_w, style  # not used: uniform marker style for all OOMs
    trans = ax.get_xaxis_transform()
    ax.plot(
        [x_center],
        [0.05],
        marker="X",
        markersize=9,
        markeredgewidth=1.6,
        markerfacecolor=OOM_X_COLOR,
        markeredgecolor=OOM_X_COLOR,
        linestyle="None",
        transform=trans,
        zorder=10,
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Plot one (benchmark, QoI) pair
# ---------------------------------------------------------------------------


def _plot_qoi(
    runs,
    *,
    bench_key,
    qoi_key,
    batch_size_train,
    batch_size_infer,
    out_dir: Path,
    results_dir: Path | None,
) -> Path | None:
    """Emit a grouped bar plot for one (benchmark, QoI) tuple."""
    bench = BENCHMARKS[bench_key]
    qoi = QOIS[qoi_key]
    bs = _batch_size_for(bench, batch_size_train, batch_size_infer)

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
    # Wider bars + bigger intra-group gap (so OOM crosses don't visually
    # touch) + larger group_spacing so the inter-group whitespace stays
    # clearly wider than the intra-group span.
    group_spacing = 1.55
    x = np.arange(len(domains), dtype=float) * group_spacing
    n_set = len(SETTING_ORDER)
    bar_w = 0.225  # data units, single bar width
    inter_gap = 0.040  # gap between adjacent bars within a group
    step = bar_w + inter_gap  # center-to-center spacing within a group

    device, model_class, params, channels, total_mem, max_domain = _config_summary(
        runs, results_dir=results_dir
    )
    world_size = _world_size_for_bench(runs, bench, bs) or 1
    # Extra vertical room so the title + 2-row legend fit above the axes
    # without overlapping the plot.
    fig, ax = plt.subplots(figsize=(12.0, 6.8))

    for i, setting_key in enumerate(SETTING_ORDER):
        style = STYLES[setting_key]
        srun = setting_runs[setting_key]
        offsets = x + (i - (n_set - 1) / 2) * step

        first_oom = None
        for d in domains:
            r = srun.get(d)
            if r is not None and r["results"]["status"] == "oom":
                first_oom = d
                break

        heights = np.full(len(domains), np.nan)
        oom_flags = np.zeros(len(domains), dtype=bool)
        for j, d in enumerate(domains):
            r = srun.get(d)
            ok = (
                r is not None
                and r["results"]["status"] == "ok"
                and qoi["extractor"](r) is not None
                and qoi["extractor"](r) > 0
            )
            if ok:
                heights[j] = qoi["extractor"](r)
            else:
                is_oom = (
                    r is not None and r["results"]["status"] in ("oom", "error")
                ) or (first_oom is not None and d >= first_oom)
                if is_oom:
                    oom_flags[j] = True

        valid = ~np.isnan(heights)
        if valid.any():
            bars = ax.bar(
                offsets[valid],
                heights[valid],
                width=bar_w,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=2.4,
            )
            # Only annotate values when the y-scale is log (where the bar
            # heights are otherwise hard to read off the axis). Linear-y
            # plots are read straight off the grid.
            if qoi["log_y"]:
                labels = [qoi["format"](h) for h in heights[valid]]
                ax.bar_label(
                    bars,
                    labels=labels,
                    rotation=90,
                    padding=4,
                    fontsize=14,
                    color="#222",
                )

        for j in np.where(oom_flags)[0]:
            _draw_oom_marker(ax, float(offsets[j]), bar_w, style)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in domains], fontsize=12)
    ax.set_xlabel("Global domain edge (pixels)", fontsize=13)
    ax.set_ylabel(qoi["ylabel"], fontsize=13)
    if qoi["log_y"]:
        ax.set_yscale("log")

    # Title line 1: what the plot shows.
    line1 = f"{bench['title']}  —  {qoi['short_title']}"
    # Title line 2: GPU + parallelism + model + key data params.
    is_training = bench["batch_size_key"] == "batch_size_train"
    gpu_str = (
        f"{world_size}x {device} (DDP)"
        if is_training and world_size > 1
        else f"{world_size}x {device}"
    )
    params_str = f"{params / 1e6:.0f}M" if params else "?M"
    md_patch_str = (
        f"  |  MD patch <= {max_domain}x{max_domain}" if max_domain is not None else ""
    )
    line2 = (
        f"{gpu_str}  |  {model_class} ({params_str} params)  |  "
        f"{channels}-channel data  |  B={bs}/rank{md_patch_str}"
    )
    # Title sits a bit further above so the legend (placed below it but
    # above the axes) does not crowd it.
    ax.set_title(f"{line1}\n{line2}", fontsize=13, pad=70)
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--", linewidth=1.0)

    # Reference line for total GPU memory capacity (peak-memory plot only).
    if qoi_key == "peak_memory" and total_mem:
        ax.axhline(
            y=total_mem,
            color="#444444",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            zorder=1,
        )
        # Left edge, just below the dashed line: x in axes fraction (so it
        # always sits inside the plot region) and y in data coords.
        ax.text(
            0.01,
            total_mem,
            f"GPU capacity = {total_mem:.1f} GB",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="top",
            fontsize=11,
            color="#444444",
        )

    handles = [
        Patch(
            facecolor=STYLES[k]["facecolor"],
            edgecolor=STYLES[k]["edgecolor"],
            hatch=STYLES[k]["hatch"],
            label=STYLES[k]["label"],
            linewidth=2.4,
        )
        for k in SETTING_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="X",
            color=OOM_X_COLOR,
            markersize=9,
            markeredgewidth=1.6,
            linestyle="None",
            label="OOM",
        )
    )
    # Legend lives ABOVE the axes (under the title), 2 rows of ~3 entries
    # so it never overlaps the bars and stays compact.
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=10,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # Headroom inside the axes; legend lives outside the axes (above) so
    # we only need enough room for value labels.
    y_lo, y_hi = ax.get_ylim()
    if qoi["log_y"]:
        ax.set_ylim(y_lo, y_hi * 2.0)
    else:
        ax.set_ylim(y_lo, y_hi * 1.15)

    fig.tight_layout()
    out = out_dir / f"{bench_key}_{qoi_key}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# LoC plot
# ---------------------------------------------------------------------------


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
    # Larger inner offset between the gray + green bars so the green fill
    # does not visually overlap the gray bar's dark outline.
    inner_offset = bar_w / 2 + 0.03

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    base_vals = [100.0 for _ in rows]
    pnm_vals = [r[2] / r[1] * 100.0 for r in rows]

    b_bars = ax.bar(
        x - inner_offset,
        base_vals,
        width=bar_w,
        facecolor=BASELINE_GRAY,
        edgecolor=BASELINE_EDGE,
        linewidth=2.4,
        zorder=2,
        label="pure PyTorch (baseline)",
    )
    p_bars = ax.bar(
        x + inner_offset,
        pnm_vals,
        width=bar_w,
        facecolor=NVIDIA_GREEN,
        edgecolor=NVIDIA_GREEN,
        linewidth=2.4,
        zorder=3,
        label=PNM,
    )

    ax.bar_label(
        b_bars,
        labels=["100%" for _ in rows],
        rotation=0,
        padding=5,
        fontsize=14,
        color="#222",
    )
    ax.bar_label(
        p_bars,
        labels=[f"{v:.0f}%" for v in pnm_vals],
        rotation=0,
        padding=5,
        fontsize=14,
        color="#222",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=13)
    ax.set_ylabel("User-facing lines of code  (relative, baseline = 100%)", fontsize=13)
    ax.set_title(
        f"Lines of user code: pure PyTorch vs {PNM}",
        fontsize=14,
        pad=50,
    )
    ax.set_ylim(0, 130)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fontsize=12,
        columnspacing=2.0,
        handletextpad=0.6,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1.0)

    fig.tight_layout()
    out = out_dir / "loc_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: load YAMLs for a device, write PNGs to ``out-dir``."""
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
                results_dir=results_dir,
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
