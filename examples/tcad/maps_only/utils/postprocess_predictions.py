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


def _time_to_failure(
    temperature: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> float:
    """First time at which the hottest point in the device crosses ``threshold``.

    Parameters
    ----------
    temperature : (T, N) array of temperature values in K.
    times : (T,) array of timestamps in seconds, monotonically increasing.
    threshold : temperature in K above which the device is considered failed.

    Returns
    -------
    Failure time in seconds, or ``NaN`` if the threshold is never exceeded over
    the simulation horizon.
    """
    # Use the per-step maximum: a thermal-runaway failure criterion is naturally
    # driven by the hottest point in the device, not its mean.
    peak = temperature.max(axis=1)
    above = np.where(peak > threshold)[0]
    if above.size == 0:
        return float("nan")
    return float(times[above[0]])


def _adaptive_bins(n_samples: int) -> int:
    """Square-root rule clipped to a sensible visual range."""
    return max(8, min(40, int(round(np.sqrt(max(n_samples, 1))))))


def time_to_failure_histogram(
    files_dir: str | Path,
    output: str = "time_to_failure_histogram.png",
    threshold_factor: float = 2.0,
    dpi: int = 120,
) -> None:
    """Compute time-to-failure for every rollout in ``files_dir`` and plot a histogram.

    Recursively scans ``files_dir`` for ``.pth`` files produced by
    ``generate.py``. For each file, the failure threshold is set to
    ``threshold_factor`` times the mean initial temperature (averaged over the
    point cloud at t=0 from the ground-truth field). The time-to-failure is the
    first timestamp at which the per-step maximum temperature exceeds that
    threshold; ``NaN`` if the threshold is never reached. Both the predicted
    and ground-truth distributions are plotted as overlaid transparent
    histograms with vertical lines marking their means.
    """
    files_dir = Path(files_dir)
    if not files_dir.exists():
        raise FileNotFoundError(f"Directory not found: {files_dir}")
    files = sorted(files_dir.rglob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found under {files_dir}")
    print(f"Found {len(files)} rollout file(s) under {files_dir}")

    ttf_pred_list: list[float] = []
    ttf_gt_list: list[float] = []
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        times = blob["time"].numpy()
        T_pred = blob["temperature_pred"].numpy()  # (T, N)
        T_gt = blob["temperature_gt"].numpy()  # (T, N)

        baseline = float(T_gt[0].mean())
        threshold = threshold_factor * baseline

        ttf_p = _time_to_failure(T_pred, times, threshold)
        ttf_g = _time_to_failure(T_gt, times, threshold)
        ttf_pred_list.append(ttf_p)
        ttf_gt_list.append(ttf_g)
        print(
            f"  {path.relative_to(files_dir)}: baseline={baseline:.1f} K | "
            f"threshold={threshold:.1f} K | ttf_gt={ttf_g:.3e}s | ttf_pred={ttf_p:.3e}s"
        )

    ttf_pred = np.array(ttf_pred_list, dtype=np.float64)
    ttf_gt = np.array(ttf_gt_list, dtype=np.float64)
    pred_finite = ttf_pred[~np.isnan(ttf_pred)]
    gt_finite = ttf_gt[~np.isnan(ttf_gt)]

    if pred_finite.size == 0 and gt_finite.size == 0:
        raise RuntimeError(
            "No simulation reached the failure threshold in either pred or GT; "
            "consider lowering --threshold-factor."
        )

    pred_mean = float(pred_finite.mean()) if pred_finite.size else float("nan")
    gt_mean = float(gt_finite.mean()) if gt_finite.size else float("nan")
    n_bins = _adaptive_bins(max(pred_finite.size, gt_finite.size))

    print(
        f"\nSummary:\n"
        f"  pred: {pred_finite.size}/{ttf_pred.size} reached threshold | "
        f"mean ttf = {pred_mean:.3e} s\n"
        f"  gt:   {gt_finite.size}/{ttf_gt.size} reached threshold | "
        f"mean ttf = {gt_mean:.3e} s\n"
        f"  bins: {n_bins}"
    )

    # Common bin edges so the two distributions share an x-axis
    combined = np.concatenate([gt_finite, pred_finite])
    edges = np.histogram_bin_edges(combined, bins=n_bins)

    color_gt = "tab:blue"
    color_pred = "tab:orange"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    if gt_finite.size:
        ax.hist(
            gt_finite,
            bins=edges,
            color=color_gt,
            alpha=0.5,
            label=f"Ground truth (n={gt_finite.size})",
        )
    if pred_finite.size:
        ax.hist(
            pred_finite,
            bins=edges,
            color=color_pred,
            alpha=0.5,
            label=f"Prediction (n={pred_finite.size})",
        )
    if not np.isnan(gt_mean):
        ax.axvline(
            gt_mean,
            color=color_gt,
            lw=2.0,
            label=f"GT mean = {gt_mean:.2e} s",
        )
    if not np.isnan(pred_mean):
        ax.axvline(
            pred_mean,
            color=color_pred,
            lw=2.0,
            label=f"Pred mean = {pred_mean:.2e} s",
        )

    ax.set_xlabel("Time-to-failure (s)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Time-to-failure distribution "
        f"(threshold = {threshold_factor:.1f}× initial mean temperature)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    # Always-visible summary so an empty distribution (e.g. an undertrained
    # model that never crosses the threshold) is still obvious at a glance.
    summary = (
        f"Threshold reached\n"
        f"  GT:   {gt_finite.size} / {ttf_gt.size}\n"
        f"  Pred: {pred_finite.size} / {ttf_pred.size}"
    )
    ax.text(
        0.98,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    fig.tight_layout()

    output_path = Path(output)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved histogram to {output_path}")


_MODES = {
    "time_to_failure": time_to_failure_histogram,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Postprocess one or more rollouts produced by generate.py."
    )
    p.add_argument(
        "--files",
        required=True,
        help="Directory containing .pth rollout files (searched recursively).",
    )
    p.add_argument(
        "--mode",
        default="time_to_failure",
        choices=sorted(_MODES),
        help="Postprocessing mode [default: time_to_failure]",
    )
    p.add_argument(
        "--output",
        default="time_to_failure_histogram.png",
        help="Output figure path [default: time_to_failure_histogram.png]",
    )
    p.add_argument(
        "--threshold-factor",
        type=float,
        default=2.0,
        help="Failure threshold = factor × mean initial temperature [default: 2.0]",
    )
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI [default: 120]")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "time_to_failure":
        time_to_failure_histogram(
            files_dir=args.files,
            output=args.output,
            threshold_factor=args.threshold_factor,
            dpi=args.dpi,
        )
    else:
        # Future modes plug in via the _MODES registry.
        raise ValueError(f"Unknown --mode {args.mode!r}")
