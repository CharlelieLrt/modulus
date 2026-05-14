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

"""YAML result serialization for the benchmark sweeps."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _git_info(repo_root: Path) -> dict:
    info: dict = {"commit": None, "branch": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    return info


class ResultBuilder:
    """Assembles the YAML schema described in the design doc.

    Required setters are called by the harness as values become available.
    ``to_dict()`` returns the final record. ``write()`` saves it to ``results/``.
    """

    def __init__(self, *, function: str, output_dir: Path | str) -> None:
        self._data: dict[str, Any] = {
            "function": function,
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat() + "Z",
            "device": {},
            "world_size": 1,
            "config": {},
            "backbone": {},
            "results": {"status": "ok"},
            "loc": {"marked_lines": None},
            "git": _git_info(Path(__file__).resolve().parents[3]),
        }
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # --- setters ----------------------------------------------------------
    def device(self, info: dict) -> "ResultBuilder":
        self._data["device"] = info
        return self

    def world_size(self, n: int) -> "ResultBuilder":
        self._data["world_size"] = n
        return self

    def config(self, **kwargs: Any) -> "ResultBuilder":
        self._data["config"].update(kwargs)
        return self

    def backbone(
        self, *, class_name: str, params: int, flops_per_step: int | None
    ) -> "ResultBuilder":
        self._data["backbone"] = {
            "class": class_name,
            "params": int(params),
            "flops_per_step": int(flops_per_step)
            if flops_per_step is not None
            else None,
        }
        return self

    def loc(self, marked_lines: int | None) -> "ResultBuilder":
        self._data["loc"]["marked_lines"] = (
            int(marked_lines) if marked_lines is not None else None
        )
        return self

    def status(self, status: str, error: str | None = None) -> "ResultBuilder":
        self._data["results"]["status"] = status
        if error:
            self._data["results"]["error"] = error
        return self

    def timing(self, summary: dict) -> "ResultBuilder":
        median_ms = summary["step_time_ms_median"]
        # Per-rank samples/sec, where each step processes B samples.
        B = self._data["config"].get("batch_size_per_rank", 1)
        sps_per_gpu = B * 1000.0 / median_ms if median_ms and median_ms > 0 else None
        self._data["results"].update(
            {
                "step_time_ms_median": median_ms,
                "step_time_ms_p25": summary["step_time_ms_p25"],
                "step_time_ms_p75": summary["step_time_ms_p75"],
                "samples_per_sec_per_gpu_median": sps_per_gpu,
                "samples_per_sec_per_gpu_p25": (
                    B * 1000.0 / summary["step_time_ms_p75"]
                    if summary["step_time_ms_p75"] and summary["step_time_ms_p75"] > 0
                    else None
                ),
                "samples_per_sec_per_gpu_p75": (
                    B * 1000.0 / summary["step_time_ms_p25"]
                    if summary["step_time_ms_p25"] and summary["step_time_ms_p25"] > 0
                    else None
                ),
                "num_measured": summary["num_measured"],
            }
        )
        return self

    def memory(self, summary: dict) -> "ResultBuilder":
        self._data["results"].update(
            {
                "peak_memory_allocated_gb_max_rank": summary[
                    "peak_memory_allocated_gb"
                ],
                "peak_memory_utilization": summary["peak_memory_utilization"],
            }
        )
        return self

    def mfu(self, *, flops_per_step: int, world_size: int = 1) -> "ResultBuilder":
        # MFU = (FLOPs per step / step time) / theoretical peak.
        # Doesn't depend on batch size or patches-per-sample because
        # flops_per_step already integrates whatever the step actually computes.
        step_ms = self._data["results"].get("step_time_ms_median")
        peak = self._data["device"].get("bf16_peak_tflops")
        if (
            step_ms is None
            or step_ms <= 0
            or peak is None
            or peak <= 0
            or flops_per_step <= 0
        ):
            self._data["results"]["mfu"] = None
            return self
        achieved_tflops = (flops_per_step * 1000.0 / step_ms) / 1e12
        self._data["results"]["mfu"] = float(achieved_tflops / peak)
        self._data["results"]["achieved_tflops_per_gpu"] = float(achieved_tflops)
        return self

    # --- output -----------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self._data)

    def write(self, name: str | None = None) -> Path:
        if name is None:
            fn = self._data["function"]
            dev = self._data["device"].get("name", "unknown")
            cfg = self._data["config"]
            domain = cfg.get("domain", ["?", "?"])
            opts = "-".join(sorted(cfg.get("optimizations", []))) or "none"
            bs = cfg.get("batch_size_per_rank", 1)
            name = f"{fn}_{dev}_d{domain[0]}_b{bs}_opt-{opts}.yaml"
        path = self._output_dir / name
        with path.open("w") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=False)
        return path


def save_yaml(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _summarize_dir(results_dir: Path | str) -> list[dict]:
    """Read every YAML in ``results_dir`` and return them as a list."""

    results_dir = Path(results_dir)
    out: list[dict] = []
    for path in sorted(results_dir.glob("*.yaml")):
        if path.name.endswith("_summary.yaml"):
            continue
        with path.open() as fh:
            out.append(yaml.safe_load(fh))
    return out


def write_summary(
    results_dir: Path | str, *, summary_name: str = "summary.yaml"
) -> Path:
    """Aggregate per-run YAMLs into one summary file."""

    results_dir = Path(results_dir)
    all_runs = _summarize_dir(results_dir)
    summary_path = results_dir / summary_name
    save_yaml({"runs": all_runs, "n_runs": len(all_runs)}, summary_path)
    return summary_path


# Optional helper, useful for ad-hoc debugging.
def env_summary() -> dict:
    import torch as _torch

    return {
        "torch_version": _torch.__version__,
        "cuda_version": _torch.version.cuda,
        "cudnn_version": _torch.backends.cudnn.version(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }
