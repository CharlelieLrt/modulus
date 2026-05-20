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

"""Calibration step — required first phase of the benchmark workflow.

Finds the maximum global domain size at which **non-multi-diffusion training**
fits with all optimizations enabled (``amp_bf16 + compile + apex_gn``), at the
training batch size (B=8). The result becomes the multi-diffusion patch-size
cap used by every benchmark::

    effective_patch_size = min(MAX_DOMAIN, current_global_domain)

The result is written to ``results/_max_domain.yaml``. Run this before
``run_sweep.py`` (which fails fast if the file is missing).

Bisection rules
---------------
* Phase 1: double from a small power-of-2 start until OOM.
* Phase 2: bisect ``(last_fit, first_oom)`` in multiples of ``PATCH_ALIGN``
  (= 16, the SongUNet downsampling alignment; every aligned-to-16 integer is
  decomposable into powers of two of degree >= 2, matching the framework
  contract).
"""

from __future__ import annotations

# Allow `torchrun calibrate.py` (file-style, no -m) by reattaching the package
# context so relative imports resolve. Required because ``submit_job`` invokes
# scripts as ``torchrun script.py`` from inside the example directory, not as
# ``python -m examples.diffusion_perf.calibrate`` from the repo root.
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _modulus_root = _os.path.abspath(_os.path.join(_here, "..", ".."))
    if _modulus_root not in _sys.path:
        _sys.path.insert(0, _modulus_root)
    __package__ = "examples.diffusion_perf"

import argparse
import datetime as _dt
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

from .bench.calibration import patch_shape_for, power_of_2_sweep
from .bench.config import (
    BATCH_SIZE_TRAIN,
    FULL_OPTS_TRAIN,
    MAX_GLOBAL_DOMAIN,
    MEASURE_STEPS,
    NPROC_PER_NODE_TRAIN,
    PATCH_ALIGN,
    WARMUP_STEPS,
    detect_device,
)

# Env vars set by an outer ``torchrun`` that must NOT leak into the inner
# ``torchrun`` subprocesses calibrate.py spawns for training probes.
_TORCHRUN_ENV_VARS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)


def _strip_torchrun_env(env: dict[str, str]) -> dict[str, str]:
    """Remove outer-torchrun env vars so the inner launcher starts clean."""
    for key in _TORCHRUN_ENV_VARS:
        env.pop(key, None)
    return env


_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_MAX_DOMAIN_YAML = _RESULTS_DIR / "_max_domain.yaml"
_TORCHRUN_PORT_BASE = 29700


@lru_cache(maxsize=1)
def _device_label() -> str:
    """Short stable device label for the current CUDA device.

    Lazy: not evaluated on the login node (which has no GPU). Matches the
    label that ``bench.results.ResultBuilder.write()`` uses for its filename
    so that probe lookups find their own outputs.
    """
    return detect_device()["name"]


def _calibration_dir() -> Path:
    """Subdirectory of ``results/`` reserved for calibration probe outputs.

    Each probe writes a full training-run YAML; isolating them here keeps
    ``plot.py`` and the sweep summary from picking up off-grid domains that
    the bisection happens to evaluate (e.g. d=576, 624, 640).
    """
    return _RESULTS_DIR / _device_label() / "calibration"


def _opts_str(opts: frozenset[str]) -> str:
    return "-".join(sorted(opts)) if opts else "none"


def _yaml_path(
    *, function: str, domain: int, opts: frozenset[str], batch_size: int
) -> Path:
    return _calibration_dir() / (
        f"{function}_{_device_label()}_d{domain}_b{batch_size}_opt-{_opts_str(opts)}.yaml"
    )


def _read_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return yaml.safe_load(path.read_text())["results"]["status"]
    except Exception:
        return "missing"


def _read_util(path: Path) -> float:
    """Read peak memory utilization (0..1) from a results YAML."""
    if not path.exists():
        return 0.0
    try:
        data = yaml.safe_load(path.read_text())
        return float(data["results"].get("peak_memory_utilization") or 0.0)
    except Exception:
        return 0.0


# Memory headroom: a probe at >MEM_FRAC_CAP utilization is treated as
# "doesn't fit at our safety margin" so a different test on a different GPU
# (same memory class) won't OOM at the boundary.
MEM_FRAC_CAP: float = 0.90


def _run_subprocess_train(
    *, domain: int, batch_size: int, port_offset: int, warmup: int, measure: int
) -> None:
    env = _strip_torchrun_env(dict(os.environ))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    port = _TORCHRUN_PORT_BASE + port_offset
    cmd = [
        "torchrun",
        f"--nproc-per-node={NPROC_PER_NODE_TRAIN}",
        f"--master-port={port}",
        "-m",
        "examples.diffusion_perf.train",
        "--function",
        "train_physicsnemo",
        "--domain",
        str(domain),
        "--opts",
        ",".join(sorted(FULL_OPTS_TRAIN)),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--measure",
        str(measure),
        "--output-dir",
        str(_calibration_dir()),
    ]
    print(f"[calibrate] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, cwd=Path(__file__).resolve().parents[2])


def _probe(
    domain: int, batch_size: int, port_offset: int, warmup: int, measure: int
) -> tuple[str, float]:
    """Returns ``(effective_status, peak_util)``.

    A probe is considered ``"oom"`` if the run actually OOMed OR if it ran but
    used more than ``MEM_FRAC_CAP`` of GPU memory (so the calibration leaves a
    safety margin that survives noise / slightly different GPUs).
    """
    _run_subprocess_train(
        domain=domain,
        batch_size=batch_size,
        port_offset=port_offset,
        warmup=warmup,
        measure=measure,
    )
    path = _yaml_path(
        function="train_physicsnemo",
        domain=domain,
        opts=FULL_OPTS_TRAIN,
        batch_size=batch_size,
    )
    raw_status = _read_status(path)
    util = _read_util(path)
    if raw_status == "ok" and util > MEM_FRAC_CAP:
        return f"over_cap({util:.0%})", util
    return raw_status, util


def find_max_domain(
    *,
    batch_size: int = BATCH_SIZE_TRAIN,
    start: int = 64,
    cap: int = 8192,
    warmup: int = WARMUP_STEPS,
    measure: int = MEASURE_STEPS,
) -> dict:
    """Two-phase bisection. Returns the calibration report dict."""

    log = []
    last_fit = 0
    last_fit_util = 0.0
    first_oom = None
    print(
        f"[calibrate] Phase 1: doubling (target ≤ {int(MEM_FRAC_CAP * 100)}% memory)",
        flush=True,
    )
    for i, d in enumerate(power_of_2_sweep(start=start, cap=cap)):
        st, util = _probe(d, batch_size, port_offset=i, warmup=warmup, measure=measure)
        log.append({"domain": d, "phase": "doubling", "status": st, "util": util})
        print(f"[calibrate] d={d}: {st} (util={util:.0%})", flush=True)
        if st == "ok":
            last_fit = d
            last_fit_util = util
        else:
            first_oom = d
            break

    if last_fit > 0 and first_oom is not None and first_oom - last_fit > PATCH_ALIGN:
        print(
            f"[calibrate] Phase 2: bisecting [{last_fit}, {first_oom}] in steps of {PATCH_ALIGN}",
            flush=True,
        )
        port = 100
        while first_oom - last_fit > PATCH_ALIGN:
            mid = ((last_fit + first_oom) // 2 // PATCH_ALIGN) * PATCH_ALIGN
            if mid <= last_fit:
                break
            st, util = _probe(
                mid, batch_size, port_offset=port, warmup=warmup, measure=measure
            )
            log.append({"domain": mid, "phase": "bisect", "status": st, "util": util})
            print(f"[calibrate] d={mid}: {st} (util={util:.0%})", flush=True)
            port += 1
            if st == "ok":
                last_fit = mid
                last_fit_util = util
            else:
                first_oom = mid

    return {
        "max_domain": last_fit,
        "max_domain_util": last_fit_util,
        "first_oom_domain": first_oom,
        "mem_frac_cap": MEM_FRAC_CAP,
        "batch_size": batch_size,
        "patch_align": PATCH_ALIGN,
        "opts": sorted(FULL_OPTS_TRAIN),
        "device": _device_label(),
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat() + "Z",
        "probe_log": log,
    }


def save_max_domain(report: dict) -> Path:
    """Persist a calibration report to ``results/_max_domain.yaml``."""
    _MAX_DOMAIN_YAML.parent.mkdir(parents=True, exist_ok=True)
    _MAX_DOMAIN_YAML.write_text(yaml.safe_dump(report, sort_keys=False))
    return _MAX_DOMAIN_YAML


def load_max_domain() -> dict | None:
    """Return the cached calibration report, or ``None`` if absent / unreadable."""
    if not _MAX_DOMAIN_YAML.exists():
        return None
    try:
        return yaml.safe_load(_MAX_DOMAIN_YAML.read_text())
    except Exception:
        return None


def main():
    """CLI entry point for the calibration step."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_TRAIN,
        help="Batch size used during calibration (default = training BS)",
    )
    parser.add_argument("--cap", type=int, default=MAX_GLOBAL_DOMAIN)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run calibration even if cached YAML exists",
    )
    args = parser.parse_args()

    if not args.force and _MAX_DOMAIN_YAML.exists():
        cached = load_max_domain()
        print(
            f"[calibrate] cached MAX_DOMAIN = {cached['max_domain']} "
            f"(use --force to recalibrate)",
            flush=True,
        )
        return

    report = find_max_domain(
        batch_size=args.batch_size,
        cap=args.cap,
        warmup=args.warmup,
        measure=args.measure,
    )
    path = save_max_domain(report)
    print(f"[calibrate] MAX_DOMAIN = {report['max_domain']}", flush=True)
    print(f"[calibrate] written to {path}", flush=True)


if __name__ == "__main__":
    main()
