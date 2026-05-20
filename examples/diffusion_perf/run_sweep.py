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

"""Sweep orchestrator.

Pre-requirement: ``calibrate.py`` has been run and produced
``results/_max_domain.yaml``. If missing, this script errors out with the
exact command to run.

For each benchmark (training, inference, inference_dps), runs the same 4
settings across ``DOMAIN_SWEEP``, with one subprocess per (setting, domain):

    1. baseline        — pure-PyTorch, FP32, no framework
    2. framework       — physicsnemo, FP32, no opts
    3. framework_opts  — physicsnemo + amp_bf16 + compile + apex_gn
    4. md              — physicsnemo + multi-diffusion (setting 3 with
                         MultiDiffusionModel2D wrap, patch_shape =
                         min(domain, MAX_DOMAIN))

Non-MD settings stop on first OOM. MD is expected never to OOM (the patch
shape is bounded by what training already proved fits).

CLI selectors (combinable):

* ``--suite`` picks which of the 3 benchmark groups run.
* ``--domains`` overrides the swept domain edges (default: full sweep
  truncated by ``--max-global-domain``).
* ``--settings`` subsets the 4 settings above (default: all 4).
* ``--skip-existing`` skips cases whose result YAML already exists, which
  makes a partially-completed sweep cheaply resumable.
"""

from __future__ import annotations

# Allow `torchrun run_sweep.py` (file-style, no -m) by reattaching the package
# context so relative imports resolve. See calibrate.py for rationale.
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _modulus_root = _os.path.abspath(_os.path.join(_here, "..", ".."))
    if _modulus_root not in _sys.path:
        _sys.path.insert(0, _modulus_root)
    __package__ = "examples.diffusion_perf"

import argparse
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from .bench.config import (
    BATCH_SIZE_INFER,
    BATCH_SIZE_TRAIN,
    DOMAIN_SWEEP_FULL,
    FULL_OPTS_INFER,
    FULL_OPTS_TRAIN,
    MAX_GLOBAL_DOMAIN,
    MEASURE_STEPS,
    MEASURE_STEPS_INFER,
    NPROC_PER_NODE_TRAIN,
    WARMUP_STEPS,
    WARMUP_STEPS_INFER,
    detect_device,
)
from .bench.calibration import patch_shape_for
from .bench.results import write_summary
from .calibrate import _strip_torchrun_env, load_max_domain

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_TORCHRUN_PORT_BASE = 29800


@lru_cache(maxsize=1)
def _device_label() -> str:
    """Short stable device label for the current CUDA device.

    Lazy: not evaluated on the login node (which has no GPU). Matches the
    label that ``bench.results.ResultBuilder.write()`` uses for its filename
    so that probe lookups find their own outputs.
    """
    return detect_device()["name"]


SETTING_NAMES = ["baseline", "framework", "framework_opts", "md"]


def _opts_to_str(opts: frozenset[str]) -> str:
    return ",".join(sorted(opts)) if opts else "none"


def _opts_str(opts: frozenset[str]) -> str:
    return "-".join(sorted(opts)) if opts else "none"


def _yaml_path(
    *, function: str, domain: int, opts: frozenset[str], batch_size: int
) -> Path:
    return _RESULTS_DIR / (
        f"{function}_{_device_label()}_d{domain}_b{batch_size}_opt-{_opts_str(opts)}.yaml"
    )


def _read_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return yaml.safe_load(path.read_text())["results"]["status"]
    except Exception:
        return "missing"


def _maybe_run(
    *,
    yaml_path: Path,
    cmd: list[str],
    multi_gpu: bool,
    skip_existing: bool,
    port_offset: int = 0,
) -> None:
    """Run ``cmd``, or skip if ``skip_existing`` and ``yaml_path`` already exists."""
    if skip_existing and yaml_path.exists():
        print(f"[skip] {yaml_path.name} already exists", flush=True)
        return
    _run_subprocess(cmd, multi_gpu=multi_gpu, port_offset=port_offset)


def _run_subprocess(
    cmd: list[str], *, multi_gpu: bool = False, port_offset: int = 0
) -> int:
    env = _strip_torchrun_env(dict(os.environ))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if multi_gpu:
        port = _TORCHRUN_PORT_BASE + port_offset
        full = [
            "torchrun",
            f"--nproc-per-node={NPROC_PER_NODE_TRAIN}",
            f"--master-port={port}",
            *cmd,
        ]
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        full = ["python", *cmd]
    print(f"[run] {' '.join(full)}", flush=True)
    rc = subprocess.run(
        full, env=env, cwd=Path(__file__).resolve().parents[2]
    ).returncode
    print(f"[run] exited rc={rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_cmd(
    *, function, domain, opts, warmup, measure, batch_size, patch_shape=None
) -> list[str]:
    cmd = [
        "-m",
        "examples.diffusion_perf.train",
        "--function",
        function,
        "--domain",
        str(domain),
        "--opts",
        _opts_to_str(opts),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--measure",
        str(measure),
    ]
    if patch_shape is not None:
        cmd += ["--patch-shape", str(patch_shape[0]), str(patch_shape[1])]
    return cmd


def run_training_suite(
    *,
    max_domain: int,
    warmup: int,
    measure: int,
    domains: list[int],
    active_settings: set[str],
    skip_existing: bool,
):
    """4 settings × ``domains`` at training B/rank DDP."""
    bs = BATCH_SIZE_TRAIN
    print(f"[training] settings={sorted(active_settings)} at B={bs} DDP", flush=True)
    # Track OOM per setting to avoid wasting time on configs guaranteed to fail
    state = {"baseline": False, "framework": False, "framework_opts": False}
    port = 0
    for d in domains:
        # Setting 1: baseline (pure pytorch, FP32)
        if "baseline" in active_settings and not state["baseline"]:
            path = _yaml_path(
                function="train_baseline",
                domain=d,
                opts=frozenset(),
                batch_size=bs,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_train_cmd(
                    function="train_baseline",
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                skip_existing=skip_existing,
                port_offset=port,
            )
            port += 1
            if _read_status(path) != "ok":
                state["baseline"] = True
                print(f"[training] baseline OOM at d={d}", flush=True)
        # Setting 2: physicsnemo (no opts, FP32)
        if "framework" in active_settings and not state["framework"]:
            path = _yaml_path(
                function="train_physicsnemo",
                domain=d,
                opts=frozenset(),
                batch_size=bs,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_train_cmd(
                    function="train_physicsnemo",
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                skip_existing=skip_existing,
                port_offset=port,
            )
            port += 1
            if _read_status(path) != "ok":
                state["framework"] = True
                print(f"[training] physicsnemo OOM at d={d}", flush=True)
        # Setting 3: physicsnemo + full opts
        if "framework_opts" in active_settings and not state["framework_opts"]:
            path = _yaml_path(
                function="train_physicsnemo",
                domain=d,
                opts=FULL_OPTS_TRAIN,
                batch_size=bs,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_train_cmd(
                    function="train_physicsnemo",
                    domain=d,
                    opts=FULL_OPTS_TRAIN,
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                skip_existing=skip_existing,
                port_offset=port,
            )
            port += 1
            if _read_status(path) != "ok":
                state["framework_opts"] = True
                print(f"[training] physicsnemo+opts OOM at d={d}", flush=True)
        # Setting 4: MD + full opts; never OOMs (patch bounded by MAX_DOMAIN)
        if "md" in active_settings and max_domain > 0:
            patch = patch_shape_for(d, max_domain)
            path = _yaml_path(
                function="train_physicsnemo_multidiffusion",
                domain=d,
                opts=FULL_OPTS_TRAIN,
                batch_size=bs,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_train_cmd(
                    function="train_physicsnemo_multidiffusion",
                    domain=d,
                    opts=FULL_OPTS_TRAIN,
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                    patch_shape=patch,
                ),
                multi_gpu=True,
                skip_existing=skip_existing,
                port_offset=port,
            )
            port += 1


# ---------------------------------------------------------------------------
# Inference (no DPS / with DPS share the same shape)
# ---------------------------------------------------------------------------


def _infer_cmd(
    module, *, function, domain, opts, warmup, measure, patch_shape=None, chunk_size=1
) -> list[str]:
    cmd = [
        "-m",
        module,
        "--function",
        function,
        "--domain",
        str(domain),
        "--opts",
        _opts_to_str(opts),
        "--warmup",
        str(warmup),
        "--measure",
        str(measure),
    ]
    if patch_shape is not None:
        cmd += ["--patch-shape", str(patch_shape[0]), str(patch_shape[1])]
    if "multidiffusion" in function:
        cmd += ["--chunk-size", str(chunk_size)]
    return cmd


def _run_inference_suite(
    *,
    module: str,
    fn_baseline: str,
    fn_physicsnemo: str,
    fn_md: str,
    max_domain: int,
    warmup: int,
    measure: int,
    label: str,
    domains: list[int],
    active_settings: set[str],
    skip_existing: bool,
):
    print(f"[{label}] settings={sorted(active_settings)}", flush=True)
    state = {"baseline": False, "framework": False, "framework_opts": False}
    for d in domains:
        if "baseline" in active_settings and not state["baseline"]:
            path = _yaml_path(
                function=fn_baseline,
                domain=d,
                opts=frozenset(),
                batch_size=BATCH_SIZE_INFER,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_infer_cmd(
                    module,
                    function=fn_baseline,
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
                skip_existing=skip_existing,
            )
            if _read_status(path) != "ok":
                state["baseline"] = True
                print(f"[{label}] baseline OOM at d={d}", flush=True)
        if "framework" in active_settings and not state["framework"]:
            path = _yaml_path(
                function=fn_physicsnemo,
                domain=d,
                opts=frozenset(),
                batch_size=BATCH_SIZE_INFER,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_infer_cmd(
                    module,
                    function=fn_physicsnemo,
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
                skip_existing=skip_existing,
            )
            if _read_status(path) != "ok":
                state["framework"] = True
                print(f"[{label}] physicsnemo OOM at d={d}", flush=True)
        if "framework_opts" in active_settings and not state["framework_opts"]:
            path = _yaml_path(
                function=fn_physicsnemo,
                domain=d,
                opts=FULL_OPTS_INFER,
                batch_size=BATCH_SIZE_INFER,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_infer_cmd(
                    module,
                    function=fn_physicsnemo,
                    domain=d,
                    opts=FULL_OPTS_INFER,
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
                skip_existing=skip_existing,
            )
            if _read_status(path) != "ok":
                state["framework_opts"] = True
                print(f"[{label}] physicsnemo+opts OOM at d={d}", flush=True)
        if "md" in active_settings and max_domain > 0:
            patch = patch_shape_for(d, max_domain)
            path = _yaml_path(
                function=fn_md,
                domain=d,
                opts=FULL_OPTS_INFER,
                batch_size=BATCH_SIZE_INFER,
            )
            _maybe_run(
                yaml_path=path,
                cmd=_infer_cmd(
                    module,
                    function=fn_md,
                    domain=d,
                    opts=FULL_OPTS_INFER,
                    warmup=warmup,
                    measure=measure,
                    patch_shape=patch,
                    chunk_size=1,
                ),
                multi_gpu=False,
                skip_existing=skip_existing,
            )


def run_inference_suite(
    *, max_domain, warmup, measure, domains, active_settings, skip_existing
):
    """4 settings × ``domains`` for the inference-no-guidance benchmark."""
    _run_inference_suite(
        module="examples.diffusion_perf.generate",
        fn_baseline="generate_baseline",
        fn_physicsnemo="generate_physicsnemo",
        fn_md="generate_physicsnemo_multidiffusion",
        max_domain=max_domain,
        warmup=warmup,
        measure=measure,
        label="inference",
        domains=domains,
        active_settings=active_settings,
        skip_existing=skip_existing,
    )


def run_inference_dps_suite(
    *, max_domain, warmup, measure, domains, active_settings, skip_existing
):
    """4 settings × ``domains`` for the inference + DPS-guidance benchmark."""
    _run_inference_suite(
        module="examples.diffusion_perf.generate_dps_guidance",
        fn_baseline="generate_dps_baseline",
        fn_physicsnemo="generate_dps_physicsnemo",
        fn_md="generate_dps_physicsnemo_multidiffusion",
        max_domain=max_domain,
        warmup=warmup,
        measure=measure,
        label="inference_dps",
        domains=domains,
        active_settings=active_settings,
        skip_existing=skip_existing,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the sweep orchestrator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=True,
        choices=["training", "inference", "inference_dps", "all"],
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS)
    parser.add_argument("--warmup-infer", type=int, default=WARMUP_STEPS_INFER)
    parser.add_argument("--measure-infer", type=int, default=MEASURE_STEPS_INFER)
    parser.add_argument(
        "--max-global-domain",
        type=int,
        default=MAX_GLOBAL_DOMAIN,
        help=(
            "Largest global domain edge to sweep. Truncates "
            "DOMAIN_SWEEP_FULL from the top. Ignored if --domains is given. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--domains",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit list of global domain edges to sweep, e.g. "
            "`--domains 512 1024 2048`. Overrides --max-global-domain."
        ),
    )
    parser.add_argument(
        "--settings",
        choices=SETTING_NAMES,
        nargs="+",
        default=SETTING_NAMES,
        help=("Subset of the 4 settings to run within each suite. Default: all 4."),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip cases whose result YAML already exists in results/, so a "
            "partially completed sweep can be cheaply resumed."
        ),
    )
    args = parser.parse_args()
    if args.domains is not None:
        domains = sorted(set(args.domains))
    else:
        domains = [d for d in DOMAIN_SWEEP_FULL if d <= args.max_global_domain]
    if not domains:
        sys.exit(
            f"[run_sweep] no domains to sweep "
            f"(--domains={args.domains}, --max-global-domain={args.max_global_domain})."
        )
    active_settings = set(args.settings)
    print(
        f"[run_sweep] sweep domains = {domains}; settings = {sorted(active_settings)}",
        flush=True,
    )

    report = load_max_domain()
    if report is None:
        sys.exit(
            "[run_sweep] missing _max_domain.yaml — run calibration first:\n"
            "    python -m examples.diffusion_perf.calibrate"
        )
    max_domain = int(report["max_domain"])
    print(
        f"[run_sweep] MAX_DOMAIN = {max_domain} (calibrated at "
        f"B={report.get('batch_size')}, opts={report.get('opts')})",
        flush=True,
    )
    if max_domain <= 0:
        sys.exit("[run_sweep] calibration found MAX_DOMAIN=0 — nothing fits.")

    if args.suite in ("training", "all"):
        run_training_suite(
            max_domain=max_domain,
            warmup=args.warmup,
            measure=args.measure,
            domains=domains,
            active_settings=active_settings,
            skip_existing=args.skip_existing,
        )
    if args.suite in ("inference", "all"):
        run_inference_suite(
            max_domain=max_domain,
            warmup=args.warmup_infer,
            measure=args.measure_infer,
            domains=domains,
            active_settings=active_settings,
            skip_existing=args.skip_existing,
        )
    if args.suite in ("inference_dps", "all"):
        run_inference_dps_suite(
            max_domain=max_domain,
            warmup=args.warmup_infer,
            measure=args.measure_infer,
            domains=domains,
            active_settings=active_settings,
            skip_existing=args.skip_existing,
        )

    summary = write_summary(_RESULTS_DIR)
    print(f"summary written to {summary}", flush=True)


if __name__ == "__main__":
    main()
