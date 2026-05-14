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

    1. baseline           — pure-PyTorch, FP32, no framework
    2. physicsnemo        — framework, FP32, no opts
    3. physicsnemo + opts — framework + amp_bf16 + compile + apex_gn
    4. MD + opts          — setting 3 with MultiDiffusionModel2D wrap,
                            patch_shape = min(domain, MAX_DOMAIN)

Non-MD settings stop on first OOM. MD is expected never to OOM (the patch
shape is bounded by what training already proved fits).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
    WARMUP_STEPS,
    WARMUP_STEPS_INFER,
)
from .bench.calibration import patch_shape_for
from .bench.results import write_summary
from .calibrate import load_max_domain

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DEVICE_LABEL = "L40s"
_TORCHRUN_PORT_BASE = 29800


def _opts_to_str(opts: frozenset[str]) -> str:
    return ",".join(sorted(opts)) if opts else "none"


def _opts_str(opts: frozenset[str]) -> str:
    return "-".join(sorted(opts)) if opts else "none"


def _yaml_path(
    *, function: str, domain: int, opts: frozenset[str], batch_size: int
) -> Path:
    return _RESULTS_DIR / (
        f"{function}_{_DEVICE_LABEL}_d{domain}_b{batch_size}_opt-{_opts_str(opts)}.yaml"
    )


def _read_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return yaml.safe_load(path.read_text())["results"]["status"]
    except Exception:
        return "missing"


def _run_subprocess(
    cmd: list[str], *, multi_gpu: bool = False, port_offset: int = 0
) -> int:
    env = dict(os.environ)
    env["PATH"] = (
        "/usr/local/cuda-12.8/bin:/home/horde/miniconda3/envs/pnm-dev-py3.12/bin:"
        + env.get("PATH", "")
    )
    env["LD_LIBRARY_PATH"] = (
        "/home/horde/miniconda3/envs/pnm-dev-py3.12/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if multi_gpu:
        port = _TORCHRUN_PORT_BASE + port_offset
        full = ["torchrun", "--nproc-per-node=4", f"--master-port={port}", *cmd]
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
    *, max_domain: int, warmup: int, measure: int, domains: list[int]
):
    """4 settings × ``domains`` at training B/rank DDP."""
    bs = BATCH_SIZE_TRAIN
    print(f"[training] 4-way comparison at B={bs} DDP", flush=True)
    # Track OOM per setting to avoid wasting time on configs guaranteed to fail
    state = {"baseline": False, "framework": False, "framework_opts": False}
    port = 0
    for d in domains:
        # Setting 1: baseline (pure pytorch, FP32)
        if not state["baseline"]:
            _run_subprocess(
                _train_cmd(
                    function="train_baseline",
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                port_offset=port,
            )
            port += 1
            if (
                _read_status(
                    _yaml_path(
                        function="train_baseline",
                        domain=d,
                        opts=frozenset(),
                        batch_size=bs,
                    )
                )
                != "ok"
            ):
                state["baseline"] = True
                print(f"[training] baseline OOM at d={d}", flush=True)
        # Setting 2: physicsnemo (no opts, FP32)
        if not state["framework"]:
            _run_subprocess(
                _train_cmd(
                    function="train_physicsnemo",
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                port_offset=port,
            )
            port += 1
            if (
                _read_status(
                    _yaml_path(
                        function="train_physicsnemo",
                        domain=d,
                        opts=frozenset(),
                        batch_size=bs,
                    )
                )
                != "ok"
            ):
                state["framework"] = True
                print(f"[training] physicsnemo OOM at d={d}", flush=True)
        # Setting 3: physicsnemo + full opts
        if not state["framework_opts"]:
            _run_subprocess(
                _train_cmd(
                    function="train_physicsnemo",
                    domain=d,
                    opts=FULL_OPTS_TRAIN,
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                ),
                multi_gpu=True,
                port_offset=port,
            )
            port += 1
            if (
                _read_status(
                    _yaml_path(
                        function="train_physicsnemo",
                        domain=d,
                        opts=FULL_OPTS_TRAIN,
                        batch_size=bs,
                    )
                )
                != "ok"
            ):
                state["framework_opts"] = True
                print(f"[training] physicsnemo+opts OOM at d={d}", flush=True)
        # Setting 4: MD + full opts; never OOMs (patch bounded by MAX_DOMAIN)
        if max_domain > 0:
            patch = patch_shape_for(d, max_domain)
            _run_subprocess(
                _train_cmd(
                    function="train_physicsnemo_multidiffusion",
                    domain=d,
                    opts=FULL_OPTS_TRAIN,
                    warmup=warmup,
                    measure=measure,
                    batch_size=bs,
                    patch_shape=patch,
                ),
                multi_gpu=True,
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
):
    print(f"[{label}] 4-way comparison", flush=True)
    state = {"baseline": False, "framework": False, "framework_opts": False}
    for d in domains:
        if not state["baseline"]:
            _run_subprocess(
                _infer_cmd(
                    module,
                    function=fn_baseline,
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
            )
            if (
                _read_status(
                    _yaml_path(
                        function=fn_baseline,
                        domain=d,
                        opts=frozenset(),
                        batch_size=BATCH_SIZE_INFER,
                    )
                )
                != "ok"
            ):
                state["baseline"] = True
                print(f"[{label}] baseline OOM at d={d}", flush=True)
        if not state["framework"]:
            _run_subprocess(
                _infer_cmd(
                    module,
                    function=fn_physicsnemo,
                    domain=d,
                    opts=frozenset(),
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
            )
            if (
                _read_status(
                    _yaml_path(
                        function=fn_physicsnemo,
                        domain=d,
                        opts=frozenset(),
                        batch_size=BATCH_SIZE_INFER,
                    )
                )
                != "ok"
            ):
                state["framework"] = True
                print(f"[{label}] physicsnemo OOM at d={d}", flush=True)
        if not state["framework_opts"]:
            _run_subprocess(
                _infer_cmd(
                    module,
                    function=fn_physicsnemo,
                    domain=d,
                    opts=FULL_OPTS_INFER,
                    warmup=warmup,
                    measure=measure,
                ),
                multi_gpu=False,
            )
            if (
                _read_status(
                    _yaml_path(
                        function=fn_physicsnemo,
                        domain=d,
                        opts=FULL_OPTS_INFER,
                        batch_size=BATCH_SIZE_INFER,
                    )
                )
                != "ok"
            ):
                state["framework_opts"] = True
                print(f"[{label}] physicsnemo+opts OOM at d={d}", flush=True)
        if max_domain > 0:
            patch = patch_shape_for(d, max_domain)
            _run_subprocess(
                _infer_cmd(
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
            )


def run_inference_suite(*, max_domain, warmup, measure, domains):
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
    )


def run_inference_dps_suite(*, max_domain, warmup, measure, domains):
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
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
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
            "DOMAIN_SWEEP_FULL from the top. Default: %(default)s."
        ),
    )
    args = parser.parse_args()
    domains = [d for d in DOMAIN_SWEEP_FULL if d <= args.max_global_domain]
    if not domains:
        sys.exit(
            f"[run_sweep] --max-global-domain={args.max_global_domain} "
            f"excludes every entry of DOMAIN_SWEEP_FULL={DOMAIN_SWEEP_FULL}."
        )
    print(f"[run_sweep] sweep domains = {domains}", flush=True)

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
        )
    if args.suite in ("inference", "all"):
        run_inference_suite(
            max_domain=max_domain,
            warmup=args.warmup_infer,
            measure=args.measure_infer,
            domains=domains,
        )
    if args.suite in ("inference_dps", "all"):
        run_inference_dps_suite(
            max_domain=max_domain,
            warmup=args.warmup_infer,
            measure=args.measure_infer,
            domains=domains,
        )

    summary = write_summary(_RESULTS_DIR)
    print(f"summary written to {summary}", flush=True)


if __name__ == "__main__":
    main()
