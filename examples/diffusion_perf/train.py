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

"""Training entry points for the diffusion perf benchmark.

Three functions: ``train_baseline``, ``train_physicsnemo`` (with cumulative
opts), ``train_physicsnemo_multidiffusion`` (full opts at once, num_patches=1).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionMSEDSMLoss,
    MultiDiffusionModel2D,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.diffusion_unets import SongUNet

from .bench import (
    BATCH_SIZE_TRAIN,
    CHANNELS,
    FIXED_DOMAIN,
    MD_POSITIONAL_EMBEDDING,
    MD_POSITIONAL_EMBEDDING_CHANNELS,
    MEASURE_STEPS,
    MemoryTracker,
    ResultBuilder,
    SongUNetAdapter,
    StepTimer,
    WARMUP_STEPS,
    count_marked_loc,
    detect_device,
    measure_flops,
    patch_shape_for,
    resolve_backbone_kwargs,
    run_with_oom_guard,
)

_THIS_FILE = Path(__file__).resolve()
_RESULTS_DIR = _THIS_FILE.parent / "results"


def _init_distributed() -> tuple[int, int, int, torch.device]:
    DistributedManager.initialize()
    dm = DistributedManager()
    return dm.rank, dm.local_rank, dm.world_size, dm.device


# ===========================================================================
# 1) train_baseline — pure-PyTorch EDM + DSM, FP32 eager
# ===========================================================================


def train_baseline(
    *,
    domain: int = FIXED_DOMAIN,
    batch_size: int = BATCH_SIZE_TRAIN,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    """Pure-PyTorch FP32 EDM training step — no framework code."""
    rank, local_rank, world_size, device = _init_distributed()
    H = W = domain
    B = batch_size
    C = CHANNELS
    backbone_kwargs = resolve_backbone_kwargs(img_resolution=H, in_channels=C)

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.train()

    sigma_data = 0.5
    P_mean, P_std = -1.2, 1.2

    def edm_training_step(x0):
        bsz = x0.shape[0]
        log_sigma = P_mean + P_std * torch.randn(bsz, device=x0.device)
        sigma = log_sigma.exp().view(bsz, 1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = x0 + sigma * noise
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma_data**2 + sigma**2).sqrt()
        c_in = 1.0 / (sigma_data**2 + sigma**2).sqrt()
        c_noise = (sigma.log() / 4).view(bsz)
        F = backbone(c_in * x_t, c_noise, None, None)
        x0_pred = c_skip * x_t + c_out * F
        w = (sigma**2 + sigma_data**2) / ((sigma * sigma_data) ** 2)
        return (w * (x0_pred - x0) ** 2).mean()

    model = (
        DistributedDataParallel(backbone, device_ids=[local_rank])
        if world_size > 1
        else backbone
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def training_step(x0):
        loss = edm_training_step(x0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # LOC-END

    return _run_and_record(
        function_name="train_baseline",
        loc_function="train_baseline",
        training_step=training_step,
        backbone_for_flops=backbone,
        H=H,
        W=W,
        B=B,
        C=C,
        domain=domain,
        opts=frozenset(),
        num_warmup=num_warmup,
        num_measured=num_measured,
        world_size=world_size,
        device=device,
        rank=rank,
        write=write,
        flops_closure=lambda: edm_training_step(torch.randn(B, C, H, W, device=device)),
        patches_per_sample=1,
        config_extras={"patch_shape": None, "num_patches": None},
    )


# ===========================================================================
# 2) train_physicsnemo — framework EDM + cumulative opt stack
# ===========================================================================


def train_physicsnemo(
    *,
    domain: int = FIXED_DOMAIN,
    optimizations: frozenset[str] = frozenset(),
    batch_size: int = BATCH_SIZE_TRAIN,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    """Training step through ``physicsnemo.diffusion`` (preconditioner + scheduler + loss)."""
    rank, local_rank, world_size, device = _init_distributed()
    H = W = domain
    B = batch_size
    C = CHANNELS
    backbone_kwargs = resolve_backbone_kwargs(
        img_resolution=H, in_channels=C, optimizations=optimizations
    )

    use_amp = "amp_bf16" in optimizations or "amp" in optimizations
    use_compile = "compile" in optimizations
    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_amp
        else nullcontext
    )
    data_dtype = torch.bfloat16 if use_amp else torch.float32

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.train()
    diffusion_model = SongUNetAdapter(backbone).to(device)
    precond = EDMPreconditioner(diffusion_model, sigma_data=0.5).to(device)
    scheduler = EDMNoiseScheduler()
    loss_fn = MSEDSMLoss(precond, scheduler)

    model = (
        DistributedDataParallel(precond, device_ids=[local_rank])
        if world_size > 1
        else precond
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def training_step(x0):
        loss = loss_fn(x0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # LOC-END

    if use_compile:
        # Compile the model first (more reliable than the loss closure under
        # fullgraph=True). Fall back to non-compiled if anything raises.
        backbone.forward = _try_compile_fullgraph(backbone.forward)

    def step_fn(x0):
        with amp_ctx():
            loss = loss_fn(x0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return _run_and_record(
        function_name="train_physicsnemo",
        loc_function="train_physicsnemo",
        training_step=step_fn,
        backbone_for_flops=backbone,
        H=H,
        W=W,
        B=B,
        C=C,
        domain=domain,
        opts=optimizations,
        num_warmup=num_warmup,
        num_measured=num_measured,
        world_size=world_size,
        device=device,
        rank=rank,
        write=write,
        flops_closure=lambda: loss_fn(
            torch.randn(B, C, H, W, device=device, dtype=data_dtype)
        ),
        patches_per_sample=1,
        config_extras={"patch_shape": None, "num_patches": None},
        data_dtype=data_dtype,
    )


# ===========================================================================
# 3) train_physicsnemo_multidiffusion — framework + MD wrap, num_patches=1
# ===========================================================================


def train_physicsnemo_multidiffusion(
    *,
    domain: int = FIXED_DOMAIN,
    optimizations: frozenset[str] = frozenset(),
    patch_shape: tuple[int, int] | None = None,
    max_domain_train: int | None = None,
    batch_size: int = BATCH_SIZE_TRAIN,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    """Training step using ``MultiDiffusionModel2D`` over a patched backbone."""
    rank, local_rank, world_size, device = _init_distributed()
    H = W = domain
    B = batch_size
    C = CHANNELS
    if patch_shape is None:
        if max_domain_train is None:
            raise ValueError("Need patch_shape or max_domain_train")
        patch_shape = patch_shape_for(domain, max_domain_train)
    Hp, Wp = patch_shape
    patches_per_sample = ((H + Hp - 1) // Hp) * ((W + Wp - 1) // Wp)
    num_patches_train = 1

    # MD backbone sees data + positional-embedding channels.
    backbone_kwargs = resolve_backbone_kwargs(
        img_resolution=Hp,
        in_channels=C + MD_POSITIONAL_EMBEDDING_CHANNELS,
        out_channels=C,
        optimizations=optimizations,
    )
    use_amp = "amp_bf16" in optimizations or "amp" in optimizations
    use_compile = "compile" in optimizations
    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_amp
        else nullcontext
    )
    data_dtype = torch.bfloat16 if use_amp else torch.float32

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.train()
    diffusion_model = SongUNetAdapter(backbone).to(device)
    precond = EDMPreconditioner(diffusion_model, sigma_data=0.5).to(device)
    md_model = MultiDiffusionModel2D(
        precond,
        global_spatial_shape=(H, W),
        positional_embedding=MD_POSITIONAL_EMBEDDING,
        channels_positional_embedding=MD_POSITIONAL_EMBEDDING_CHANNELS,
    )
    md_model.set_random_patching(patch_shape=(Hp, Wp), patch_num=num_patches_train)
    md_model = md_model.to(device)
    scheduler = EDMNoiseScheduler()
    loss_fn = MultiDiffusionMSEDSMLoss(md_model, scheduler)

    model = (
        DistributedDataParallel(md_model, device_ids=[local_rank])
        if world_size > 1
        else md_model
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def training_step(x0):
        loss = loss_fn(x0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # LOC-END

    if use_compile:
        backbone.forward = _try_compile_fullgraph(backbone.forward)

    def step_fn(x0):
        with amp_ctx():
            loss = loss_fn(x0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return _run_and_record(
        function_name="train_physicsnemo_multidiffusion",
        loc_function="train_physicsnemo_multidiffusion",
        training_step=step_fn,
        backbone_for_flops=backbone,
        H=H,
        W=W,
        B=B,
        C=C,
        domain=domain,
        opts=optimizations,
        num_warmup=num_warmup,
        num_measured=num_measured,
        world_size=world_size,
        device=device,
        rank=rank,
        write=write,
        flops_closure=lambda: loss_fn(
            torch.randn(B, C, H, W, device=device, dtype=data_dtype)
        ),
        patches_per_sample=patches_per_sample,
        config_extras={
            "patch_shape": list(patch_shape),
            "num_patches": num_patches_train,
            "patches_per_sample_global": patches_per_sample,
            "positional_embedding": MD_POSITIONAL_EMBEDDING,
            "positional_embedding_channels": MD_POSITIONAL_EMBEDDING_CHANNELS,
        },
        data_dtype=data_dtype,
    )


# ===========================================================================
# Helpers (instrumentation; NOT counted)
# ===========================================================================


def _try_compile_fullgraph(fn):
    """``torch.compile(fn, fullgraph=True)`` with graceful fallback."""
    try:
        compiled = torch.compile(fn, fullgraph=True, mode="default")
        # Force the graph-break detection now so we crash early if anything
        # wouldn't actually be a single graph.
        return compiled
    except Exception as exc:
        print(
            f"[compile] fullgraph compile failed ({exc.__class__.__name__}: {exc}); "
            f"leaving uncompiled.",
            flush=True,
        )
        return fn


def _run_and_record(
    *,
    function_name,
    loc_function,
    training_step,
    backbone_for_flops,
    H,
    W,
    B,
    C,
    domain,
    opts,
    num_warmup,
    num_measured,
    world_size,
    device,
    rank,
    write,
    flops_closure,
    patches_per_sample,
    config_extras,
    data_dtype=torch.float32,
):
    n_params = sum(p.numel() for p in backbone_for_flops.parameters())
    builder = ResultBuilder(function=function_name, output_dir=_RESULTS_DIR)
    builder.device(detect_device()).world_size(world_size).config(
        domain=[H, W],
        batch_size_per_rank=B,
        channels=C,
        optimizations=sorted(opts),
        num_steps_measured=num_measured,
        num_steps_warmup=num_warmup,
        solver=None,
        solver_steps=None,
        **config_extras,
    )

    loc_count = count_marked_loc(_THIS_FILE, function_name=loc_function)
    builder.loc(loc_count)

    flops_per_step = None
    autocast_dtype = data_dtype if data_dtype != torch.float32 else None
    if rank == 0:
        try:
            flops_per_step = measure_flops(
                flops_closure,
                include_backward=True,
                autocast_dtype=autocast_dtype,
            )
        except Exception as exc:
            print(f"[rank0] FLOP measurement skipped: {exc}", flush=True)
    builder.backbone(
        class_name=type(backbone_for_flops).__name__,
        params=n_params,
        flops_per_step=flops_per_step,
    )

    timer = StepTimer(warmup=num_warmup, measure=num_measured)
    mem = MemoryTracker()
    mem.reset()
    guard = {"oom": False, "error": None}

    try:
        with run_with_oom_guard() as guard:
            for _ in range(timer.total):
                x0 = torch.randn(B, C, H, W, device=device, dtype=data_dtype)
                timer.start()
                training_step(x0)
                timer.stop()
            torch.cuda.synchronize()
            mem.snapshot()
    except Exception as exc:
        builder.status("error", error=repr(exc))
        if rank == 0 and write:
            builder.write()
        if dist.is_initialized():
            dist.barrier()
        return builder.to_dict()

    if guard["oom"]:
        builder.status("oom", error=guard["error"])
    else:
        summary = timer.summary()
        builder.timing(summary)
        if patches_per_sample != 1:
            builder._data["results"]["patches_per_sample_global"] = patches_per_sample
            for key in (
                "samples_per_sec_per_gpu_median",
                "samples_per_sec_per_gpu_p25",
                "samples_per_sec_per_gpu_p75",
            ):
                v = builder._data["results"].get(key)
                if v is not None:
                    builder._data["results"][key + "_per_patch"] = v
                    builder._data["results"][key] = v / patches_per_sample
        builder.memory(
            mem.summary(total_memory_gb=builder.to_dict()["device"]["total_memory_gb"])
        )
        if flops_per_step:
            builder.mfu(flops_per_step=flops_per_step, world_size=world_size)

    if rank == 0 and write:
        path = builder.write()
        print(f"[rank0] wrote {path}", flush=True)
    if dist.is_initialized():
        dist.barrier()
    return builder.to_dict()


# ===========================================================================
# CLI
# ===========================================================================

_DISPATCH = {
    "train_baseline": train_baseline,
    "train_physicsnemo": train_physicsnemo,
    "train_physicsnemo_multidiffusion": train_physicsnemo_multidiffusion,
}


def _parse_opts(spec: str) -> frozenset[str]:
    if not spec or spec == "none":
        return frozenset()
    return frozenset(part.strip() for part in spec.split(",") if part.strip())


def main():
    """CLI entry point for training subprocesses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True, choices=list(_DISPATCH))
    parser.add_argument("--domain", type=int, default=FIXED_DOMAIN)
    parser.add_argument("--opts", default="none")
    parser.add_argument("--patch-shape", type=int, nargs=2, default=None)
    parser.add_argument("--max-domain-train", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_TRAIN)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the result YAML is written. Defaults to "
            "examples/diffusion_perf/results/. Calibration probes pass a "
            "subdirectory here to keep their outputs out of the sweep results."
        ),
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        global _RESULTS_DIR
        _RESULTS_DIR = Path(args.output_dir)

    fn = _DISPATCH[args.function]
    kwargs = dict(
        domain=args.domain,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_measured=args.measure,
    )
    if args.function != "train_baseline":
        kwargs["optimizations"] = _parse_opts(args.opts)
    if args.function == "train_physicsnemo_multidiffusion":
        if args.patch_shape is not None:
            kwargs["patch_shape"] = tuple(args.patch_shape)
        if args.max_domain_train is not None:
            kwargs["max_domain_train"] = args.max_domain_train

    fn(**kwargs)


if __name__ == "__main__":
    main()
