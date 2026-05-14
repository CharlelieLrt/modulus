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

"""Inference entry points WITHOUT guidance (plain sampling).

Three functions:
  * ``generate_baseline`` — pure-PyTorch EDM Heun, FP32.
  * ``generate_physicsnemo`` — framework ``sample()``; optimizations are direct
    variants (not cumulative): ``{fp16, compile, apex_gn}`` vs.
    ``{amp_bf16, compile, apex_gn}``, each combinable with subsets.
  * ``generate_physicsnemo_multidiffusion`` — MD predictor with chunk
    streaming + activation checkpointing; full opts applied at once.

DPS guidance is in ``generate_dps_guidance.py``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from physicsnemo.diffusion.multi_diffusion import (
    MultiDiffusionModel2D,
    MultiDiffusionPredictor,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample
from physicsnemo.models.diffusion_unets import SongUNet

from .bench import (
    BATCH_SIZE_INFER,
    CHANNELS,
    FIXED_DOMAIN,
    FULL_OPTS_INFER,
    MD_POSITIONAL_EMBEDDING,
    MD_POSITIONAL_EMBEDDING_CHANNELS,
    MEASURE_STEPS,
    MemoryTracker,
    ResultBuilder,
    SOLVER_STEPS,
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


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype_and_autocast(opts: frozenset[str]):
    """Decode the precision opts.

    Returns ``(data_dtype, autocast_dtype | None)``. The model parameters
    stay FP32 in all paths; only the data tensors and autocast dtype change.

      * ``{amp_bf16}`` — data BF16, autocast bf16 wraps the forward (matches
        the PyTorch best-practice pattern for AMP-BF16: FP32 master weights +
        BF16 compute. Allocating data in BF16 saves the activation-memory
        bloat that an FP32 input tensor would cause at large spatial sizes.)
      * empty (or other) — data FP32, no autocast.
    """
    if "amp_bf16" in opts:
        return torch.bfloat16, torch.bfloat16
    return torch.float32, None


def _try_compile_fullgraph(fn):
    try:
        return torch.compile(fn, fullgraph=True, mode="default")
    except Exception as exc:
        print(
            f"[compile] fullgraph compile failed ({exc.__class__.__name__}: {exc}); "
            f"leaving uncompiled.",
            flush=True,
        )
        return fn


# ===========================================================================
# 1) generate_baseline — pure-PyTorch EDM Heun, FP32
# ===========================================================================


def generate_baseline(
    *,
    domain: int = FIXED_DOMAIN,
    num_steps: int = SOLVER_STEPS,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    device = _device()
    H = W = domain
    B = BATCH_SIZE_INFER
    C = CHANNELS
    backbone_kwargs = resolve_backbone_kwargs(img_resolution=H, in_channels=C)

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.eval()

    sigma_data = 0.5
    sigma_min, sigma_max, rho = 0.002, 80.0, 7.0

    @torch.inference_mode()
    def edm_x0(x, sigma_scalar):
        sigma = torch.full((x.shape[0], 1, 1, 1), float(sigma_scalar), device=x.device)
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma_data**2 + sigma**2).sqrt()
        c_in = 1.0 / (sigma_data**2 + sigma**2).sqrt()
        c_noise = (sigma.log() / 4).view(x.shape[0])
        return c_skip * x + c_out * backbone(c_in * x, c_noise, None, None)

    @torch.inference_mode()
    def heun_sample(shape):
        step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
        sigmas = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)])
        x = sigma_max * torch.randn(shape, device=device)
        for i in range(num_steps):
            s_cur = sigmas[i].to(x.dtype)
            s_next = sigmas[i + 1].to(x.dtype)
            denoised = edm_x0(x, s_cur)
            d_cur = (x - denoised) / s_cur
            x_next = x + (s_next - s_cur) * d_cur
            if s_next > 0:
                denoised_next = edm_x0(x_next, s_next)
                d_next = (x_next - denoised_next) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
            x = x_next
        return x

    def sample_step():
        return heun_sample((B, C, H, W))

    # LOC-END

    return _run_and_record(
        function_name="generate_baseline",
        loc_function="generate_baseline",
        sample_step=sample_step,
        backbone_for_flops=backbone,
        domain=domain,
        opts=frozenset(),
        num_warmup=num_warmup,
        num_measured=num_measured,
        device=device,
        write=write,
        flops_closure=lambda: edm_x0(torch.randn(B, C, H, W, device=device), 1.0),
        flops_multiplier=2 * num_steps,
        config_extras={
            "patch_shape": None,
            "num_patches": None,
            "solver_steps": num_steps,
        },
        data_dtype=torch.float32,
    )


# ===========================================================================
# 2) generate_physicsnemo — framework sample(), variants of fp16 / amp_bf16
# ===========================================================================


def generate_physicsnemo(
    *,
    domain: int = FIXED_DOMAIN,
    optimizations: frozenset[str] = frozenset(),
    num_steps: int = SOLVER_STEPS,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    device = _device()
    H = W = domain
    B = BATCH_SIZE_INFER
    C = CHANNELS
    backbone_kwargs = resolve_backbone_kwargs(
        img_resolution=H, in_channels=C, optimizations=optimizations
    )
    data_dtype, autocast_dtype = _select_dtype_and_autocast(optimizations)

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.eval()
    diffusion_model = SongUNetAdapter(backbone).to(device)
    precond = EDMPreconditioner(diffusion_model, sigma_data=0.5).to(device)
    scheduler = EDMNoiseScheduler()

    @torch.inference_mode()
    def sample_step():
        xN = scheduler.init_latents(
            (C, H, W),
            tN=torch.full((B,), scheduler.sigma_max, device=device, dtype=data_dtype),
            device=device,
            dtype=data_dtype,
        )
        return sample(
            scheduler.get_denoiser(
                x0_predictor=lambda x, t: precond(x, t, condition=None)
            ),
            xN,
            scheduler,
            num_steps=num_steps,
            solver="heun",
        )

    # LOC-END

    if "compile" in optimizations:
        backbone.forward = _try_compile_fullgraph(backbone.forward)

    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=autocast_dtype))
        if autocast_dtype is not None
        else nullcontext
    )

    def step_in_amp():
        with autocast_ctx():
            return sample_step()

    return _run_and_record(
        function_name="generate_physicsnemo",
        loc_function="generate_physicsnemo",
        sample_step=step_in_amp,
        backbone_for_flops=backbone,
        domain=domain,
        opts=optimizations,
        num_warmup=num_warmup,
        num_measured=num_measured,
        device=device,
        write=write,
        flops_closure=lambda: precond(
            torch.randn(B, C, H, W, device=device, dtype=data_dtype),
            torch.full((B,), 1.0, device=device, dtype=data_dtype),
            condition=None,
        ),
        flops_multiplier=2 * num_steps,
        config_extras={
            "patch_shape": None,
            "num_patches": None,
            "solver_steps": num_steps,
            "guidance": "none",
        },
        data_dtype=data_dtype,
    )


# ===========================================================================
# 3) generate_physicsnemo_multidiffusion — MD predictor + chunk streaming
# ===========================================================================


def generate_physicsnemo_multidiffusion(
    *,
    domain: int = FIXED_DOMAIN,
    patch_shape: tuple[int, int] | None = None,
    max_domain_infer: int | None = None,
    chunk_size: int = 1,
    overlap_pix: int = 0,
    optimizations: frozenset[str] | None = None,
    num_steps: int = SOLVER_STEPS,
    num_warmup: int = WARMUP_STEPS,
    num_measured: int = MEASURE_STEPS,
    write: bool = True,
):
    if optimizations is None:
        optimizations = FULL_OPTS_INFER
    device = _device()
    H = W = domain
    B = BATCH_SIZE_INFER
    C = CHANNELS
    if patch_shape is None:
        if max_domain_infer is None:
            raise ValueError("need patch_shape or max_domain_infer")
        patch_shape = patch_shape_for(domain, max_domain_infer)
    Hp, Wp = patch_shape
    # MD backbone sees data + 4 pos-emb channels, emits data only.
    backbone_kwargs = resolve_backbone_kwargs(
        img_resolution=Hp,
        in_channels=C + MD_POSITIONAL_EMBEDDING_CHANNELS,
        out_channels=C,
        optimizations=optimizations,
    )
    data_dtype, autocast_dtype = _select_dtype_and_autocast(optimizations)

    # LOC-START
    backbone = SongUNet(**backbone_kwargs).to(device)
    backbone.eval()
    diffusion_model = SongUNetAdapter(backbone).to(device)
    precond = EDMPreconditioner(diffusion_model, sigma_data=0.5).to(device)
    md_model = MultiDiffusionModel2D(
        precond,
        global_spatial_shape=(H, W),
        positional_embedding=MD_POSITIONAL_EMBEDDING,
        channels_positional_embedding=MD_POSITIONAL_EMBEDDING_CHANNELS,
    )
    md_model = md_model.to(device)
    scheduler = EDMNoiseScheduler()

    predictor = MultiDiffusionPredictor(
        md_model,
        chunk_size=chunk_size,
        use_checkpointing=False,
    )
    predictor.set_patching(
        overlap_pix=overlap_pix,
        boundary_pix=0,
        patch_shape=(Hp, Wp),
        global_shape=(H, W),
    )

    @torch.inference_mode()
    def sample_step():
        xN = scheduler.init_latents(
            (C, H, W),
            tN=torch.full((B,), scheduler.sigma_max, device=device, dtype=data_dtype),
            device=device,
            dtype=data_dtype,
        )
        return sample(
            scheduler.get_denoiser(x0_predictor=predictor),
            xN,
            scheduler,
            num_steps=num_steps,
            solver="heun",
        )

    # LOC-END

    if "compile" in optimizations:
        backbone.forward = _try_compile_fullgraph(backbone.forward)

    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=autocast_dtype))
        if autocast_dtype is not None
        else nullcontext
    )

    def step_in_amp():
        with autocast_ctx():
            return sample_step()

    n_patches = ((H + Hp - 1) // Hp) * ((W + Wp - 1) // Wp)
    # FLOP closure bypasses precond/adapter: MD injects pos_emb via the
    # condition TensorDict, but for counting purposes we feed the backbone
    # directly with the already-concatenated (C + pos_emb_C) input it sees
    # under the hood. Preconditioner overhead is negligible (elementwise).
    return _run_and_record(
        function_name="generate_physicsnemo_multidiffusion",
        loc_function="generate_physicsnemo_multidiffusion",
        sample_step=step_in_amp,
        backbone_for_flops=backbone,
        domain=domain,
        opts=optimizations,
        num_warmup=num_warmup,
        num_measured=num_measured,
        device=device,
        write=write,
        flops_closure=lambda: backbone(
            torch.randn(
                B,
                C + MD_POSITIONAL_EMBEDDING_CHANNELS,
                Hp,
                Wp,
                device=device,
                dtype=data_dtype,
            ),
            torch.full((B,), 1.0, device=device, dtype=data_dtype),
            None,
            None,
        ),
        flops_multiplier=2 * num_steps * n_patches,
        config_extras={
            "patch_shape": list(patch_shape),
            "num_patches": int(n_patches),
            "chunk_size": chunk_size,
            "use_checkpointing": False,
            "solver_steps": num_steps,
            "guidance": "none",
            "positional_embedding": MD_POSITIONAL_EMBEDDING,
            "positional_embedding_channels": MD_POSITIONAL_EMBEDDING_CHANNELS,
        },
        data_dtype=data_dtype,
    )


# ===========================================================================
# Recorder (instrumentation; NOT counted)
# ===========================================================================


def _run_and_record(
    *,
    function_name,
    loc_function,
    sample_step,
    backbone_for_flops,
    domain,
    opts,
    num_warmup,
    num_measured,
    device,
    write,
    flops_closure,
    flops_multiplier,
    config_extras,
    data_dtype=torch.float32,
):
    B = BATCH_SIZE_INFER
    C = CHANNELS
    H = W = domain
    n_params = sum(p.numel() for p in backbone_for_flops.parameters())

    builder = ResultBuilder(function=function_name, output_dir=_RESULTS_DIR)
    builder.device(detect_device()).world_size(1).config(
        domain=[H, W],
        batch_size_per_rank=B,
        channels=C,
        optimizations=sorted(opts),
        num_steps_measured=num_measured,
        num_steps_warmup=num_warmup,
        solver="heun",
        **config_extras,
    )
    builder.loc(count_marked_loc(_THIS_FILE, function_name=loc_function))

    flops_per_step = None
    autocast_dtype = data_dtype if data_dtype != torch.float32 else None
    try:
        flops_per_call = measure_flops(
            flops_closure,
            include_backward=False,
            autocast_dtype=autocast_dtype,
        )
        flops_per_step = flops_per_call * flops_multiplier
    except Exception as exc:
        print(f"FLOP measurement skipped: {exc}", flush=True)
    builder.backbone(
        class_name=type(backbone_for_flops).__name__,
        params=n_params,
        flops_per_step=flops_per_step,
    )
    # Free any cached memory from FLOP measurement before the timing loop
    torch.cuda.empty_cache()

    timer = StepTimer(warmup=num_warmup, measure=num_measured)
    mem = MemoryTracker()
    mem.reset()
    guard = {"oom": False, "error": None}

    try:
        with run_with_oom_guard() as guard:
            for _ in range(timer.total):
                timer.start()
                sample_step()
                timer.stop()
            torch.cuda.synchronize()
            mem.snapshot()
    except Exception as exc:
        builder.status("error", error=repr(exc))
        if write:
            builder.write()
        return builder.to_dict()

    if guard["oom"]:
        builder.status("oom", error=guard["error"])
    else:
        summary = timer.summary()
        builder.timing(summary).memory(
            mem.summary(total_memory_gb=builder.to_dict()["device"]["total_memory_gb"])
        )
        if flops_per_step:
            builder.mfu(flops_per_step=flops_per_step, world_size=1)

    if write:
        path = builder.write()
        print(f"wrote {path}", flush=True)
    return builder.to_dict()


# ===========================================================================
# CLI
# ===========================================================================

_DISPATCH = {
    "generate_baseline": generate_baseline,
    "generate_physicsnemo": generate_physicsnemo,
    "generate_physicsnemo_multidiffusion": generate_physicsnemo_multidiffusion,
}


def _parse_opts(spec: str) -> frozenset[str]:
    if not spec or spec == "none":
        return frozenset()
    return frozenset(part.strip() for part in spec.split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True, choices=list(_DISPATCH))
    parser.add_argument("--domain", type=int, default=FIXED_DOMAIN)
    parser.add_argument("--opts", default="none")
    parser.add_argument("--patch-shape", type=int, nargs=2, default=None)
    parser.add_argument("--max-domain-infer", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=SOLVER_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure", type=int, default=MEASURE_STEPS)
    args = parser.parse_args()

    fn = _DISPATCH[args.function]
    kwargs = dict(
        domain=args.domain,
        num_steps=args.num_steps,
        num_warmup=args.warmup,
        num_measured=args.measure,
    )
    if args.function == "generate_physicsnemo":
        kwargs["optimizations"] = _parse_opts(args.opts)
    if args.function == "generate_physicsnemo_multidiffusion":
        kwargs["chunk_size"] = args.chunk_size
        if args.patch_shape is not None:
            kwargs["patch_shape"] = tuple(args.patch_shape)
        if args.max_domain_infer is not None:
            kwargs["max_domain_infer"] = args.max_domain_infer
        if args.opts != "none":
            kwargs["optimizations"] = _parse_opts(args.opts)

    fn(**kwargs)


if __name__ == "__main__":
    main()
