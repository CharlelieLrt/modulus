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

"""Static configuration for the diffusion perf benchmark."""

from __future__ import annotations

import re

import torch

# ---------------------------------------------------------------------------
# Backbone configuration (~80M, self-attention at the 2 deepest UNet levels,
# computed conditionally from ``img_resolution`` since a fixed
# ``attn_resolutions`` list is incorrect for arbitrary global resolutions).
# ---------------------------------------------------------------------------

BACKBONE_KWARGS: dict = {
    "model_channels": 128,
    "channel_mult": [1, 2, 2, 2, 2],
    "num_blocks": 4,
    "dropout": 0.13,
    "embedding_type": "positional",
    "encoder_type": "standard",
    "decoder_type": "standard",
}


def _attn_resolutions_for(img_resolution: int, channel_mult: list[int]) -> list[int]:
    """Self-attention resolutions covering the 2 deepest UNet levels.

    Each UNet level halves the spatial resolution. With ``len(channel_mult)``
    levels indexed ``0..N-1``, the resolutions at level ``i`` are
    ``img_resolution // 2**i``. We return the two deepest ones (levels ``N-2``
    and ``N-1``) so that self-attention always applies to the bottleneck and
    the level just above it, regardless of the global resolution.
    """
    n_levels = len(channel_mult)
    return [
        img_resolution // (2 ** (n_levels - 2)),
        img_resolution // (2 ** (n_levels - 1)),
    ]


def _apex_gn_available() -> bool:
    try:
        import apex.contrib.group_norm  # noqa: F401

        return True
    except Exception:
        return False


_HAS_APEX_GN = _apex_gn_available()


def resolve_backbone_kwargs(
    *,
    img_resolution: int,
    in_channels: int,
    out_channels: int | None = None,
    optimizations: frozenset[str] | None = None,
) -> dict:
    """Return SongUNet kwargs.

    ``out_channels`` defaults to ``in_channels`` (non-MD case). For MD with
    positional embeddings, the caller passes ``in_channels = data_C +
    pos_emb_C`` and ``out_channels = data_C`` so the backbone receives the
    concatenated input but emits only the data channels.
    """
    opts = optimizations or frozenset()
    channel_mult = BACKBONE_KWARGS["channel_mult"]
    out = {
        "img_resolution": img_resolution,
        "in_channels": in_channels,
        "out_channels": in_channels if out_channels is None else out_channels,
        "attn_resolutions": _attn_resolutions_for(img_resolution, channel_mult),
        **BACKBONE_KWARGS,
    }
    if "amp_bf16" in opts or "amp" in opts:
        out["amp_mode"] = True
    if "apex_gn" in opts and _HAS_APEX_GN:
        out["use_apex_gn"] = True
    return out


# Channels added by the multi-diffusion sinusoidal positional embedding.
MD_POSITIONAL_EMBEDDING: str = "sinusoidal"
MD_POSITIONAL_EMBEDDING_CHANNELS: int = 4


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

# Maximum global domain edge swept by ``run_sweep.py``. Override on the CLI
# with ``--max-global-domain N`` (any value below 8192 truncates the sweep
# from the top; above 8192, you must extend ``DOMAIN_SWEEP_FULL`` first so
# the new size is a power of 2 multiple).
MAX_GLOBAL_DOMAIN: int = 8192
DOMAIN_SWEEP_FULL: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
DOMAIN_SWEEP: list[int] = [d for d in DOMAIN_SWEEP_FULL if d <= MAX_GLOBAL_DOMAIN]
FIXED_DOMAIN: int = 256
CHANNELS: int = 16
BATCH_SIZE_TRAIN: int = 4
BATCH_SIZE_INFER: int = 1
# Number of GPUs per node used for DDP training in the orchestrators
# (calibrate.py, run_sweep.py): one full node per training run, so this is
# the node's GPU count (4 on GB200 / L40s, 8 on H100-SXM). Inference is
# single-GPU regardless. Cross-GPU throughput comparisons should note the
# rank count, since it shifts per-rank DDP overhead.
NPROC_PER_NODE_TRAIN: int = 4
SOLVER_STEPS: int = 18
WARMUP_STEPS: int = 6
MEASURE_STEPS: int = 15
WARMUP_STEPS_INFER: int = 3
MEASURE_STEPS_INFER: int = 5
PATCH_ALIGN: int = 16  # 5 UNet levels => 4 downsamples => multiple of 2**4
OBSERVATION_FRAC: float = 0.005
OBSERVATION_STD: float = 0.05
OBSERVATION_CHANNEL_FRAC: float = 0.5

# Single full-opt set applied at once (no cumulative sweep)
FULL_OPTS_TRAIN: frozenset[str] = frozenset({"amp_bf16", "compile", "apex_gn"})
FULL_OPTS_INFER: frozenset[str] = frozenset({"amp_bf16", "compile", "apex_gn"})


# ---------------------------------------------------------------------------
# GPU hardware
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "H100-SXM-80GB": 989.0,
    "H100-PCIe-80GB": 756.0,
    "L40s": 362.0,
    "B100": 1800.0,
    "GB200": 2500.0,
    "A100-SXM-80GB": 312.0,
    "A100-SXM-40GB": 312.0,
}
GPU_PEAK_TFLOPS_FP16 = dict(GPU_PEAK_TFLOPS_BF16)
GPU_TOTAL_MEMORY_GB: dict[str, float] = {
    "H100-SXM-80GB": 80.0,
    "H100-PCIe-80GB": 80.0,
    "L40s": 48.0,
    "B100": 192.0,
    "GB200": 192.0,
    "A100-SXM-80GB": 80.0,
    "A100-SXM-40GB": 40.0,
}

_DEVICE_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"NVIDIA H100[^\d]*SXM[^\d]*80\s*GB", re.I), "H100-SXM-80GB"),
    (re.compile(r"NVIDIA H100[^\d]*80\s*GB.*PCIe", re.I), "H100-PCIe-80GB"),
    (re.compile(r"NVIDIA H100", re.I), "H100-SXM-80GB"),
    (re.compile(r"NVIDIA L40S?", re.I), "L40s"),
    (re.compile(r"NVIDIA B100", re.I), "B100"),
    (re.compile(r"NVIDIA GB200", re.I), "GB200"),
    (re.compile(r"NVIDIA A100[^\d]*80\s*GB", re.I), "A100-SXM-80GB"),
    (re.compile(r"NVIDIA A100[^\d]*40\s*GB", re.I), "A100-SXM-40GB"),
    (re.compile(r"NVIDIA A100", re.I), "A100-SXM-40GB"),
]


def detect_device(device_idx: int | None = None) -> dict:
    """Resolve a short, stable device label + capability dict for the given
    CUDA device index (defaults to the current device).

    ``total_memory_gb`` is read directly from PyTorch in **decimal GB**
    (``total_memory / 1e9``) to match the convention used by
    ``torch.cuda.max_memory_allocated() / 1e9`` for ``peak_memory_*``.
    The vendor's marketing label (e.g. L40 "48 GB" = 48 GiB) is GiB-based,
    so the value reported here will be larger than the marketing number
    (e.g. ~51.54 GB decimal for an L40)."""
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    raw_name = torch.cuda.get_device_name(device_idx)
    short = next(
        (label for pat, label in _DEVICE_NAME_PATTERNS if pat.search(raw_name)),
        raw_name,
    )
    return {
        "name": short,
        "raw_name": raw_name,
        "bf16_peak_tflops": GPU_PEAK_TFLOPS_BF16.get(short),
        "fp16_peak_tflops": GPU_PEAK_TFLOPS_FP16.get(short),
        "total_memory_gb": torch.cuda.get_device_properties(device_idx).total_memory
        / 1e9,
        "capability": list(torch.cuda.get_device_capability(device_idx)),
        "apex_gn_available": _HAS_APEX_GN,
    }
