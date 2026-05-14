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

"""Profiling utilities for the diffusion perf benchmark."""

from .adapter import SongUNetAdapter
from .calibration import (
    aligned_sweep,
    find_max_domain,
    patch_shape_for,
    power_of_2_sweep,
)
from .config import (
    BACKBONE_KWARGS,
    BATCH_SIZE_INFER,
    BATCH_SIZE_TRAIN,
    CHANNELS,
    DOMAIN_SWEEP,
    FIXED_DOMAIN,
    FULL_OPTS_INFER,
    FULL_OPTS_TRAIN,
    GPU_PEAK_TFLOPS_BF16,
    GPU_PEAK_TFLOPS_FP16,
    GPU_TOTAL_MEMORY_GB,
    MD_POSITIONAL_EMBEDDING,
    MD_POSITIONAL_EMBEDDING_CHANNELS,
    MEASURE_STEPS,
    MEASURE_STEPS_INFER,
    OBSERVATION_CHANNEL_FRAC,
    OBSERVATION_FRAC,
    OBSERVATION_STD,
    PATCH_ALIGN,
    SOLVER_STEPS,
    WARMUP_STEPS,
    WARMUP_STEPS_INFER,
    detect_device,
    resolve_backbone_kwargs,
)
from .flops import measure_flops
from .loc import count_marked_loc
from .memory import MemoryTracker, run_with_oom_guard
from .results import ResultBuilder, save_yaml
from .timing import StepTimer, median_iqr

__all__ = [
    "BACKBONE_KWARGS",
    "BATCH_SIZE_INFER",
    "BATCH_SIZE_TRAIN",
    "CHANNELS",
    "DOMAIN_SWEEP",
    "FIXED_DOMAIN",
    "FULL_OPTS_INFER",
    "FULL_OPTS_TRAIN",
    "GPU_PEAK_TFLOPS_BF16",
    "GPU_PEAK_TFLOPS_FP16",
    "GPU_TOTAL_MEMORY_GB",
    "MD_POSITIONAL_EMBEDDING",
    "MD_POSITIONAL_EMBEDDING_CHANNELS",
    "MEASURE_STEPS",
    "MEASURE_STEPS_INFER",
    "MemoryTracker",
    "OBSERVATION_CHANNEL_FRAC",
    "OBSERVATION_FRAC",
    "OBSERVATION_STD",
    "PATCH_ALIGN",
    "ResultBuilder",
    "SOLVER_STEPS",
    "SongUNetAdapter",
    "StepTimer",
    "WARMUP_STEPS",
    "WARMUP_STEPS_INFER",
    "aligned_sweep",
    "count_marked_loc",
    "detect_device",
    "find_max_domain",
    "measure_flops",
    "median_iqr",
    "patch_shape_for",
    "power_of_2_sweep",
    "resolve_backbone_kwargs",
    "run_with_oom_guard",
    "save_yaml",
]
