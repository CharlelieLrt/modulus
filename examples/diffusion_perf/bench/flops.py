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

"""FLOP counting for MFU computation."""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


def measure_flops(
    closure: Callable[[], torch.Tensor],
    *,
    include_backward: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> int:
    """Run ``closure`` once and return the FLOPs counted by torch's flop counter.

    Set ``autocast_dtype`` to wrap the closure in
    ``torch.autocast("cuda", dtype=autocast_dtype)``. Necessary when the
    actual run uses AMP and the closure passes already-cast inputs into
    FP32-weight modules: without autocast, op-level dtype mismatches abort
    the FLOP probe.
    """

    counter = FlopCounterMode(display=False, depth=0)
    grad_ctx: contextlib.AbstractContextManager
    grad_ctx = torch.enable_grad() if include_backward else torch.no_grad()
    if autocast_dtype is not None:
        amp_ctx: contextlib.AbstractContextManager = torch.autocast(
            "cuda", dtype=autocast_dtype
        )
    else:
        amp_ctx = contextlib.nullcontext()
    with counter, grad_ctx, amp_ctx:
        out = closure()
        if include_backward:
            if not torch.is_tensor(out):
                raise TypeError(
                    "Closure must return a scalar loss tensor when "
                    "include_backward=True"
                )
            out.backward()
    return int(counter.get_total_flops())
