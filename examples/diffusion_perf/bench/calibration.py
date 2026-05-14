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

"""Patch-size calibration.

Finds the largest power-of-2 global domain (``MAX_DOMAIN``) that fits in
non-multi-diffusion mode. The multi-diffusion patch shape is then chosen as
``min(current_global_domain, MAX_DOMAIN)``, aligned down to a multiple of 16
(SongUNet requires multiples of 2^(N_levels - 1) = 16).

``MAX_DOMAIN`` is a power of 2, but the resulting patch shape need not be
since ``min(global, MAX_DOMAIN)`` can be any multiple of 16.
"""

from __future__ import annotations

from .config import PATCH_ALIGN


def _next_aligned(x: int, align: int = PATCH_ALIGN) -> int:
    """Round ``x`` down to the nearest positive multiple of ``align``."""
    if x < align:
        return align
    return (x // align) * align


def patch_shape_for(global_domain: int, max_domain: int) -> tuple[int, int]:
    """Return the patch shape for a multi-diffusion run.

    ``min(global_domain, max_domain)`` is aligned to the nearest multiple of
    ``PATCH_ALIGN`` to satisfy the backbone's downsampling constraint.
    """
    raw = min(global_domain, max_domain)
    p = _next_aligned(raw)
    return p, p


def power_of_2_sweep(start: int = 64, cap: int = 16384) -> list[int]:
    """Generate the doubling sequence used to find MAX_DOMAIN."""
    out = []
    d = start
    while d <= cap:
        out.append(d)
        d *= 2
    return out


def find_max_domain(
    probe_fn,
    *,
    start: int = 64,
    cap: int = 8192,
) -> int:
    """Probe ``probe_fn(domain) -> bool`` (True = fits) by doubling.

    Returns the largest power-of-2 domain that returns True.
    """
    last_fit = 0
    for d in power_of_2_sweep(start=start, cap=cap):
        if probe_fn(d):
            last_fit = d
        else:
            break
    return last_fit


def aligned_sweep(start: int, cap: int, align: int = PATCH_ALIGN) -> list[int]:
    """Generate multiples of ``align`` from ``start`` up to ``cap``."""
    out = []
    d = ((start + align - 1) // align) * align
    while d <= cap:
        out.append(d)
        d += align
    return out
