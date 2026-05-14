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

"""Peak GPU memory tracking and OOM-guarded execution."""

from __future__ import annotations

import gc
from contextlib import contextmanager

import torch


class MemoryTracker:
    """Tracks peak allocated memory across a measured window."""

    def __init__(self) -> None:
        self.peak_bytes: int = 0
        self.snapshot_bytes: int = 0

    def reset(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.peak_bytes = 0
        self.snapshot_bytes = 0

    def snapshot(self) -> None:
        self.peak_bytes = torch.cuda.max_memory_allocated()
        self.snapshot_bytes = torch.cuda.memory_allocated()

    def summary(self, total_memory_gb: float | None = None) -> dict:
        peak_gb = self.peak_bytes / 1e9
        util = (peak_gb / total_memory_gb) if total_memory_gb else None
        return {
            "peak_memory_allocated_gb": peak_gb,
            "peak_memory_utilization": util,
            "current_allocated_gb": self.snapshot_bytes / 1e9,
        }


@contextmanager
def run_with_oom_guard():
    """Yields a dict ``{"oom": False}`` and flips ``oom`` to True on OOM.

    Use as:

        guard = {"oom": False}
        try:
            with run_with_oom_guard() as guard:
                ...
        finally:
            pass
        if guard["oom"]: ...
    """

    flag = {"oom": False, "error": None}
    try:
        yield flag
    except torch.cuda.OutOfMemoryError as exc:
        flag["oom"] = True
        flag["error"] = str(exc)
        gc.collect()
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda error" in msg and "memory" in msg:
            flag["oom"] = True
            flag["error"] = str(exc)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            raise
