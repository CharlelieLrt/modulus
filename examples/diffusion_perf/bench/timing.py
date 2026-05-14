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

"""CUDA-event timing with warmup, median + IQR statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    """Return (median, p25, p75) for a list of floats."""

    if not values:
        return float("nan"), float("nan"), float("nan")
    t = torch.tensor(values, dtype=torch.float64)
    return (
        float(t.median().item()),
        float(t.quantile(0.25).item()),
        float(t.quantile(0.75).item()),
    )


@dataclass
class StepTimer:
    """Records per-step wall time via CUDA events.

    Use as:

        timer = StepTimer(warmup=10, measure=50)
        for _ in range(timer.total):
            timer.start()
            ...                       # one training/inference step
            timer.stop()
        result = timer.summary()

    The first ``warmup`` recorded steps are discarded. Only ``measure`` steps
    contribute to the summary.
    """

    warmup: int = 10
    measure: int = 50
    _events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    _idx: int = 0
    _times_ms: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.warmup + self.measure

    def start(self) -> None:
        e_start = torch.cuda.Event(enable_timing=True)
        e_stop = torch.cuda.Event(enable_timing=True)
        e_start.record()
        self._events.append((e_start, e_stop))

    def stop(self) -> None:
        self._events[-1][1].record()
        self._idx += 1

    def _materialize(self) -> list[float]:
        if self._times_ms:
            return self._times_ms
        torch.cuda.synchronize()
        for e_start, e_stop in self._events:
            self._times_ms.append(e_start.elapsed_time(e_stop))
        return self._times_ms

    def summary(self) -> dict:
        all_times_ms = self._materialize()
        measured = all_times_ms[self.warmup : self.warmup + self.measure]
        median_ms, p25_ms, p75_ms = median_iqr(measured)
        return {
            "step_time_ms_median": median_ms,
            "step_time_ms_p25": p25_ms,
            "step_time_ms_p75": p75_ms,
            "warmup_step_times_ms": all_times_ms[: self.warmup],
            "measured_step_times_ms": measured,
            "num_measured": len(measured),
        }
