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

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import TCADMapsDataset  # noqa: E402
from dataset.dataset import _collate  # noqa: E402


class Welford:
    """Numerically stable running mean / variance using Welford's algorithm.

    Stores (count, mean, M2) as Python floats so we can all-reduce them across
    ranks with Chan's parallel combine formula.
    """

    def __init__(self) -> None:
        self.count: float = 0.0
        self.mean: float = 0.0
        self.M2: float = 0.0

    def update(self, batch: torch.Tensor) -> None:
        """Absorb a batch of scalar samples (any shape; flattened internally)."""
        x = batch.detach().reshape(-1)
        n_b = float(x.numel())
        if n_b == 0:
            return
        mean_b = float(x.mean().item())
        var_b = float(x.var(unbiased=False).item())
        m2_b = var_b * n_b

        if self.count == 0.0:
            self.count = n_b
            self.mean = mean_b
            self.M2 = m2_b
            return

        # Parallel combine (Chan et al., "Updating formulae and a pairwise
        # algorithm for computing sample variances")
        delta = mean_b - self.mean
        new_count = self.count + n_b
        self.mean = self.mean + delta * n_b / new_count
        self.M2 = self.M2 + m2_b + delta * delta * self.count * n_b / new_count
        self.count = new_count

    @property
    def std(self) -> float:
        """Population standard deviation derived from the running M2."""
        if self.count < 2:
            return 0.0
        return math.sqrt(self.M2 / self.count)

    def all_reduce(self) -> None:
        """Combine statistics across all distributed ranks in place."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            return

        triples = torch.tensor([self.count, self.mean, self.M2], dtype=torch.float64)
        gathered = [torch.zeros_like(triples) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, triples)

        # Reduce pairwise
        total_count, total_mean, total_m2 = 0.0, 0.0, 0.0
        for t in gathered:
            c, m, m2 = t.tolist()
            if c == 0.0:
                continue
            if total_count == 0.0:
                total_count, total_mean, total_m2 = c, m, m2
                continue
            delta = m - total_mean
            new_count = total_count + c
            total_mean = total_mean + delta * c / new_count
            total_m2 = total_m2 + m2 + delta * delta * total_count * c / new_count
            total_count = new_count

        self.count, self.mean, self.M2 = total_count, total_mean, total_m2


def main() -> None:
    """Compute dataset-wide z-score stats and dump them to JSON.

    Scans the TCAD maps dataset once, accumulating a running mean/variance
    for pooled X/Y/Z coordinates, temperature, and potential, plus the mean
    per-simulation final time (used as the ``t`` scale factor). Writes a
    JSON file consumable by :class:`TCADMapsDataset`.
    """
    p = argparse.ArgumentParser(
        description="Compute z-score normalization stats for the TCAD maps dataset."
    )
    p.add_argument("--data-dir", required=True, help="Path to maps_only/data/")
    p.add_argument("--output", default="stats.json", help="Output JSON path")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--thickness",
        default=None,
        help="Restrict stats computation to a single thickness "
        "(e.g. '2nm'). Default: use every thickness in the dataset.",
    )
    args = p.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("compute_stats")
    rank_zero = RankZeroLoggingWrapper(logger, dist)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence "no stats_file" warning
        dataset = TCADMapsDataset(
            args.data_dir, n_steps=1, stats_file=None, thickness=args.thickness
        )
    rank_zero.info(
        f"Thickness filter: {args.thickness if args.thickness is not None else '<all>'}"
    )
    rank_zero.info(f"Dataset size: {len(dataset)} samples")

    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=False,
            drop_last=False,
        )
        if dist.world_size > 1
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=False,
        drop_last=False,
    )

    coord_stats = Welford()
    temp_stats = Welford()
    pot_stats = Welford()

    for i, (batch, _meta) in enumerate(loader):
        positions = batch["positions"]  # (B, N, 3)
        variables = batch["variables"]  # (B, 1, 2, N)
        coord_stats.update(positions)
        temp_stats.update(variables[:, 0, 0])
        pot_stats.update(variables[:, 0, 1])
        if i % 100 == 0:
            rank_zero.info(
                f"Processed {i * args.batch_size * max(1, dist.world_size)} / "
                f"{len(dataset)} samples"
            )

    # Final-time statistic: mean of per-simulation terminal time
    # Every rank has the full _time_arrays dict, so no reduce needed.
    t_finals = np.array([arr[-1] for arr in dataset._time_arrays.values()])
    t_final_mean = float(np.mean(t_finals))

    # Cross-rank reduce for Welford accumulators
    coord_stats.all_reduce()
    temp_stats.all_reduce()
    pot_stats.all_reduce()

    if dist.rank == 0:
        stats = {
            "coords": {"mean": float(coord_stats.mean), "std": float(coord_stats.std)},
            "temperature": {
                "mean": float(temp_stats.mean),
                "std": float(temp_stats.std),
            },
            "potential": {"mean": float(pot_stats.mean), "std": float(pot_stats.std)},
            # t is normalized as t/t_final_mean: stored as (mean=0, std=t_final_mean)
            # so that the (x - mean)/std formula preserves t=0 -> 0.
            "t": {"mean": 0.0, "std": t_final_mean},
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        rank_zero.info(f"Wrote stats to {out_path}")
        rank_zero.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
