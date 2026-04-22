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

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataset import TCADDatapipe  # noqa: E402


def init_stats(var_name: str, num_channels: int, device) -> Dict[str, torch.Tensor]:
    """Initialize statistics accumulators for a variable.

    Parameters
    ----------
    var_name : str
        Variable name
    num_channels : int
        Number of channels in the variable
    device : torch.device
        Device to store tensors on

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary of statistics accumulators
    """
    stats = {}
    stats[f"sum_{var_name}"] = torch.zeros(num_channels, device=device)
    stats[f"sum_{var_name}2"] = torch.zeros(num_channels, device=device)
    stats[f"min_{var_name}"] = torch.full((num_channels,), float("inf"), device=device)
    stats[f"max_{var_name}"] = torch.full((num_channels,), float("-inf"), device=device)
    stats[f"count_{var_name}"] = torch.zeros(num_channels, device=device)
    return stats


def accumulate_stats(
    stats_dict: Dict[str, torch.Tensor],
    var_name: str,
    data: torch.Tensor,
    device,
) -> None:
    """Accumulate statistics for a variable.

    Parameters
    ----------
    stats_dict : Dict[str, torch.Tensor]
        Dictionary to accumulate statistics in
    var_name : str
        Variable name
    data : torch.Tensor
        Data tensor (shape: [num_nodes, num_channels] or
        [num_edges, num_channels])
    device : torch.device
        Device
    """
    # Handle both 1D and 2D tensors
    if data.dim() == 1:
        data = data.unsqueeze(-1)

    num_channels = data.shape[-1]

    # Initialize if first time seeing this variable
    if f"sum_{var_name}" not in stats_dict:
        stats_dict.update(init_stats(var_name, num_channels, device))

    # Accumulate per-channel statistics
    for c in range(num_channels):
        channel_data = data[:, c]
        n = channel_data.numel()

        if n > 0:
            stats_dict[f"sum_{var_name}"][c] += channel_data.sum()
            stats_dict[f"sum_{var_name}2"][c] += (channel_data**2).sum()
            stats_dict[f"min_{var_name}"][c] = torch.minimum(
                stats_dict[f"min_{var_name}"][c], channel_data.min()
            )
            stats_dict[f"max_{var_name}"][c] = torch.maximum(
                stats_dict[f"max_{var_name}"][c], channel_data.max()
            )
            stats_dict[f"count_{var_name}"][c] += n


def main() -> None:
    """Compute dataset statistics (mean/std/min/max)."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics (mean/std/min/max)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the dataset directory (containing sim_* folders)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers per device",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of consecutive timesteps per sample",
    )
    args = parser.parse_args()

    # Validate dataset directory
    data_dir = Path(args.dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # General python logger
    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")
    logger0.info(f"Computing statistics for dataset: {data_dir}")

    # Build datapipe
    logger0.info("Initializing datapipe...")
    datapipe = TCADDatapipe(
        data_dir=data_dir,
        num_steps=args.num_steps,
        radius_defects=0.0,
        batch_size_per_device=args.batch_size,
        compute_connectivity=False,
        add_boundary=False,
        shuffle=False,
        num_workers=args.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    dataset = datapipe.dataset
    logger0.info(f"Dataset size: {len(dataset)} samples")
    logger0.info(f"Max new defects: {dataset.max_new_defects}")

    # Get node feature variable names from dataset
    node_variables = dataset.variables
    logger0.info(f"Node features: {', '.join(node_variables)}")

    # Initialize statistics accumulators
    node_stats: Dict[str, torch.Tensor] = {}

    # Accumulate statistics
    logger0.info("Accumulating statistics...")
    num_batches = 0

    for batch_snapshots in datapipe:
        # batch_snapshots is a list of Batch objects (one per timestep)
        for batched_graph in batch_snapshots:
            # Accumulate coordinate statistics (pos attribute)
            if batched_graph.pos is not None:
                accumulate_stats(node_stats, "coords", batched_graph.pos, dist.device)

            # Accumulate statistics for each node feature variable
            for var_name in node_variables:
                attr_name = f"defect_{var_name}"
                if hasattr(batched_graph, attr_name):
                    data = getattr(batched_graph, attr_name)
                    if data is not None:
                        accumulate_stats(node_stats, var_name, data, dist.device)

        num_batches += 1
        if num_batches % 10 == 0:
            logger0.info(f"Processed {num_batches} batches...")

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Reduce across ranks
    logger0.info("Reducing across ranks...")
    if dist.world_size > 1:
        # Build list of all variables to reduce
        all_vars = ["coords"] + node_variables

        for var_name in all_vars:
            if f"sum_{var_name}" in node_stats:
                torch.distributed.all_reduce(
                    node_stats[f"sum_{var_name}"],
                    op=torch.distributed.ReduceOp.SUM,
                )
                torch.distributed.all_reduce(
                    node_stats[f"sum_{var_name}2"],
                    op=torch.distributed.ReduceOp.SUM,
                )
                torch.distributed.all_reduce(
                    node_stats[f"min_{var_name}"],
                    op=torch.distributed.ReduceOp.MIN,
                )
                torch.distributed.all_reduce(
                    node_stats[f"max_{var_name}"],
                    op=torch.distributed.ReduceOp.MAX,
                )
                torch.distributed.all_reduce(
                    node_stats[f"count_{var_name}"],
                    op=torch.distributed.ReduceOp.SUM,
                )

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Compute final statistics on rank 0
    logger0.info("Computing final statistics...")
    if dist.rank == 0:
        final_stats = {}

        # Process coordinates (3 channels: x, y, z)
        if "sum_coords" in node_stats:
            coord_names = ["x", "y", "z"]
            for c, coord_name in enumerate(coord_names):
                count = node_stats["count_coords"][c].item()
                if count > 0:
                    mean = node_stats["sum_coords"][c].item() / count
                    mean2 = node_stats["sum_coords2"][c].item() / count
                    std = math.sqrt(max(0, mean2 - mean**2))

                    final_stats[coord_name] = {
                        "mean": mean,
                        "std": std,
                        "min": node_stats["min_coords"][c].item(),
                        "max": node_stats["max_coords"][c].item(),
                    }

        # Process node feature variables (each is 1D)
        for var_name in node_variables:
            if f"sum_{var_name}" in node_stats:
                count = node_stats[f"count_{var_name}"][0].item()
                if count > 0:
                    mean = node_stats[f"sum_{var_name}"][0].item() / count
                    mean2 = node_stats[f"sum_{var_name}2"][0].item() / count
                    std = math.sqrt(max(0, mean2 - mean**2))

                    final_stats[var_name] = {
                        "mean": mean,
                        "std": std,
                        "min": node_stats[f"min_{var_name}"][0].item(),
                        "max": node_stats[f"max_{var_name}"][0].item(),
                    }

        # Add max_new_defects to stats
        final_stats["max_new_defects"] = dataset.max_new_defects

        # Save stats.json
        out_file = data_dir / "stats.json"
        with open(out_file, "w") as f:
            json.dump(final_stats, f, indent=4)
        logger0.info(f"Statistics written to {out_file}")

    # Make sure all ranks wait for I/O completion
    if dist.world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
