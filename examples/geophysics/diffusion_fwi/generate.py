# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from datetime import datetime
from pathlib import Path
from functools import partial

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import wandb

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from physicsnemo.launch.logging.wandb import initialize_wandb

from datasets.dataset import EFWIDatapipe
from utils.preconditioning import edm_precond
from utils.diffusion import DiffusionAdapter, generate, stochastic_sampler
from datasets.transforms import ZscoreNormalize, Interpolate
from utils.plot import plot_prediction


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    Generate predictions using the trained diffusion FWI model.
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = PythonLogger("generate")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    # Initialize wandb: resume from training run if possible
    wandb_id = getattr(cfg.wandb, "wandb_id", None)
    if wandb_id is not None:
        rank_zero_logger.info(f"Connecting to existing wandb run: {wandb_id}")
    initialize_wandb(
        project=f"DiffusionFWI-{'Training' if wandb_id is not None else 'Generation'}",
        entity=(cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "PhysicsNeMo"),
        mode=cfg.wandb.mode,
        results_dir=cfg.io.output_dir,
        wandb_id=wandb_id,
        resume="must" if wandb_id is not None else None,
        name=f"generate-{timestamp}",
    )

    device = dist.device
    rank_zero_logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    global_seed: int = cfg.generation.global_seed
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)

    # Define random seeds and split them across ranks
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Initialize the validation dataset
    # TODO: remove distributed? (see corrdiff)
    val_dataset = EFWIDatapipe(
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="test",
        batch_size_per_device=1,
        shuffle=True,
        num_workers=cfg.generation.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        seed=global_seed,
        use_sharding=False,
    )

    # Define dataset transform
    # Zscore normalization
    stats_mean = val_dataset.get_stats("mean")
    stats_std = val_dataset.get_stats("std")
    val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    img_H, img_W = list(cfg.generation.x_resolution)

    # Interpolation to the UNet model accepted resolution
    interp_size = {var: (img_H, img_W) for var in cfg.dataset.x_vars}
    interp_size.update({var: (img_W,) for var in cfg.dataset.y_vars})
    interp_dim = {var: (-2, -1) for var in cfg.dataset.x_vars}
    interp_dim.update({var: (-1,) for var in cfg.dataset.y_vars})
    interp_mode = {var: "bilinear" for var in cfg.dataset.x_vars}
    interp_mode.update({var: "bilinear" for var in cfg.dataset.y_vars})
    val_dataset = Interpolate(
        val_dataset,
        size=interp_size,
        dim=interp_dim,
        mode=interp_mode,
    )

    # Load diffusion model
    checkpoint_path = to_absolute_path(cfg.model.checkpoint_path)
    rank_zero_logger.info(f"Loading diffusion model from {checkpoint_path}")
    try:
        diffusion_net = Module.from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        rank_zero_logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
    except Exception as e:
        rank_zero_logger.error(f"Error loading checkpoint: {e}")
        return
    diffusion_net = diffusion_net.eval().to(device)
    rank_zero_logger.info("Diffusion model loaded successfully.")
    rank_zero_logger.info(
        f"Using model {diffusion_net.__class__.__name__} "
        f"with {diffusion_net.num_parameters()} parameters."
    )
    model = DiffusionAdapter(
        model=diffusion_net,
        args_map=("x", "t", {"y": "y"}),
    )
    # EDM preconditioning wrapper
    model_fn = partial(edm_precond, model, sigma_data=0.5)

    # TODO: add missing parameters there
    sampler_fn = partial(
        stochastic_sampler,
        physics_informed=cfg.sampler.physics_informed,
    )

    output_dir = Path(to_absolute_path(cfg.io.output_dir))
    rank_zero_logger.info(f"Starting generation, saving results to {output_dir}...")
    for i, data in enumerate(val_dataset):
        # Skip samples not in specified indices
        if i * dist.world_size > cfg.generation.num_samples:
            break

        x = torch.cat(
            [data.get(var, None) for var in list(cfg.dataset.x_vars) if var in data],
            dim=1,
        )  # (1, C_x, H, W)
        y = torch.cat(
            [data.get(var, None) for var in list(cfg.dataset.y_vars) if var in data],
            dim=1,
        )  # (1, C_y, T, W)

        # Generate ensemble predictions
        x_pred_rank = generate(
            sampler_fn=sampler_fn,
            x_channels=cfg.dataset.x_channels,
            x_resolution=(img_H, img_W),
            rank_batches=rank_batches,
            cond={
                "y": y.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(
                    memory_format=torch.channels_last
                ),
            },
            device=device,
        )

        # Gather predictions to rank 0
        x_pred = gather_tensors(x_pred_rank, dist)

        # Compute statistics and metrics on rank 0
        if dist.rank == 0:
            data_pred = {
                var: x_pred[:, i : i + 1]
                for i, var in enumerate(cfg.dataset.x_vars)
                if var in data
            }
            data_true, x_mean_pred, x_std_pred = {}, {}, {}
            rmse, mae = {}, {}
            for var in data_pred.keys():
                data_true[var] = data[var] * stats_std[var] + stats_mean[var]
                data_pred[var] = data_pred[var] * stats_std[var] + stats_mean[var]
                x_mean_pred[var] = data_pred[var].mean(dim=0, keepdim=True)
                x_std_pred[var] = data_pred[var].std(dim=0, keepdim=True)
                rmse[var] = RMSE(data_pred[var], data_true[var])
                mae[var] = MAE(data_pred[var], data_true[var])
            data_input = {
                var: data[var] * stats_std[var] + stats_mean[var]
                for var in list(cfg.dataset.y_vars)
                if var in data
            }

            # Log metrics
            rank_zero_logger.info(f"Sample {i}:")
            metrics = {}
            for var in data_pred.keys():
                rank_zero_logger.info(
                    f"{var} - RMSE: {rmse[var]:.6f}, MAE: {mae[var]:.6f}"
                )
                metrics.update(
                    {
                        f"sample_{i}/{var}_rmse": rmse[var],
                        f"sample_{i}/{var}_mae": mae[var],
                    }
                )
            wandb.log(metrics)

            # Plot results
            output_path = output_dir / f"sample_{i}"
            output_path.mkdir(parents=True, exist_ok=True)
            plot_prediction(
                sample_idx=i,
                inputs=data_input,
                targets=data_true,
                predictions=data_pred,
                statistics={"mean": x_mean_pred, "std": x_std_pred},
                metrics={"rmse": rmse, "mae": mae},
                save_dir=output_path,
                sources_to_plot=3,
            )

            # Save raw numpy arrays
            output_path = output_dir / f"sample_{i}" / "numpy"
            output_path.mkdir(parents=True, exist_ok=True)
            save_data = {}
            for var in data_pred.keys():
                save_data[f"{var}_pred"] = data_pred[var].cpu().numpy()
                save_data[f"{var}_true"] = data_true[var].cpu().numpy()
                save_data[f"{var}_mean"] = x_mean_pred[var].cpu().numpy()
                save_data[f"{var}_std"] = x_std_pred[var].cpu().numpy()
                save_data[f"{var}_ensemble"] = data_pred[var].cpu().numpy()
            for var in list(cfg.dataset.y_vars):
                if var in data:
                    data_input = data[var] * stats_std[var] + stats_mean[var]
                    save_data[f"{var}"] = data_input.cpu().numpy()
            np.savez_compressed(output_path / "data.npz", **save_data)

    rank_zero_logger.success("Generation completed!")
    wandb.finish()
    return


def gather_tensors(tensor, dist):
    """
    Gather tensors from all ranks to rank 0.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to gather
    dist : DistributedManager
        The distributed manager instance

    Returns
    -------
    torch.Tensor or None
        Concatenated tensor on rank 0, None on other ranks
    """
    if dist.world_size > 1:
        if dist.rank == 0:
            gathered_tensors = [
                torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device)
                for _ in range(dist.world_size)
            ]
        else:
            gathered_tensors = None

        torch.distributed.barrier()
        torch.distributed.gather(
            tensor,
            gather_list=gathered_tensors if dist.rank == 0 else None,
            dst=0,
        )

        if dist.rank == 0:
            return torch.cat(gathered_tensors, dim=0)
        else:
            return None
    else:
        return tensor


if __name__ == "__main__":
    main()
