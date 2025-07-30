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

from typing import Dict, Optional, Tuple
from datetime import datetime

from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np

from physicsnemo.datapipes.cae.efwi_datapipe import EFWIDatapipe
from physicsnemo.utils.transforms import ZscoreNormalize, MinMaxNormalize
from physicsnemo import Module
from physicsnemo.launch.logging import (
    PythonLogger,
    LaunchLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.distributed import DistributedManager

from utils.plotting import (
    plot_seismic_data,
    plot_velocity_model,
    plot_predictions,
    plot_residuals_histograms,
)
from utils.transforms import TimeFFTTransform


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Generic Python logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = PythonLogger("generate")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging(f"launch-generate-{timestamp}.log")

    # Initialize wandb: force resume from training run
    if not hasattr(cfg.logging.wandb, "wandb_id") or cfg.logging.wandb.wandb_id is None:
        raise ValueError(
            "No wandb ID provided in config. "
            "Please specify cfg.wandb.wandb_id to connect to an existing run."
        )
    wandb_id = cfg.logging.wandb.wandb_id
    rank_zero_logger.info(f"Connecting to existing wandb run: {wandb_id}")
    initialize_wandb(
        project="ElasticNet-Training",
        entity=(
            cfg.logging.wandb.entity
            if hasattr(cfg.logging.wandb, "entity")
            else "PhysicsNeMo"
        ),
        mode=(
            cfg.logging.wandb.mode if hasattr(cfg.logging.wandb, "mode") else "offline"
        ),
        wandb_id=wandb_id,
        resume="must",
        results_dir=cfg.io.output_dir,
        name=f"generate-{timestamp}",
        group=f"DDP_Group_{timestamp}" if dist.world_size > 1 else None,
    )
    LaunchLogger.initialize(use_wandb=True)

    # Load model from checkpoint
    try:
        checkpoint_path = to_absolute_path(cfg.io.checkpoint_dir)
        rank_zero_logger.info(f"Loading checkpoint from {checkpoint_path}")
        model = Module.from_checkpoint(checkpoint_path)
        model = model.eval().to(dist.device)
        rank_zero_logger.success("Model loaded successfully")
    except FileNotFoundError:
        rank_zero_logger.error(f"Checkpoint not found at {cfg.io.checkpoint_dir}")
        return
    except Exception as e:
        rank_zero_logger.error(f"Error loading checkpoint: {e}")
        return

    # Instantiate the validation dataset
    val_dataset = EFWIDatapipe(
        name=cfg.dataset.name,
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="test",
        batch_size_per_device=cfg.generate.batch_size,
        shuffle=False,  # No need to shuffle for generation
        num_workers=cfg.generate.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # Handle dataset transforms parameters
    transform_cfg = cfg.dataset.transform
    if hasattr(transform_cfg, "normalize"):
        normalize_name = transform_cfg.normalize
    else:
        normalize_name = "ZscoreNormalize"
    if hasattr(transform_cfg, "temporal_fft"):
        temporal_fft = transform_cfg.temporal_fft
    else:
        temporal_fft = None
    if hasattr(transform_cfg, "nb_modes"):
        nb_fft_modes = transform_cfg.nb_modes
    else:
        nb_fft_modes = None
    apply_fft = temporal_fft in ("replace", "concatenate")
    concat_fft = temporal_fft == "concatenate"
    if normalize_name == "MinMaxNormalize":
        stats_min = val_dataset.get_stats("min")
        stats_max = val_dataset.get_stats("max")
        val_dataset = MinMaxNormalize(val_dataset, stats_min, stats_max)
    else:
        stats_mean = val_dataset.get_stats("mean")
        stats_std = val_dataset.get_stats("std")
        val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    rank_zero_logger.info(f"Normalize transform: {normalize_name}")
    if apply_fft:
        mode_str = "concatenate" if concat_fft else "replace"
        rank_zero_logger.info(
            f"Applying temporal FFT transform (mode={mode_str}, "
            f"nb_modes={nb_fft_modes})"
        )
        val_dataset = TimeFFTTransform(
            val_dataset,
            concat=concat_fft,
            nb_modes=nb_fft_modes,
        )

    figs_to_show = [] if bool(cfg.logging.show_figures) else None

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            # Process only specified samples
            if i > max(cfg.generate.sample_indices):
                break
            if i not in cfg.generate.sample_indices:
                continue

            ux, uz = data["ux"], data["uz"]
            vp_target, vs_target = data["vp"], data["vs"]
            pred = model(ux, uz)
            vp_pred = pred[:, 0:1, :, :]
            vs_pred = pred[:, 1:2, :, :]
            vp_rmse = RMSE(vp_pred, vp_target)
            vp_mae = MAE(vp_pred, vp_target)
            vs_rmse = RMSE(vs_pred, vs_target)
            vs_mae = MAE(vs_pred, vs_target)

            rank_zero_logger.info(f"Sample {i}:")
            rank_zero_logger.info(f"  VP - RMSE: {vp_rmse:.6f}, MAE: {vp_mae:.6f}")
            rank_zero_logger.info(f"  VS - RMSE: {vs_rmse:.6f}, MAE: {vs_mae:.6f}")
            metrics = {
                f"sample_{i}/vp_rmse": vp_rmse,
                f"sample_{i}/vp_mae": vp_mae,
                f"sample_{i}/vs_rmse": vs_rmse,
                f"sample_{i}/vs_mae": vs_mae,
            }
            wandb.log(metrics)

            # Make plots
            inputs = {"ux": ux, "uz": uz}
            targets = {"vp": vp_target, "vs": vs_target}
            predictions = {"vp": vp_pred, "vs": vs_pred}
            # Note: for input channels, select at most 6 sources to plot
            sources_to_plot = min(6, ux.shape[1])
            with LaunchLogger("generate") as launchlog:
                figs = make_plots(
                    inputs,
                    targets,
                    predictions,
                    sources_to_plot,
                    sample_idx=i,
                    return_figs=True,
                )
                for fig_name, (fig, data) in figs.items():
                    if data is None:
                        # Log to local file and wandb
                        launchlog.log_figure(
                            fig,
                            artifact_file=f"sample_{i}/{fig_name}",
                            plot_dir=cfg.io.output_dir,
                            log_to_file=True,
                            log_to_backend=True,
                        )
                    # If additional data is present, use it for custom wandb logging
                    else:
                        for key, value in data.items():
                            if key.startswith("histogram"):
                                wandb.log(
                                    {
                                        f"sample_{i}/{key}": wandb.Histogram(
                                            np_histogram=value
                                        )
                                    }
                                )
                        # Save figure locally only
                        launchlog.log_figure(
                            fig,
                            artifact_file=f"sample_{i}/{fig_name}",
                            plot_dir=cfg.io.output_dir,
                            log_to_file=True,
                            log_to_backend=False,
                        )

                if figs_to_show is not None:
                    figs_to_show.extend([fd[0] for fd in figs.values()])
                else:
                    for fig, _ in figs.values():
                        plt.close(fig)

    # Display figures after processing all samples
    if figs_to_show:
        plt.show(block=True)

    rank_zero_logger.info("Generation completed successfully")
    wandb.finish()


def make_plots(
    inputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    sources_to_plot: int,
    sample_idx: int,
    return_figs: bool = False,
) -> Optional[Dict[str, Tuple[plt.Figure, Optional[Dict]]]]:
    """
    Plot predictions vs ground truth for vp, vs, and inputs ux, uz.
    Creates 4 figures:
        • seismic data (inputs)
        • ground truth velocity model
        • predictions and residuals
        • residual histograms

    Parameters
    ----------
    - inputs : Dict[str, torch.Tensor]
        Dictionary containing 'ux' and 'uz' tensors with shape
        [batch_size, nb_sources, nb_timesteps, nb_receivers]
    - targets : Dict[str, torch.Tensor]
        Dictionary containing 'vp' and 'vs' tensors (ground truth) with
        shape [batch_size, 1, output_shape[0], output_shape[1]]
    - predictions : Dict[str, torch.Tensor]
        Dictionary containing 'vp' and 'vs' tensors (model predictions) with
        shape [batch_size, 1, output_shape[0], output_shape[1]]
    - sources_to_plot : int
        Number of source channels to plot from each input (ux, uz)
    - sample_idx : int
        Index of the current sample
    - return_figs : bool, optional
        If True, return the figures instead of closing them, by default False

    Returns
    -------
    - Optional[Dict[str, Tuple[plt.Figure, Optional[Dict]]]]
        Dictionary mapping figure names to (figure, data) tuples when
        return_figs is True. ``data`` is ``None`` for all plots except
        "residual_histograms", where it contains two separate entries:
        "histogram_residuals_vp" and "histogram_residuals_vs".
    """
    figures = {}

    # Get inputs, targets, and predictions
    ux = inputs["ux"][0].cpu().numpy()  # shape: [nb_sources, height, width]
    uz = inputs["uz"][0].cpu().numpy()
    vp_true = targets["vp"][0].cpu().numpy()
    vs_true = targets["vs"][0].cpu().numpy()
    vp_pred = predictions["vp"][0].cpu().numpy()
    vs_pred = predictions["vs"][0].cpu().numpy()

    # Calculate min/max values for consistent color scales
    vp_vmin = min(np.min(vp_true), np.min(vp_pred))
    vp_vmax = max(np.max(vp_true), np.max(vp_pred))
    vs_vmin = min(np.min(vs_true), np.min(vs_pred))
    vs_vmax = max(np.max(vs_true), np.max(vs_pred))

    # Figures for seismic data (inputs)
    fig_inputs = plot_seismic_data(
        ux,
        uz,
        sources_to_plot,
        return_figs,
    )
    if fig_inputs is not None and return_figs:
        figures["inputs"] = fig_inputs

    # Figures for velocity model (ground truth)
    fig_gt = plot_velocity_model(
        vp_true,
        vs_true,
        vp_vmin,
        vp_vmax,
        vs_vmin,
        vs_vmax,
        return_figs,
    )
    if fig_gt is not None and return_figs:
        figures["ground_truth"] = fig_gt

    # Figures for predictions and residuals
    fig_pred = plot_predictions(
        vp_true,
        vs_true,
        vp_pred,
        vs_pred,
        vp_vmin,
        vp_vmax,
        vs_vmin,
        vs_vmax,
        return_figs,
    )
    if fig_pred is not None and return_figs:
        figures["predictions_residuals"] = fig_pred

    # Figures for residual histograms
    fig_hist = plot_residuals_histograms(
        vp_true - vp_pred,
        vs_true - vs_pred,
        return_figs,
    )
    if fig_hist is not None and return_figs:
        figures["residual_histograms"] = fig_hist

    # Add suptitles to figures
    if "inputs" in figures:
        figures["inputs"][0].suptitle(f"Sample {sample_idx}: seismic data")
    if "ground_truth" in figures:
        figures["ground_truth"][0].suptitle(
            f"Sample {sample_idx}: ground truth velocity model"
        )
    if "predictions_residuals" in figures:
        figures["predictions_residuals"][0].suptitle(
            f"Sample {sample_idx}: predictions"
        )
    if "residual_histograms" in figures:
        figures["residual_histograms"][0].suptitle(
            f"Sample {sample_idx}: residual histograms"
        )

    return figures if return_figs else None


if __name__ == "__main__":
    main()
