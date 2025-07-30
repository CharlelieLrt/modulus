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

import hydra
import time
import torch
import importlib.util
import datetime

import torch.nn.functional as F
import torch.fft
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import mlflow

from physicsnemo.datapipes.cae.efwi_datapipe import EFWIDatapipe
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from physicsnemo.models.geophysics.elastic_net import ElasticNet
from physicsnemo.utils.transforms import ZscoreNormalize, MinMaxNormalize
from utils.transforms import TimeFFTTransform, SpatialFourierTransform


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # General python logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging(f"launch-train-{timestamp}.log")

    # Initialize Weights & Biases
    checkpoint_dir = get_checkpoint_dir(str(cfg.io.checkpoint_dir), "elasticnet")
    if cfg.io.load_checkpoint:
        metadata = {"wandb_id": None}
        load_checkpoint(checkpoint_dir, metadata=metadata)
        wandb_id, resume = metadata["wandb_id"], "must"
        rank_zero_logger.info(f"Resuming wandb run with ID: {wandb_id}")
    else:
        wandb_id, resume = None, None
    initialize_wandb(
        project="ElasticNet-Training",
        entity=cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "PhysicsNeMo",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        results_dir=cfg.wandb.results_dir,
        wandb_id=wandb_id,
        resume=resume,
        save_code=True,
    )

    # Initialize MLflow
    initialize_mlflow(
        experiment_name="Modulus-Launch",
        experiment_desc="ElasticNet Training",
        run_desc="ElasticNet Training run",
        user_name="PhysicsNeMo User",
        mode="offline" if cfg.wandb.mode == "offline" else "online",
    )
    LaunchLogger.initialize(use_mlflow=True)

    # Log parameters to MLflow
    if dist.rank == 0:
        mlflow.log_params(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    # Preprocess parameters for normalization
    normalization = cfg.dataset.transform.normalize or "ZscoreNormalize"

    # Preprocess parameters for frequency domain features
    nb_sources = cfg.dataset.nb_sources
    time_featurization = cfg.dataset.transform.temporal_fft
    apply_fft = time_featurization in ("replace", "concatenate")
    concat_fft = time_featurization == "concatenate"
    if time_featurization == "replace":
        nb_sources *= 2
    elif time_featurization == "concatenate":
        nb_sources += 2 * nb_sources

    # Preprocess parameters for spatial Fourier features
    fourier_embedding = cfg.dataset.transform.fourier_embedding
    nb_frequencies = cfg.dataset.transform.nb_frequencies
    include_input = cfg.dataset.transform.include_input
    emb_channels = 4 * nb_frequencies + (2 if include_input else 0)
    if fourier_embedding is not None:
        nb_sources += emb_channels

    # Add learnable input embeddings
    learnable_embeddings = cfg.model.learnable_embeddings or 0

    # Determine activation function
    activation_name = getattr(cfg.model, "activation", None)

    # Determine temporal encoder
    if cfg.model.fno_temporal_encoder is not None:
        fno_temporal_encoder = OmegaConf.to_container(cfg.model.fno_temporal_encoder)
    else:
        fno_temporal_encoder = None

    # Instantiate model
    model = ElasticNet(
        nb_sources=nb_sources,
        nb_timesteps=cfg.dataset.nb_timesteps,
        nb_receivers=cfg.dataset.nb_receivers,
        initial_channels=cfg.model.initial_channels,
        limit_shape=cfg.model.limit_shape,
        output_shape=list(cfg.dataset.subsurface_resolution),
        checkpointing_level=cfg.model.checkpointing_level,
        learnable_embeddings=learnable_embeddings,
        activation=activation_name,
        fno_temporal_encoder=fno_temporal_encoder,
    ).to(dist.device)

    rank_zero_logger.info(
        f"Using model ElasticNet with {model.num_parameters()} parameters."
    )

    # Weighted loss function
    def weighted_loss(pred, target):
        """
        Compute a weighted combination of L1 and L2 losses.
        """
        return cfg.training.weight_l1 * F.l1_loss(
            pred, target
        ) + cfg.training.weight_l2 * F.mse_loss(pred, target)

    # Distributed learning (Data parallel)
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Initialize the training dataset
    train_dataset = EFWIDatapipe(
        name=cfg.dataset.name,
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="train",
        batch_size_per_device=cfg.training.batch_size_per_device,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # Apply dataset transforms
    rank_zero_logger.info(f"Using normalize transform: {normalization}")
    if normalization == "MinMaxNormalize":
        stats_min = train_dataset.get_stats("min")
        stats_max = train_dataset.get_stats("max")
        train_dataset = MinMaxNormalize(train_dataset, stats_min, stats_max)
    else:
        stats_mean = train_dataset.get_stats("mean")
        stats_std = train_dataset.get_stats("std")
        train_dataset = ZscoreNormalize(train_dataset, stats_mean, stats_std)
    if apply_fft:
        mode_str = "concatenate" if concat_fft else "replace"
        rank_zero_logger.info(f"Applying temporal FFT transform (mode={mode_str})")
        train_dataset = TimeFFTTransform(
            train_dataset,
            concat=concat_fft,
            nb_modes=cfg.dataset.transform.nb_modes,
        )

    # Apply Fourier embedding transform
    if fourier_embedding is not None:
        rank_zero_logger.info(
            f"Applying Fourier embedding transform: nb_frequencies={nb_frequencies}, "
            f"include_input={include_input})"
        )
        train_dataset = SpatialFourierTransform(
            train_dataset,
            num_frequencies=nb_frequencies,
            include_input=include_input,
        )

    # Initialize the validation dataset
    val_dataset = EFWIDatapipe(
        name=cfg.dataset.name,
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="test",
        batch_size_per_device=cfg.val.batch_size_per_device,
        shuffle=True,
        num_workers=cfg.val.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    if normalization == "MinMaxNormalize":
        val_dataset = MinMaxNormalize(val_dataset, stats_min, stats_max)
    else:
        val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    if apply_fft:
        val_dataset = TimeFFTTransform(
            val_dataset,
            concat=concat_fft,
            nb_modes=cfg.dataset.transform.nb_modes,
        )
    if fourier_embedding is not None:
        val_dataset = SpatialFourierTransform(
            val_dataset,
            num_frequencies=nb_frequencies,
            include_input=include_input,
        )

    # Create optimizer: use FusedAdam if available
    optimizer_class = None
    if torch.cuda.is_available():
        try:
            optimizer_class = getattr(
                importlib.import_module("apex.optimizers"), "FusedAdam"
            )
            use_FusedAdam = True
        except ImportError:
            pass
    if optimizer_class is None:
        optimizer_class = torch.optim.AdamW
        use_FusedAdam = False
        rank_zero_logger.info("Using AdamW optimizer")
    else:
        rank_zero_logger.info("Using FusedAdam optimizer")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu",
        enabled=cfg.training.amp,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg.training.scheduler.factor,
        patience=cfg.training.scheduler.patience,
    )

    # Load checkpoint if it exists or if explicitly requested
    loaded_epoch, total_samples_trained = 0, 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        metadata = {"total_samples_trained": total_samples_trained}
        loaded_epoch = load_checkpoint(
            checkpoint_dir,
            models=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=dist.device,
            metadata=metadata,
        )
        total_samples_trained = metadata["total_samples_trained"]

    # Log initial learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    rank_zero_logger.info(f"Starting learning rate: {current_lr}")
    if dist.rank == 0:
        wandb.log({"lr": current_lr, "epoch": loaded_epoch})

    # Training loop
    rank_zero_logger.info("Training started...")
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_epochs + 1):
        model.train()
        epoch_loss, epoch_samples = 0.0, 0
        time_start = time.time()
        train_dataset.set_epoch(epoch)

        # Use LaunchLogger for training
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_dataset), epoch_alert_freq=1
        ) as launchlog:
            for i, data in enumerate(train_dataset):
                ux, uz = data["ux"], data["uz"]
                vp_target, vs_target = data["vp"], data["vs"]
                batch_size = ux.shape[0]
                epoch_samples += batch_size

                optimizer.zero_grad(**({} if use_FusedAdam else {"set_to_none": True}))

                # Forward pass with AMP
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=cfg.training.amp,
                ):
                    pred = model(ux, uz)
                    target = torch.cat([vp_target, vs_target], dim=1)
                    loss = weighted_loss(pred, target)
                epoch_loss += loss.item() * batch_size

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Log mini-batch metrics
                current_lr = optimizer.param_groups[0]["lr"]
                batch_metrics = {"batch_loss": loss.item(), "lr": current_lr}
                launchlog.log_minibatch(batch_metrics)
                if dist.rank == 0:
                    wandb.log(batch_metrics)
                rank_zero_logger.info(
                    f"lr: {current_lr}, batch: {i}, batch loss: {loss.item()}"
                )

            # Compute mean loss for the epoch
            mean_loss, epoch_samples_all_ranks = average_loss(
                dist, epoch_loss, epoch_samples
            )
            time_end = time.time()
            total_samples_trained += epoch_samples_all_ranks

            # Log epoch metrics
            metrics = {
                "mean_loss": mean_loss,
                "time_per_epoch": time_end - time_start,
                "lr": current_lr,
                "epoch": epoch,
                "total_samples_trained": total_samples_trained,
                "epoch_samples": epoch_samples_all_ranks,
            }
            launchlog.log_epoch(metrics)
            if dist.rank == 0:
                wandb.log(metrics)
            msg = f"epoch: {epoch}, mean loss: {mean_loss:10.3e}"
            msg += f", time per epoch: {(time_end - time_start):10.3e}"
            msg += f", total samples: {total_samples_trained}"
            rank_zero_logger.info(msg)

        # Synchronize processes before validation
        if dist.world_size > 1:
            torch.distributed.barrier()

        # Run validation with LaunchLogger
        with LaunchLogger("valid", epoch=epoch) as launchlog:
            model.eval()
            mean_val_loss = validation_step(
                model,
                val_dataset,
                weighted_loss,
                dist,
            )

            # Log validation metrics
            val_metrics = {
                "val_loss": mean_val_loss,
                "epoch": epoch,
                "total_samples_trained": total_samples_trained,
            }
            launchlog.log_epoch(val_metrics)
            if dist.rank == 0:
                wandb.log(val_metrics)
            rank_zero_logger.info(f"epoch: {epoch}, val loss: {mean_val_loss}")

        # Adjust learning rate based on validation loss
        scheduler.step(mean_val_loss)

        # Save checkpoint periodically
        if dist.world_size > 1:
            torch.distributed.barrier()
        if epoch % cfg.io.checkpoint_freq == 0 and dist.rank == 0:
            save_checkpoint(
                checkpoint_dir,
                models=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                metadata={
                    "total_samples_trained": total_samples_trained,
                    "wandb_id": wandb.run.id,
                },
            )
            rank_zero_logger.info(f"Saved checkpoint at epoch {epoch}")

    # Finish logging
    wandb.finish()
    if dist.rank == 0:
        mlflow.end_run()
    rank_zero_logger.info("Training completed!")


@torch.no_grad()
def validation_step(model, dataset, criterion, dist):
    """
    Perform validation on a dataset and return the average loss.
    """
    loss_epoch = 0.0
    num_samples = 0.0

    for i, data in enumerate(dataset):
        ux, uz = data["ux"], data["uz"]
        vp_target, vs_target = data["vp"], data["vs"]
        pred = model(ux, uz)
        target = torch.cat([vp_target, vs_target], dim=1)
        loss = criterion(pred, target)
        loss_epoch += loss.item() * ux.shape[0]
        num_samples += ux.shape[0]

    # Average validation loss across all ranks
    mean_val_loss, num_samples_all_ranks = average_loss(dist, loss_epoch, num_samples)

    return mean_val_loss


def average_loss(dist, loss_value: float, sample_count: int) -> tuple[float, int]:
    """
    Average the loss value over all ranks.
    """
    if dist.world_size > 1:
        tensor = torch.tensor([loss_value, float(sample_count)], device=dist.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor[0].item() / tensor[1].item(), tensor[1].item()
    else:
        return (loss_value / sample_count), sample_count


if __name__ == "__main__":
    main()
