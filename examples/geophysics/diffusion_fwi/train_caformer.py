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
import torch
import wandb
import importlib.util
import time
import datetime
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from functools import partial
import mlflow
import numpy as np
from physicsnemo.datapipes.cae.efwi_datapipe_combined import EFWIDatapipe
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from physicsnemo.models.geophysics.diffusion_improved import DiffusionPIO
from physicsnemo.models.diffusion import edm_precond
from physicsnemo.metrics.diffusion import NoResidualLoss
from physicsnemo.models.diffusion.conditional import ConditionalDiffusionAdapter
from physicsnemo.utils.transforms import MinMaxNormalize, Normalize
# from physicsnemo import Module


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # General python logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging(f"launch-{timestamp}.log")

    # Initialize Weights & Biases
    checkpoint_dir = get_checkpoint_dir(str(cfg.io.checkpoint_dir), "caformer_noflip")
    if cfg.io.load_checkpoint:
        metadata = {"wandb_id": None}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        wandb_id, resume = metadata["wandb_id"], "must"
        rank_zero_logger.info(f"Resuming wandb run with ID: {wandb_id}")
    else:
        wandb_id, resume = None, None
    initialize_wandb(
        project="DiffusionCaformer-Training",
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
        experiment_desc="Diffusion Caformer Training",
        run_desc="Diffusion Caformer Training run",
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


    # Initialize diffusion model
    model_args = {}
    if hasattr(cfg.model, "model_args") and cfg.model.model_args is not None:
        model_args = OmegaConf.to_container(cfg.model.model_args)
        rank_zero_logger.info(f"Using model configuration: {model_args}")
    conditioning_model_kwargs = {}
    if (
        hasattr(cfg.model, "conditioning_model_kwargs")
        and cfg.model.conditioning_model_kwargs is not None
    ):
        conditioning_model_kwargs = OmegaConf.to_container(
            cfg.model.conditioning_model_kwargs
        )
        rank_zero_logger.info(
            f"Using conditioning model configuration: {conditioning_model_kwargs}"
        )

    model_backbone = DiffusionPIO(
        nb_sources=cfg.dataset.nb_sources,
        nb_timesteps=cfg.dataset.nb_timesteps,
        nb_receivers=cfg.dataset.nb_receivers,
        img_resolution=list(cfg.dataset.subsurface_resolution),
        x_mean_conditioning=False,
        state_channels=3,  # vp, vs, rho
        unet_kwargs=model_args,
        perceiver_args = conditioning_model_kwargs
    ).to(dist.device)
    rank_zero_logger.info(f"Parameters : {sum(p.numel() for p in model_backbone.parameters())}")
    # Thin wrapper around the model_backbone to convert it into a conditional
    # diffusion model compatible with EDM preconditioning and ResidualLoss
    model = ConditionalDiffusionAdapter(
        model=model_backbone,
        args_map=("x", {"x_mean": "x_mean", "noise": "noise", "ux": "ux", "uz": "uz"}),
    )

    # Distributed learning (Data parallel)
    if dist.world_size > 1:
        # Wrap the conditional model in DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # EDM preconditioning wrapper
    model_fn = partial(edm_precond, model=model, sigma_data=0.5)

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
        folder_name = "samples_new"
    )

    # Define transforms (default to Normalize)
    transform_name = (
        cfg.dataset.transform if hasattr(cfg.dataset, "transform") else "Normalize"
    )
    rank_zero_logger.info(f"Using transform: {transform_name}")
    if transform_name == "MinMaxNormalize":
        transforms_u = MinMaxNormalize
        transforms_v = MinMaxNormalize
    elif transform_name == "Normalize":
        transforms_u = Normalize
        transforms_v = Normalize
    elif transform_name == "Mixed":
        transforms_u = MinMaxNormalize
        transforms_v = Normalize
    else:
        raise ValueError(
            f"Unsupported transform: {transform_name}. "
            "Supported transforms are 'MinMaxNormalize', 'Normalize', and 'Mixed'."
        )
        
    train_dataset.set_transforms(
        # ux=transforms_u,
        # uz=transforms_u,
        vp=transforms_v,
        vs=transforms_v,
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
        folder_name = "samples_new"
    )
    val_dataset.set_transforms(
        # ux=transforms_u,
        # uz=transforms_u,
        vp=transforms_v,
        vs=transforms_v,
    )

    # Initialize residual loss with pre-trained regression model
    # Always use hr_mean_conditioning=True
    loss_fn = NoResidualLoss(P_mean=0.0,P_std=1.0,sigma_data=0.5)

    # Create optimizer
    optimizer_class = None
    if torch.cuda.is_available():
        try:
            optimizer_class = getattr(
                importlib.import_module("apex.optimizers"), "FusedAdam"
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
            use_FusedAdam = True
        except ImportError:
            pass
    if optimizer_class is None:
        optimizer_class = torch.optim.AdamW
        rank_zero_logger.info("Using AdamW optimizer")
        use_FusedAdam = False
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_epochs,   # number of epochs over which to anneal
        eta_min=5e-6                        # minimum LR at the end of the cycle
    )

    # Load checkpoint if explicitly requested
    loaded_epoch, total_samples_trained = 0, 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        metadata = {"total_samples_trained": total_samples_trained}
        loaded_epoch = load_checkpoint(
            checkpoint_dir,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
            metadata_dict=metadata,
        )
        total_samples_trained = metadata["total_samples_trained"]

    # Log initial learning rate
    optimizer.param_groups[0]["lr"] = 5e-4
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

        # Setup dataset for current epoch
        train_dataset.set_epoch(epoch)

        # Use LaunchLogger for training
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_dataset), epoch_alert_freq=1
        ) as launchlog:
            for i, data in enumerate(train_dataset):
                ux = torch.nn.functional.pad(data["vx"].permute(0, 1, 3, 2),pad=(0,1))
                uz = torch.nn.functional.pad(data["vz"].permute(0, 1, 3, 2),pad=(0,1))
                vp_target, vs_target, rho_target = data["vp"][:,None], data["vs"][:,None], data["rho"][:,None]
                # #Augmentation
                # if np.random.random() < 0.5:
                #     ux= torch.flip(ux,dims=[-3,-1])
                #     vp_target= torch.flip(vp_target,dims=[-1])
                #     uz= torch.flip(uz,dims=[-3,-1])
                #     vs_target= torch.flip(vs_target,dims=[-1])

                batch_size = ux.shape[0]
                epoch_samples += batch_size

                optimizer.zero_grad(**({} if use_FusedAdam else {"set_to_none": True}))

                loss = loss_fn(
                    diffusion_model=model_fn,  # Use model_fn instead of model
                    x=torch.cat([vp_target, vs_target, rho_target], dim=1),
                    condition={"ux": ux, "uz": uz},
                )
                loss = torch.mean(loss)

                epoch_loss += loss.item() * batch_size

                # Optimize
                loss.backward()
                optimizer.step()

                # Log mini-batch metrics
                current_lr = optimizer.param_groups[0]["lr"]
                batch_metrics = {"batch_loss": loss.item(), "lr": current_lr}
                launchlog.log_minibatch(batch_metrics)
                if dist.rank == 0:
                    wandb.log(batch_metrics)
                if i % cfg.io.log_freq == 0:
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
                "epoch": epoch,
                "mean_loss": mean_loss,
                "time_per_epoch": time_end - time_start,
                "lr": current_lr,
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
                model_fn,
                val_dataset,
                loss_fn,
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
        scheduler.step()
        # Save checkpoint periodically
        if dist.world_size > 1:
            torch.distributed.barrier()
        if epoch % cfg.io.checkpoint_freq == 0 and dist.rank == 0:
            save_checkpoint(
                checkpoint_dir,
                models=model,
                optimizer=optimizer,
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
def validation_step(model, dataset, loss_fn, dist):
    """
    Perform validation on a dataset and return the average loss.
    """
    loss_epoch = 0.0
    num_samples = 0.0

    for i, data in enumerate(dataset):
        ux = torch.nn.functional.pad(data["vx"].permute(0, 1, 3, 2),pad=(0,1))
        uz = torch.nn.functional.pad(data["vz"].permute(0, 1, 3, 2),pad=(0,1))
        vp_target, vs_target, rho_target = data["vp"][:,None], data["vs"][:,None], data["rho"][:,None]

        # Forward pass with validation data
        loss = loss_fn(
            diffusion_model=model,
            x=torch.cat([vp_target, vs_target, rho_target], dim=1),
            condition={"ux": ux, "uz": uz},
        )
        loss = torch.mean(loss)
        loss_epoch += loss.item() * ux.shape[0]
        num_samples += ux.shape[0]

    # Average validation loss across all ranks
    mean_val_loss, num_samples_all_ranks = average_loss(dist, loss_epoch, num_samples)

    return mean_val_loss


def average_loss(dist, loss_value: float, sample_count: int) -> float:
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
