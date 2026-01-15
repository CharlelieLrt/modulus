# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import time
import importlib

import zarr
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from  torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.distributed.utils import reduce_loss

# TODO: update with base DiffusionUNet once refactor is complete
from physicsnemo.diffusion.multi_diffusion import RandomPatching2D
from physicsnemo.diffusion.utils.utils import InfiniteSampler

# TODO: replace with updated APIs once refactor is complete
from utils import EDMPreconditioner, EDMLoss

from data import HRRRSurfaceDataset
from nn import HRRRSurfaceDiffusionNet

# Compilation settings
torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def main():
    # Configuration
    batch_size_per_gpu = 1
    load_checkpoint_from_file = False
    checkpoint_dir = "./checkpoints"
    max_training_samples = 10*(365*24*10)
    checkpoint_frequency = 1000
    validation_frequency = 1000
    num_validation_samples = 100
    logging_frequency = 1
    use_apex = False

    # Initialize distributed environment
    DistributedManager.initialize()
    dist = DistributedManager()

    # Setup logging
    logger = PythonLogger("main")
    logger.logger.setLevel("INFO")
    import logging
    logger.logger.addHandler(logging.StreamHandler())
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)


    patching = RandomPatching2D(
        img_shape=HRRRSurfaceDiffusionNet.IMG_RESOLUTION,
        patch_shape=(448, 448),
        patch_num=4,
    )
    model = HRRRSurfaceDiffusionNet(
        use_apex=use_apex
    )
    model = (
        EDMPreconditioner(
            model=model,
            sigma_data=1.0,
        )
    ).to(dist.device).to(memory_format=torch.channels_last)
    rank_zero_logger.info(f"Training model with {model.num_parameters()} parameters.")

    # Setup DDP for multi-GPU training
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    if load_checkpoint_from_file:
        load_checkpoint(checkpoint_dir, models=model)

    # Compile model
    model = torch.compile(model)

    # Create data loaders
    # Needs zarr 3.0
    root = zarr.open_group(store='s3://hrrr-surface-sda/zarr-v2', mode='r', storage_options={'endpoint_url': 'https://pdx.s8k.io'})
    time_coord = root['time'][:]
    sidx = np.where(time_coord == np.datetime64("2023-01-01T00:00:00"))[0][0]
    eidx = np.where(time_coord == np.datetime64("2024-12-31T23:00:00"))[0][0]
    time_idx = np.arange(sidx, eidx)
    dataset = HRRRSurfaceDataset("s3://hrrr-surface-sda/zarr-v2", time_idx, storage_options={'endpoint_url': 'https://pdx.s8k.io'})
    train_iter = DataLoader(dataset, batch_size=batch_size_per_gpu,
            sampler=InfiniteSampler(dataset=dataset, shuffle=True),
            num_workers=8,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            prefetch_factor=4,
            persistent_workers=False)
    num_training_samples = len(dataset)

    sidx = np.where(time_coord == np.datetime64("2025-01-01T00:00:00"))[0][0]
    eidx = np.where(time_coord == np.datetime64("2025-06-01T00:00:00"))[0][0]
    time_idx = np.arange(sidx, eidx, 25)
    dataset = HRRRSurfaceDataset("s3://hrrr-surface-sda/zarr-v2", time_idx, storage_options={'endpoint_url': 'https://pdx.s8k.io'})
    val_iter = DataLoader(dataset, batch_size=batch_size_per_gpu,
            sampler=InfiniteSampler(dataset=dataset, shuffle=False),
            num_workers=2,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            prefetch_factor=2,
            persistent_workers=False)

    # Create loss function with multi-diffusion support
    loss_fn = EDMLoss(
        model=model,
        P_mean=0.0,
        P_std=1.0,
        sigma_data=1.0,
        patching=patching,
    )

    # Initialize optimizer
    if use_apex:
        FusedAdam = getattr(importlib.import_module("apex.optimizers"), "FusedAdam")
        optimizer = FusedAdam(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.0,
        )

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_training_samples // num_training_samples,
        eta_min=5e-6,
    )

    # Load checkpoint if requested
    current_samples_trained = 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if load_checkpoint_from_file:
        metadata = {"current_samples_trained": current_samples_trained}
        load_checkpoint(
            checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
            metadata_dict=metadata,
        )
        current_samples_trained = metadata["current_samples_trained"]
        rank_zero_logger.info(
            f"Resumed from samples trained: {current_samples_trained}"
        )

    # Training loop (batch-based with InfiniteSampler)
    rank_zero_logger.info("Training started...")

    # Running average for loss
    loss_running_mean = 0.0
    n_loss_running_mean = 1

    total_batch_size = batch_size_per_gpu * dist.world_size

    # Counters for periodic tasks
    samples_since_scheduler_update = 0
    samples_since_logging = 0
    samples_since_validation = 0
    samples_since_checkpoint = 0

    while current_samples_trained < max_training_samples:
        tick_start_time = time.time()

        model.train()

        # Get next batch from infinite sampler
        # c = torch.Size([4, 1059, 1799])
        # x = torch.Size([16, 1059, 1799])
        _, (c, x) = next(enumerate(train_iter))
        c = c.to(dist.device, non_blocking=True).to(memory_format=torch.channels_last)
        x = x.to(dist.device, non_blocking=True).to(memory_format=torch.channels_last)
        batch_size = x.shape[0]

        # Forward pass
        optimizer.zero_grad(**({} if use_apex else {"set_to_none": True}))
        loss = loss_fn(x, {"c": c}).mean()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        mean_loss = reduce_loss(loss.item() * batch_size, dst_rank=0) 

        # Update running mean of loss
        if dist.rank == 0:
            loss_running_mean += (mean_loss / total_batch_size - loss_running_mean) / n_loss_running_mean
            n_loss_running_mean += 1
            current_samples_trained += total_batch_size

        # Update scheduler periodically
        samples_since_scheduler_update += total_batch_size
        if samples_since_scheduler_update >= num_training_samples:
            scheduler.step()
            samples_since_scheduler_update = 0

        # Periodic logging
        samples_since_logging += total_batch_size
        if samples_since_logging >= logging_frequency:
            elapsed = time.time() - tick_start_time
            rank_zero_logger.info(
                f"Samples trained: {current_samples_trained}, "
                f"loss: {loss_running_mean:.3e}, "
                f"learning rate: {optimizer.param_groups[0]['lr']:.2e}, "
                f"time per 1k samples: {(elapsed / (samples_since_logging)) * 1000:.1f}s"
            )
            # Reset running mean after logging
            loss_running_mean = 0.0
            n_loss_running_mean = 1
            tick_start_time = time.time()
            samples_since_logging = 0

        # Validation step
        samples_since_validation += total_batch_size
        if samples_since_validation >= validation_frequency:
            model.eval()
            val_loss_running_mean = 0.0
            n_val_loss_running_mean = 1
            current_validation_samples = 0
            with torch.no_grad():
                while current_validation_samples < num_validation_samples:
                    _, (c, x) = next(enumerate(val_iter))
                    c = c.to(dist.device, non_blocking=True).to(memory_format=torch.channels_last)
                    x = x.to(dist.device, non_blocking=True).to(memory_format=torch.channels_last)
                    batch_size = x.shape[0]
                    
                    val_loss = loss_fn(x, {"c": c}).mean()
                    mean_val_loss = (
                        reduce_loss(val_loss.item() * batch_size, dst_rank=0)
                    )
                    if dist.rank == 0:
                        val_loss_running_mean += (
                            mean_val_loss / total_batch_size - val_loss_running_mean
                        ) / n_val_loss_running_mean
                        n_val_loss_running_mean += 1
                    current_validation_samples += total_batch_size
            rank_zero_logger.info(
                f"Samples trained: {current_samples_trained}, "
                f"val_loss: {val_loss_running_mean:.3e}, "
            )
            samples_since_validation = 0

        # Periodic checkpoint
        samples_since_checkpoint += total_batch_size
        if samples_since_checkpoint >= checkpoint_frequency:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={
                        "current_samples_trained": current_samples_trained,
                    },
                )
                rank_zero_logger.info(
                    f"Saved checkpoint at samples trained: {current_samples_trained}"
                )
            samples_since_checkpoint = 0

    # Cleanup
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
