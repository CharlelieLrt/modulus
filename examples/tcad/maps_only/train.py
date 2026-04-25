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

import logging
import math
import random
import time

import hydra
import numpy as np
import torch
from dataset import TCADMapsDataPipe
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.nn import TimeConditionedGeoTransolver

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import reduce_loss
from physicsnemo.utils import (
    get_checkpoint_dir,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


def _teacher_forcing_schedule(
    samples_trained: int,
    annealing_samples: int,
    initial_fraction: float,
    final_fraction: float,
) -> float:
    """Monotonic tanh schedule for the teacher-forcing fraction.

    At ``samples_trained == annealing_samples`` the fraction has traversed 95%
    of the way from ``initial_fraction`` to ``final_fraction``; past that, the
    tanh continues to asymptote toward ``final_fraction``. A single parameter
    (``annealing_samples``) fully determines the transition width; steepness
    is fixed so that the 95%-crossing lines up with that parameter.

    Parameters
    ----------
    samples_trained : int
        Number of training samples seen so far.
    annealing_samples : int
        Target number of samples at which the fraction should have covered 95%
        of the transition. Sets the tanh width.
    initial_fraction : float
        Starting value at ``samples_trained == 0``.
    final_fraction : float
        Asymptotic value as ``samples_trained → ∞``.
    """
    # tanh(k/2) = 0.9 → k = 2*atanh(0.9) ≈ 2.944
    k = 2.0 * math.atanh(0.9)
    progress = samples_trained / max(1, annealing_samples)
    s = 0.5 * (1.0 + math.tanh(k * (progress - 0.5)))
    return initial_fraction + (final_fraction - initial_fraction) * s


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg_scheduler: DictConfig,
    num_training_samples: int,
    max_training_samples: int,
):
    """Return an LR scheduler according to cfg, or ``None`` for constant lr.

    Keeps the scheduler choice as an opt-in config block so recipes can run
    with or without annealing. ``cfg_scheduler`` is expected to have a
    ``name`` field; anything else depends on the scheduler kind.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to wrap.
    cfg_scheduler : DictConfig or None
        Scheduler config block. ``None`` or ``name=null`` disables scheduling.
        Currently supported names: ``"cosine_annealing"`` with ``eta_min``.
    num_training_samples : int
        Size of the training dataset (one ``scheduler.step()`` per that many
        samples seen).
    max_training_samples : int
        Total number of samples the training loop will see before stopping;
        used to compute ``T_max`` of the cosine schedule.
    """
    if cfg_scheduler is None:
        return None
    name = cfg_scheduler.name
    if name is None:
        return None
    if name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_training_samples // num_training_samples),
            eta_min=float(cfg_scheduler.eta_min),
        )
    raise ValueError(
        f"Unknown scheduler name {name!r}; expected null or 'cosine_annealing'."
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config_train")
def main(cfg: DictConfig) -> None:
    """Train a :class:`TimeConditionedGeoTransolver` with the push-forward trick.

    Sample-based loop driven by ``InfiniteSampler``. Each iteration runs two
    forward passes (teacher-forcing on the ground-truth current state, then
    push-forward on the detached first-pass output) and combines the two
    losses per sample according to the teacher-forcing schedule. Loss and
    sample counts are reduced across DDP ranks before logging.
    """
    # Distributed init
    DistributedManager.initialize()
    dist = DistributedManager()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger = PythonLogger("main")
    logger.logger.setLevel(logging.INFO)
    rank_zero = RankZeroLoggingWrapper(logger, dist)
    rank_zero.info(f"Rank {dist.rank}/{dist.world_size} | device {dist.device}")

    checkpoint_dir = get_checkpoint_dir(str(cfg.io.checkpoint_dir), "tcad_maps_only")

    # Model instantiation
    model = TimeConditionedGeoTransolver(
        functional_dim=5,  # temperature + potential + XYZ concatenated
        out_dim=2,
        global_dim=2,
        geometry_dim=3,  # XYZ positions also fed through the geometry projector
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        n_head=cfg.model.n_head,
        slice_num=cfg.model.slice_num,
        mlp_ratio=cfg.model.mlp_ratio,
        plus=cfg.model.plus,
        time_embed_channels=cfg.model.time_embed_channels,
    ).to(dist.device)
    rank_zero.info(f"Model parameters: {model.num_parameters():,}")

    if cfg.io.load_checkpoint:
        load_checkpoint(checkpoint_dir, models=model)

    # Compile before DDP wrap (per PyTorch recommendation)
    if cfg.model.compile:
        rank_zero.info("Compiling model with torch.compile ...")
        model = torch.compile(model)

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    # Resume metadata (for InfiniteSampler start_idx)
    current_samples_trained = 0
    if cfg.io.load_checkpoint:
        metadata = {"current_samples_trained": 0}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        current_samples_trained = int(metadata["current_samples_trained"])
        rank_zero.info(f"Resuming at samples trained: {current_samples_trained}")
    total_batch_size = cfg.training.batch_size_per_gpu * dist.world_size
    sampler_start_idx = current_samples_trained

    # DataLoader
    train_loader = TCADMapsDataPipe(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        batch_size_per_device=cfg.training.batch_size_per_gpu,
        n_steps=cfg.dataset.n_steps,
        stats_file=to_absolute_path(cfg.dataset.stats_file),
        shuffle=True,
        num_workers=cfg.training.num_workers,
        process_rank=dist.rank,
        world_size=dist.world_size,
        start_idx=sampler_start_idx,
        seed=cfg.seed,
    )
    num_training_samples = len(train_loader.dataset)
    rank_zero.info(f"Training dataset: {num_training_samples} samples")
    train_iter = iter(train_loader)

    # Pre-fetch stats for normalization
    coord_mean, coord_std = train_loader.get_stats("coords")
    T_mean, T_std = train_loader.get_stats("temperature")
    V_mean, V_std = train_loader.get_stats("potential")
    _, t_scale = train_loader.get_stats("t")
    # (1, 1, 2, 1) for broadcasting over (B, S, V, N)
    var_mean = torch.tensor([T_mean, V_mean], device=dist.device).view(1, 1, 2, 1)
    var_std = torch.tensor([T_std, V_std], device=dist.device).view(1, 1, 2, 1)
    rank_zero.info(
        f"Stats loaded | coord: ({coord_mean:.3e}, {coord_std:.3e}) | "
        f"T: ({T_mean:.3f}, {T_std:.3f}) | V: ({V_mean:.3f}, {V_std:.3f}) | "
        f"t_scale: {t_scale:.3e}"
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        cfg_scheduler=cfg.training.scheduler,
        num_training_samples=num_training_samples,
        max_training_samples=cfg.training.max_training_samples,
    )
    rank_zero.info(
        f"Scheduler: {scheduler.__class__.__name__ if scheduler is not None else 'None (constant lr)'}"
    )

    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        load_checkpoint(
            checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
        )

    # Push-forward schedule parameters
    pf_cfg = cfg.training.push_forward
    annealing_samples = int(pf_cfg.annealing_samples)
    initial_fraction = float(pf_cfg.initial_fraction)
    final_fraction = float(pf_cfg.final_fraction)
    rank_zero.info(
        f"Push-forward schedule: initial={initial_fraction}, "
        f"final={final_fraction}, annealing_samples={annealing_samples}"
    )

    # Training loop
    rank_zero.info("Training started...")
    samples_since_logging = 0
    samples_since_checkpoint = 0
    samples_since_scheduler_update = 0
    tick_start = time.time()

    # Loss EMA for logging. Time constant = half a dataset traversal in
    # optimizer steps, so alpha = 2 * total_batch_size / num_training_samples.
    ema_alpha = 2.0 * total_batch_size / max(1, num_training_samples)
    loss_ema: float | None = None
    rank_zero.info(
        f"Loss EMA: alpha={ema_alpha:.3e} "
        f"(tau ≈ {1.0 / ema_alpha:.0f} optimizer steps ≈ 0.5 epoch)"
    )

    while current_samples_trained < cfg.training.max_training_samples:
        model.train()
        sample, _meta = next(train_iter)
        sample = sample.to(dist.device, non_blocking=True)

        # Normalization
        positions = (sample["positions"] - coord_mean) / coord_std  # (B, N, 3)
        variables = (sample["variables"] - var_mean) / var_std  # (B, 3, 2, N)
        t_norm = sample["time"] / t_scale  # (B, 3), t=0 preserved

        # Unpack 3 consecutive timesteps: n, n+1, n+2
        x_n_vals = variables[:, 0].transpose(-1, -2).contiguous()  # (B, N, 2)
        x_n1 = variables[:, 1].transpose(-1, -2).contiguous()  # (B, N, 2)
        x_n2 = variables[:, 2].transpose(-1, -2).contiguous()  # (B, N, 2)
        # Concat normalized positions onto the input token stream → (B, N, 5).
        # Targets stay at 2 channels (we only predict temperature + potential).
        x_n = torch.cat([x_n_vals, positions], dim=-1).contiguous()

        t_n_ = t_norm[:, 0]  # (B,)
        t_n1_ = t_norm[:, 1]  # (B,)
        dt_1 = t_norm[:, 1] - t_norm[:, 0]  # (B,) step n   → n+1
        dt_2 = t_norm[:, 2] - t_norm[:, 1]  # (B,) step n+1 → n+2

        # Pass 1: teacher-forcing input (ground-truth x_n → predicts x_{n+1})
        optimizer.zero_grad(set_to_none=True)
        global_emb_1 = torch.stack([t_n_, dt_1], dim=-1).unsqueeze(1)  # (B, 1, 2)
        x_pred_p1 = model(
            local_embedding=x_n,
            geometry=positions,
            global_embedding=global_emb_1,
            t=t_n_,
            dt=dt_1,
        )

        # Pass 2: push-forward input (detached pass-1 output → predicts x_{n+2})
        # x_pred_p1 has 2 channels; re-concat positions to restore functional_dim=5.
        x_pred_p1_det = torch.cat(
            [x_pred_p1.detach().clone(), positions], dim=-1
        ).contiguous()
        global_emb_2 = torch.stack([t_n1_, dt_2], dim=-1).unsqueeze(1)  # (B, 1, 2)
        x_pred_p2 = model(
            local_embedding=x_pred_p1_det,
            geometry=positions,
            global_embedding=global_emb_2,
            t=t_n1_,
            dt=dt_2,
        )

        # Per-sample MSE for each pass, shape (B,)
        se_p1 = ((x_pred_p1 - x_n1) ** 2).mean(dim=(1, 2))
        se_p2 = ((x_pred_p2 - x_n2) ** 2).mean(dim=(1, 2))

        # Random per-sample mask: 1 = teacher forcing, 0 = push-forward
        batch_size = x_n.shape[0]
        tf_fraction = _teacher_forcing_schedule(
            samples_trained=current_samples_trained,
            annealing_samples=annealing_samples,
            initial_fraction=initial_fraction,
            final_fraction=final_fraction,
        )
        num_tf = int(round(tf_fraction * batch_size))
        tf_mask = torch.zeros(batch_size, device=dist.device)
        if num_tf > 0:
            perm = torch.randperm(batch_size, device=dist.device)
            tf_mask[perm[:num_tf]] = 1.0
        pf_mask = 1.0 - tf_mask

        # Losses: sum reduction per component, then batch-average
        loss_tf_sum = (se_p1 * tf_mask).sum()
        loss_pf_sum = (se_p2 * pf_mask).sum()
        loss = (loss_tf_sum + loss_pf_sum) / batch_size

        loss.backward()
        optimizer.step()

        # Update local EMA of the loss (one update per optimizer step).
        loss_val = loss.item()
        loss_ema = (
            loss_val
            if loss_ema is None
            else ema_alpha * loss_val + (1.0 - ema_alpha) * loss_ema
        )

        current_samples_trained += total_batch_size
        samples_since_scheduler_update += total_batch_size
        samples_since_logging += total_batch_size
        samples_since_checkpoint += total_batch_size

        # Scheduler step once per dataset-equivalent number of samples
        if (
            scheduler is not None
            and samples_since_scheduler_update >= num_training_samples
        ):
            scheduler.step()
            samples_since_scheduler_update = 0

        # Periodic logging — only reduce across ranks at log tick. The EMA is
        # rank-local; mean of rank-local EMAs equals EMA of cross-rank-mean
        # loss by linearity.
        if samples_since_logging >= cfg.io.logging_frequency:
            # Collective reduction — must be called by every rank.
            loss_sum = reduce_loss(loss_ema, dst_rank=0)
            if dist.rank == 0:
                reduced_loss = loss_sum / dist.world_size
                elapsed = time.time() - tick_start
                steps = samples_since_logging / total_batch_size
                rank_zero.info(
                    f"samples: {current_samples_trained:>10d} | "
                    f"tf_frac: {tf_fraction:.3f} | "
                    f"loss: {reduced_loss:.3e} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"throughput: {samples_since_logging / elapsed / 1000:.3f} ksamp/s | "
                    f"step: {elapsed / steps:.3f}s"
                )
            tick_start = time.time()
            samples_since_logging = 0

        # Periodic checkpoint
        if samples_since_checkpoint >= cfg.io.checkpoint_frequency:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={"current_samples_trained": current_samples_trained},
                )
                rank_zero.info(
                    f"Saved checkpoint at samples: {current_samples_trained}"
                )
            samples_since_checkpoint = 0

    rank_zero.info("Training completed.")


if __name__ == "__main__":
    main()
