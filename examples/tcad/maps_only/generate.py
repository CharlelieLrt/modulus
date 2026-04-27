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
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from dataset import TCADMapsDataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from utils.nn import TimeConditionedGeoTransolver

from physicsnemo.utils.logging import PythonLogger


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Autoregressively roll out a trained checkpoint on selected simulations.

    Loads a ``.mdlus`` checkpoint, iterates through every ``(thickness,
    sim_id)`` pair listed in the config, and for each one feeds the
    ground-truth initial state into the model and steps it forward to the
    final timestep. Writes one ``.pth`` file per simulation containing the
    mesh, time axis, predicted and ground-truth fields, and per-timestep MSE
    in physical units.
    """
    logger = PythonLogger("generate")
    logger.logger.setLevel(logging.INFO)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model. from_checkpoint raises if the path is not a valid .mdlus file.
    ckpt_path = to_absolute_path(str(cfg.checkpoint))
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = TimeConditionedGeoTransolver.from_checkpoint(ckpt_path)
    model = model.to(device).eval()
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # One timestep per dataset sample for sequential rollout.
    dataset = TCADMapsDataset(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        n_steps=1,
        stats_file=to_absolute_path(cfg.dataset.stats_file),
        thickness=cfg.dataset.thickness,
    )
    th_filter = cfg.dataset.thickness
    logger.info(
        f"Dataset: {len(dataset)} samples across all sims "
        f"| thickness filter: {th_filter if th_filter is not None else '<all>'}"
    )

    # Stats for manual normalize / unnormalize.
    coord_mean, coord_std = dataset.get_stats("coords")
    T_mean, T_std = dataset.get_stats("temperature")
    V_mean, V_std = dataset.get_stats("potential")
    _, t_scale = dataset.get_stats("t")
    var_mean = torch.tensor([T_mean, V_mean], device=device).view(1, 2, 1)
    var_std = torch.tensor([T_std, V_std], device=device).view(1, 2, 1)
    logger.info(
        f"Stats | coord: ({coord_mean:.3e}, {coord_std:.3e}) | "
        f"T: ({T_mean:.3f}, {T_std:.3f}) | V: ({V_mean:.3f}, {V_std:.3f}) | "
        f"t_scale: {t_scale:.3e}"
    )

    output_dir = Path(to_absolute_path(str(cfg.io.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Expand the ``rollout.simulations`` list, resolving any ``sim_id: "all"``
    # entry into one ``(thickness, sim_id)`` pair per available simulation.
    pairs: list[tuple[str, int]] = []
    for sim_cfg in cfg.rollout.simulations:
        thickness = str(sim_cfg.thickness)
        sim_id_raw = sim_cfg.sim_id
        if isinstance(sim_id_raw, str) and sim_id_raw.lower() == "all":
            pairs.extend((thickness, sid) for sid in dataset.get_sim_ids(thickness))
        else:
            pairs.append((thickness, int(sim_id_raw)))
    logger.info(f"Rolling out {len(pairs)} simulation(s)")
    logging_frequency = int(cfg.io.logging_frequency)

    overall_start = time.time()

    for thickness, sim_id in pairs:
        indices = dataset.get_sim_indices(thickness, sim_id)

        # Gather all ground-truth timesteps for this sim (raw physical units).
        gt_vars_list: list[torch.Tensor] = []
        times_list: list[float] = []
        positions_raw: torch.Tensor | None = None
        thickness_m: float = 0.0
        for idx in indices:
            sample, _ = dataset[idx]
            if positions_raw is None:
                positions_raw = sample["positions"]  # (N, 3) meters
                thickness_m = float(sample["thickness"].item())
            gt_vars_list.append(sample["variables"][0])  # (V, N) raw
            times_list.append(float(sample["time"][0].item()))

        gt_vars = torch.stack(gt_vars_list, dim=0)  # (T, V, N) raw
        times = torch.tensor(times_list, dtype=torch.float32)  # (T,) seconds
        T_len, V, N = gt_vars.shape

        # Normalized quantities for model input.
        positions_norm = (
            ((positions_raw - coord_mean) / coord_std).to(device).unsqueeze(0)
        )
        t_norm = (times / t_scale).to(device)
        gt_vars_norm = (gt_vars.to(device) - var_mean) / var_std
        # Dimensionless thickness as a (1, 1) tensor; model multiplies by
        # max_positions internally.
        thickness_t = torch.tensor(
            [[thickness_m / coord_std]], dtype=torch.float32, device=device
        )

        # Predicted states, seeded with the ground-truth initial condition.
        pred_vars_norm = torch.zeros_like(gt_vars_norm)
        pred_vars_norm[0] = gt_vars_norm[0]

        logger.info(
            f"[{thickness}/sim{sim_id}] Rolling out {T_len} timesteps "
            f"(thickness={thickness_m * 1e9:.1f} nm)"
        )
        start = time.time()
        with torch.no_grad():
            for i in range(T_len - 1):
                x_curr_vals = (
                    pred_vars_norm[i].transpose(0, 1).unsqueeze(0)
                )  # (1, N, 2)
                # Match training-time input layout: concat positions → (1, N, 5)
                x_curr = torch.cat([x_curr_vals, positions_norm], dim=-1).contiguous()
                t_n = t_norm[i : i + 1]
                dt_n = t_norm[i + 1 : i + 2] - t_n
                x_pred = model(
                    local_embedding=x_curr,
                    geometry=positions_norm,
                    t=t_n,
                    dt=dt_n,
                    thickness=thickness_t,
                )
                pred_vars_norm[i + 1] = x_pred.squeeze(0).transpose(0, 1).contiguous()

                if (i + 1) % logging_frequency == 0 or i + 1 == T_len - 1:
                    pred_unnorm_step = pred_vars_norm[i + 1] * var_std[0] + var_mean[0]
                    gt_unnorm_step = gt_vars[i + 1].to(device)
                    mse_T = (
                        ((pred_unnorm_step[0] - gt_unnorm_step[0]) ** 2).mean().item()
                    )
                    mse_V = (
                        ((pred_unnorm_step[1] - gt_unnorm_step[1]) ** 2).mean().item()
                    )
                    logger.info(
                        f"[{thickness}/sim{sim_id}] step {i + 1:>4d}/{T_len - 1} | "
                        f"t={times[i + 1].item():.2e}s | "
                        f"mse_T={mse_T:.3e} K^2 | mse_V={mse_V:.3e} V^2"
                    )

        # Un-normalize predictions → physical units.
        pred_vars = pred_vars_norm * var_std + var_mean

        gt_dev = gt_vars.to(device)
        mse_T_t = ((pred_vars[:, 0] - gt_dev[:, 0]) ** 2).mean(dim=-1)
        mse_V_t = ((pred_vars[:, 1] - gt_dev[:, 1]) ** 2).mean(dim=-1)
        mse_T_mean = float(mse_T_t.mean().item())
        mse_V_mean = float(mse_V_t.mean().item())

        elapsed = time.time() - start
        logger.info(
            f"[{thickness}/sim{sim_id}] rollout done in {elapsed:.1f}s | "
            f"mean_mse_T={mse_T_mean:.3e} K^2 | mean_mse_V={mse_V_mean:.3e} V^2"
        )

        # Save everything needed to plot + compare against ground truth.
        out = {
            # Mesh (physical units, meters). Split X/Y/Z for convenience.
            "positions": positions_raw.cpu(),
            "x": positions_raw[:, 0].cpu(),
            "y": positions_raw[:, 1].cpu(),
            "z": positions_raw[:, 2].cpu(),
            # Time axis and identity.
            "time": times.cpu(),
            "thickness": torch.tensor([thickness_m], dtype=torch.float32),
            "thickness_str": thickness,
            "sim_id": sim_id,
            # Fields in physical units.
            "temperature_pred": pred_vars[:, 0].cpu(),
            "potential_pred": pred_vars[:, 1].cpu(),
            "temperature_gt": gt_vars[:, 0].cpu(),
            "potential_gt": gt_vars[:, 1].cpu(),
            # Per-timestep MSE.
            "mse_temperature": mse_T_t.cpu(),
            "mse_potential": mse_V_t.cpu(),
        }
        sim_dir = output_dir / thickness
        sim_dir.mkdir(parents=True, exist_ok=True)
        out_path = sim_dir / f"rollout_sim{sim_id}.pth"
        torch.save(out, out_path)
        logger.info(f"[{thickness}/sim{sim_id}] saved → {out_path}")

    logger.info(
        f"All rollouts complete in {time.time() - overall_start:.1f}s. "
        f"Outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
