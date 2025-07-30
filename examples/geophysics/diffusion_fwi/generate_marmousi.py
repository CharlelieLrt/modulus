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
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from matplotlib import gridspec

from physicsnemo.datapipes.cae.efwi_datapipe_combined import EFWIDatapipe
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from physicsnemo.utils.generative import deterministic_sampler, stochastic_sampler, posteriori_sampler
from physicsnemo.utils.transforms import MinMaxNormalize, Normalize
from physicsnemo.models.geophysics.convnext_unet import  ConvNextUNet
from physicsnemo.models.diffusion import edm_precond
from physicsnemo.models.diffusion.conditional import ConditionalDiffusionAdapter

import deepwave
from deepwave import elastic

def RMSE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Root Mean Square Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()

def classify_lithology(vp, vs):
    """
    Classify lithology element-wise based on Vp, Vs, and Vp/Vs ratio.

    Inputs:
        vp: P-wave velocity in m/s (2D array)
        vs: S-wave velocity in m/s (2D array)

    Returns:
        lith: array of rock type strings
        alpha, beta: Gardner coefficients
        salt_mask: boolean mask to override density with fixed value

    References:
    - Gardner et al. (1974), Geophysics, https://doi.org/10.1190/1.1440465
    - Mavko et al. (2009), "The Rock Physics Handbook"
    - Castagna et al. (1985), Geophysics, 50(4), 571–581
    - Gray & Head (2000), "Modeling, migration, and velocity analysis in salt", Geophysics
    """

    vpr = vp / vs  # Vp/Vs ratio
    lith = np.full(vp.shape, "Unknown", dtype=object)
    alpha = np.zeros_like(vp, dtype=float)
    beta = np.zeros_like(vp, dtype=float)

    # === Lithology Masks ===
    shale_mask = vpr > 2.0  # Castagna et al., 1985

    sandstone_mask = (
        (vpr >= 1.6) & (vpr <= 2.2) &
        (vp >= 2500) & (vp <= 5000) &
        ~shale_mask
    )

    limestone_mask = (
        (vp >= 5000) & (vp <= 6500) &
        (vpr >= 1.7) & (vpr <= 1.95) &
        ~shale_mask & ~sandstone_mask
    )

    dolomite_mask = (
        (vp >= 5500) & (vp <= 7000) &
        (vpr >= 1.65) & (vpr <= 1.9) &
        ~shale_mask & ~sandstone_mask & ~limestone_mask
    )

    coal_mask = (
        (vp < 3600) & (vpr > 1.8) &
        ~shale_mask & ~sandstone_mask & ~limestone_mask & ~dolomite_mask
    )

    anhydrite_mask = (
        (vp >= 5800) & (vp <= 6800) &
        (vpr <= 1.8) &
        ~shale_mask & ~dolomite_mask & ~limestone_mask
    )

    # Salt: Vp ~4500 m/s, Vs ≈ 0 → large Vp/Vs
    salt_mask = (
        (vp >= 4300) & (vp <= 4700) &
        (vs < 700) &         # Vs near zero
        (vpr >= 6.0)         # Very high Vp/Vs
    )

    # === Assign Lithologies and Gardner Parameters ===
    # Gardner et al. (1974), Mavko et al. (2009)
    lith[shale_mask] = "Shale"
    alpha[shale_mask], beta[shale_mask] = 0.31, 0.2928

    lith[sandstone_mask] = "Sandstone"
    alpha[sandstone_mask], beta[sandstone_mask] = 0.25, 0.28

    lith[limestone_mask] = "Limestone"
    alpha[limestone_mask], beta[limestone_mask] = 0.30, 0.25

    lith[dolomite_mask] = "Dolomite"
    alpha[dolomite_mask], beta[dolomite_mask] = 0.29, 0.25

    lith[coal_mask] = "Coal"
    alpha[coal_mask], beta[coal_mask] = 0.24, 0.25

    lith[anhydrite_mask] = "Anhydrite"
    alpha[anhydrite_mask], beta[anhydrite_mask] = 0.27, 0.25

    lith[salt_mask] = "Salt"
    # Do not assign alpha/beta for salt, use fixed density

    # Fallback
    fallback_mask = (alpha == 0) & (~salt_mask)
    lith[fallback_mask] = "Unknown"
    alpha[fallback_mask], beta[fallback_mask] = 0.31, 0.25  # Generic fallback

    return lith, alpha, beta, salt_mask


def compute_density(vp, alpha, beta, salt_mask=None):
    """
    Compute density using Gardner’s rule.

    Inputs:
        vp: P-wave velocity in m/s
        alpha, beta: Gardner coefficients (same shape as vp)
        salt_mask: optional boolean mask to fix salt density

    Returns:
        rho: estimated density (g/cm³)

    Notes:
    - For salt, we override Gardner with a fixed value: ρ = 2.15 g/cm³
      (Gray & Head, 2000; Mavko et al.)
    """
    rho = alpha * vp ** beta
    if salt_mask is not None:
        rho = np.where(salt_mask, 2.15, rho)
    return rho



def diffusion_step(
    net,
    sampler_fn,
    latents_shape,
    img_lr,
    device,
    ensemble_size,
    x_tar,
    x_mean=None,
    seed=42,
):
    """
    Perform diffusion sampling to generate outputs.

    Parameters
    ----------
    net : torch.nn.Module
        The diffusion model
    sampler_fn : callable
        The sampling function
    latents_shape : tuple
        Shape of latents to generate
    img_lr : torch.Tensor
        Low-resolution input
    device : torch.device
        Device to run generation on
    ensemble_size : int
        Number of ensemble members to generate
    x_mean : torch.Tensor, optional
        Conditioning mean for model (from regression), by default None
    seed : int, optional
        Random seed, by default 42

    Returns
    -------
    torch.Tensor
        Generated samples, shape [ensemble_size, channels, height, width]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    channels = latents_shape[1]
    height = latents_shape[2]
    width = latents_shape[3]

    # Create random noise vectors for each sample in the ensemble
    z = torch.randn(
        (ensemble_size, channels, height, width),
        device=device,
        dtype=torch.float32,
    )
    # print(img_lr.shape)
    # Expand low-res input to match ensemble size
    expanded_img_lr = img_lr.expand(ensemble_size, -1, -1, -1)

    # Prepare conditioning dict
    condition = {
        "ux": expanded_img_lr[:, : img_lr.shape[1] // 2],
        "uz": expanded_img_lr[:, img_lr.shape[1] // 2 :],
    }

    # Add x_mean conditioning if provided
    condition["x_mean"] = x_mean

    # Random Measurements
    measurement = x_tar
    # Perform sampling
    samples = sampler_fn(
        net=net,
        latents=z,
        img_lr=condition,
        measurement=measurement.squeeze(),
    )

    return samples.detach()



def plot_prediction(
    inputs,
    targets,
    predictions,
    metrics,
    save_dir,
    sources_to_plot=3,
    idx_to_plot=[1, 3, 5],
):
    """
    Plot predictions vs ground truth for vp, vs, rho, and inputs vx, vz.
    Also plots ensemble samples and variances.

    Parameters
    ----------
    inputs : dict
        Dictionary containing 'vx' and 'vz' tensors
    targets : dict
        Dictionary containing 'vp', 'vs', 'rho' tensors (ground truth)
    predictions : dict
        Dictionary containing 'vp', 'vs', 'rho', 'vp_ensemble', 'vs_ensemble', 'rho_ensemble'
    metrics : dict
        Dictionary containing RMSE and MAE values for vp, vs, rho
    save_dir : Path
        Directory to save the plots
    sources_to_plot : int
        Number of source channels to plot from each input
    idx_to_plot : list
        Indices of ensemble members to visualize
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load input wavefields
    vx = inputs["vx"][0].cpu().numpy()  # shape: [nb_sources, H, W]
    vz = inputs["vz"][0].cpu().numpy()
    
    # Load targets
    vp_true = targets["vp"][0].cpu().numpy()
    vs_true = targets["vs"][0].cpu().numpy()
    rho_true = targets["rho"][0].cpu().numpy()

    # Load predictions (mean)
    vp_pred = predictions["vp"][0].cpu().numpy()
    vs_pred = predictions["vs"][0].cpu().numpy()
    rho_pred = predictions["rho"][0].cpu().numpy()

    # Load ensembles
    vp_ensemble = predictions["vp_ensemble"].cpu().numpy()
    vs_ensemble = predictions["vs_ensemble"].cpu().numpy()
    rho_ensemble = predictions["rho_ensemble"].cpu().numpy()

    # Extract metrics
    vp_rmse, vp_mae = metrics["vp_rmse"], metrics["vp_mae"]
    vs_rmse, vs_mae = metrics["vs_rmse"], metrics["vs_mae"]
    rho_rmse, rho_mae = metrics["rho_rmse"], metrics["rho_mae"]

    ########################
    # 1. Plot vx, vz inputs
    ########################
    nb_sources = vx.shape[0]
    source_indices = list(range(nb_sources)) if sources_to_plot >= nb_sources else \
        np.linspace(0, nb_sources - 1, sources_to_plot, dtype=int).tolist()

    if len(source_indices) > 0:
        fig1 = plt.figure(figsize=(3 * len(source_indices), 6))
        gs = gridspec.GridSpec(2, len(source_indices), figure=fig1, wspace=0.05, hspace=0.1)
        H, W = vx[0].shape

        for j, src_idx in enumerate(source_indices):
            ax = fig1.add_subplot(gs[0, j])
            im = ax.imshow(vx[src_idx], cmap="viridis", extent=[0, W, 0, H],aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"vx source {src_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig1.add_subplot(gs[1, j])
            im = ax.imshow(vz[src_idx], cmap="viridis", extent=[0, W, 0, H],aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"vz source {src_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig1.savefig(Path(save_dir) / "inputs.png")
        plt.close(fig1)

    ########################
    # 2. Plot Predictions
    ########################
    def plot_output_comparison(var_name, true, pred, ensemble, rmse, mae, idx_row):
        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())

        ax = fig2.add_subplot(gs[idx_row, 0])
        im = ax.imshow(true.squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Ground Truth {var_name}")
        ax.set_xticks([])
        ax.set_yticks([])

        for i, idx in enumerate(idx_to_plot):
            ax = fig2.add_subplot(gs[idx_row, 1 + i])
            ax.imshow(ensemble[idx].squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Sample {i+1} {var_name}")
            ax.set_xticks([])
            ax.set_yticks([])

        ax = fig2.add_subplot(gs[idx_row, 1 + len(idx_to_plot)])
        ax.imshow(pred.squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Mean {var_name}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar_ax = fig2.add_subplot(gs[idx_row, -1])
        plt.colorbar(im, cax=cbar_ax)

    num_samples = len(idx_to_plot)
    fig2 = plt.figure(figsize=(5 * (num_samples + 2), 12))
    width_ratios = [1] * (num_samples + 2) + [0.05]
    gs = fig2.add_gridspec(3, num_samples + 3, width_ratios=width_ratios, wspace=0.05, hspace=0.25)

    plot_output_comparison("vp", vp_true, vp_pred, vp_ensemble, vp_rmse, vp_mae, idx_row=0)
    plot_output_comparison("vs", vs_true, vs_pred, vs_ensemble, vs_rmse, vs_mae, idx_row=1)
    plot_output_comparison("rho", rho_true, rho_pred, rho_ensemble, rho_rmse, rho_mae, idx_row=2)

    fig2.savefig(Path(save_dir) / "predictions.png")
    plt.close(fig2)

    ########################
    # 3. Plot Ensemble Variance
    ########################
    vp_var = np.var(vp_ensemble, axis=0)
    vs_var = np.var(vs_ensemble, axis=0)
    rho_var = np.var(rho_ensemble, axis=0)

    fig3 = plt.figure(figsize=(15, 4))

    ax1 = fig3.add_subplot(1, 3, 1)
    im1 = ax1.imshow(vp_var.squeeze(), cmap="plasma")
    ax1.set_title("VP Ensemble Variance")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1)

    ax2 = fig3.add_subplot(1, 3, 2)
    im2 = ax2.imshow(vs_var.squeeze(), cmap="plasma")
    ax2.set_title("VS Ensemble Variance")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2)

    ax3 = fig3.add_subplot(1, 3, 3)
    im3 = ax3.imshow(rho_var.squeeze(), cmap="plasma")
    ax3.set_title("RHO Ensemble Variance")
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3)

    fig3.tight_layout()
    fig3.savefig(Path(save_dir) / "ensemble_variance.png")
    plt.close(fig3)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """
    Generate predictions using the trained diffusion FWI model and regression model.
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    logger = PythonLogger("generate")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging("generate.log")

    device = dist.device
    rank_zero_logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.generation.seed)
    np.random.seed(cfg.generation.seed)


    # Load diffusion model
    diffusion_checkpoint_path = to_absolute_path(
        cfg.model.diffusion_checkpoint_path
    )
    rank_zero_logger.info(
        f"Loading diffusion model from {diffusion_checkpoint_path}"
    )
    diffusion_net = Module.from_checkpoint(diffusion_checkpoint_path)
    diffusion_net = diffusion_net.eval().requires_grad_(False).to(device)
    diffusion_net = ConditionalDiffusionAdapter(
        model= diffusion_net,
        args_map=("x", {"x_mean": "x_mean", "noise": "noise", "ux": "ux", "uz": "uz"}),
    )
    diffusion_edm_net = partial(edm_precond, model=diffusion_net, sigma_data=0.5)
    rank_zero_logger.info("Diffusion model loaded successfully")

    # Prepare sampler for posterior diffusion model

    scale = 1
    sampler_fn = partial(
                posteriori_sampler,
                num_steps=cfg.sampler.num_steps,
                sigma_min=cfg.sampler.sigma_min,
                sigma_max=cfg.sampler.sigma_max,
                physics_informed = True,
                std = 1e-1,gamma_cond=1, mu=1,scale=scale,power=2
            )

    # Create output directory
    if scale > 0:
        output_path = Path(to_absolute_path(cfg.io.output_dir+"_marmosi_pi"))
    else:
        output_path = Path(to_absolute_path(cfg.io.output_dir+"_marmosi"))
    output_path.mkdir(parents=True, exist_ok=True)

    #Stats
    vp_mean =  3035.069357508522; vp_std = 890.3956
    vs_mean = 1712.469452191763; vs_std = 551.9505919227604
    # Main generation loop
    marm_vp = np.load("../data/raw_dataset/marmosi/marm_vp.npy")[::15,::10]*1000#*vp_mean/marm_vp.mean()
    marm_vs = np.load("../data/raw_dataset/marmosi/marm_vs.npy")[::15,::10]*1000#*vs_mean/marm_vs.mean()
    
    patch_size=70
    marm_dataset = []
    for i in range(5):
        data = {}
        row = np.random.randint(100, marm_vp.shape[0] - patch_size)
        col = np.random.randint(100, marm_vs.shape[1] - patch_size)
        patch_vp = marm_vp[row:row+patch_size, col:col+patch_size]
        patch_vs = marm_vs[row:row+patch_size, col:col+patch_size]
        patch_vp_mean =  patch_vp.mean(); patch_vp_std = patch_vp.std()
        patch_vs_mean = patch_vs.mean(); patch_vs_std = patch_vs.std()
        patch_vp = ((patch_vp - patch_vp_mean)/patch_vp_std ) *vp_std + vp_mean
        patch_vs = ((patch_vs - patch_vs_mean)/patch_vs_std ) *vs_std + vs_mean
        lith, alpha, beta, salt_mask = classify_lithology(patch_vp, patch_vs)
        patch_rho = compute_density(patch_vp, alpha, beta, salt_mask)
        patch_vp = torch.tensor(patch_vp).float().to(device)
        patch_vs = torch.tensor(patch_vs).float().to(device)
        patch_rho = torch.tensor(patch_rho).float().to(device)
        # ############### Acquisition Geometry #################
        ny, nx = 70, 70
        dx = 5.0
        nt = 1000
        dt = 0.001
        freq = 15
        peak_time = 1.5 / freq
        n_shots = 5
        source_depth = 1
        receiver_depth = 1
        n_receivers_per_shot = 69

        source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = torch.arange(n_shots) * 17

        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = torch.arange(n_receivers_per_shot).repeat(n_shots, 1)

        source_amplitudes = deepwave.wavelets.ricker(freq, nt, dt, peak_time).repeat(n_shots, 1, 1).to(device) * 100000.0
        vz, vx = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(patch_vp, patch_vs, patch_rho),
            grid_spacing=dx, dt=dt,
            source_amplitudes_y=source_amplitudes, source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations, source_locations_x=source_locations,
            receiver_locations_y=receiver_locations, receiver_locations_x=receiver_locations,
            pml_freq=freq, pml_width=[20, 20, 20, 20]
        )[-2:]
        data['vp'] = (patch_vp[None,None,:]-vp_mean)/vp_std
        data['vs'] = (patch_vs[None,None,:]-vs_mean)/vs_std
        data['rho'] = patch_rho[None,None,:]
        data['vx'] = vx[None,:]
        data['vz'] = vz[None,:]
        marm_dataset.append(data)

    rank_zero_logger.info("Data loaded!! Starting generation...")
    for i, data in enumerate(marm_dataset):
        # Extract data
        ux = torch.nn.functional.pad(data["vx"].permute(0, 1, 3, 2),pad=(0,1))
        uz = torch.nn.functional.pad(data["vz"].permute(0, 1, 3, 2),pad=(0,1))
        vp_target, vs_target, rho_target = data["vp"], data["vs"], data["rho"]
        # Combine inputs for processing
        img_lr = torch.cat([ux, uz], dim=1)
        target = torch.cat([vp_target,vs_target,rho_target],dim=1)
        # Run diffusion model with ensemble generation
        # Define latent shapes for generation
        latents_shape = (
            1,  # Batch size
            3,  # Channels (vp, vs, rho)
            cfg.dataset.subsurface_resolution[0],
            cfg.dataset.subsurface_resolution[1],
        )

        # Use regression output as conditioning if available
        x_mean = None
        # model_fn = partial(edm_precond, model=diffusion_net, sigma_data=0.5)
        # Generate ensemble predictions
        start_time  = time.time()
        ensemble_outputs = diffusion_step(
            net=diffusion_edm_net,
            sampler_fn=sampler_fn,
            latents_shape=latents_shape,
            img_lr=img_lr,
            device=device,
            x_tar=target, 
            ensemble_size=cfg.generation.num_ensembles,
            x_mean=x_mean,
            seed=cfg.generation.seed + i,  # Different seed for each sample
        )

        # Split ensemble outputs
        ensemble_vp = ensemble_outputs[:, 0:1]
        ensemble_vs = ensemble_outputs[:, 1:2]
        ensemble_rho = ensemble_outputs[:, 2:3]

        # Calculate mean for final prediction
        diffusion_vp = ensemble_vp.mean(dim=0, keepdim=True)
        diffusion_vs = ensemble_vs.mean(dim=0, keepdim=True)
        diffusion_rho = ensemble_rho.mean(dim=0, keepdim=True)

        # Combine outputs based on inference mode
        final_vp, final_vs = diffusion_vp, diffusion_vs
        final_vp = final_vp * vp_std + vp_mean
        final_vs = final_vs * vs_std + vs_mean
        final_rho = diffusion_rho
        vp_target = vp_target * vp_std + vp_mean
        vs_target = vs_target * vs_std + vs_mean
        ensemble_vp = ensemble_vp * vp_std + vp_mean if ensemble_vp is not None else None
        ensemble_vs = ensemble_vs * vs_std + vs_mean if ensemble_vs is not None else None
        # Calculate metrics
        vp_rmse = RMSE(final_vp, vp_target)
        vp_mae = MAE(final_vp, vp_target)
        rho_mae = MAE(diffusion_rho,rho_target)
        rho_rmse = RMSE(diffusion_rho,rho_target)
        vs_rmse = RMSE(final_vs, vs_target)
        vs_mae = MAE(final_vs, vs_target)

        # Print metrics
        rank_zero_logger.info(f"Sample {i}:")
        rank_zero_logger.info(f"  VP - RMSE: {vp_rmse:.6f}, MAE: {vp_mae:.6f}")
        rank_zero_logger.info(f"  VS - RMSE: {vs_rmse:.6f}, MAE: {vs_mae:.6f}")
        rank_zero_logger.info(f"  rho - RMSE: {rho_rmse:.6f}, MAE: {rho_mae:.6f}")
        rank_zero_logger.info(
            f"  Generation time: {time.time() - start_time:.2f} seconds"
        )

        # Prepare data for plotting
        inputs = {"vx": ux, "vz": uz}
        targets = {"vp": vp_target, "vs": vs_target, "rho": rho_target}
        predictions = {
            "vp": final_vp,
            "vs": final_vs,
            "rho": final_rho,
            "vp_diffusion": diffusion_vp,
            "vs_diffusion": diffusion_vs,
            "rho_diffusion": diffusion_rho,
        }

        # Add ensemble outputs if available
        if ensemble_vp is not None and ensemble_vs is not None:
            predictions["vp_ensemble"] = ensemble_vp
            predictions["vs_ensemble"] = ensemble_vs
            predictions["rho_ensemble"] = ensemble_rho

        metrics = {
            "vp_rmse": vp_rmse,
            "vp_mae": vp_mae,
            "rho_rmse": rho_rmse,
            "rho_mae": rho_mae,
            "vs_rmse": vs_rmse,
            "vs_mae": vs_mae,
        }

        # Plot results
        if cfg.io.plot_results:
            sample_dir = output_path / f"sample_{i}"
            plot_prediction(
                inputs, targets, predictions, metrics, sample_dir, sources_to_plot=3
            )

        # Save raw numpy arrays
        if cfg.io.save_numpy:
            np_dir = output_path / f"sample_{i}" / "numpy"
            np_dir.mkdir(parents=True, exist_ok=True)

            # Save predictions
            np.save(np_dir / "vp_pred.npy", final_vp.cpu().numpy())
            np.save(np_dir / "vs_pred.npy", final_vs.cpu().numpy())
            np.save(np_dir / "rho_pred.npy", final_rho.cpu().numpy())
            np.save(np_dir / "rho_target.npy", rho_target.cpu().numpy())
            np.save(np_dir / "vp_target.npy", vp_target.cpu().numpy())
            np.save(np_dir / "vs_target.npy", vs_target.cpu().numpy())
            np.save(np_dir / "vx.npy", ux.cpu().numpy())
            np.save(np_dir / "vz.npy", uz.cpu().numpy())

            # Save ensemble predictions if available
            if ensemble_vp is not None:
                np.save(np_dir / "vp_ensemble.npy", ensemble_vp.cpu().numpy())
                np.save(np_dir / "vs_ensemble.npy", ensemble_vs.cpu().numpy())
                np.save(np_dir / "rho_ensemble.npy", ensemble_rho.cpu().numpy())

            # Save individual model contributions if available
            if diffusion_vp is not None:
                np.save(np_dir / "vp_diffusion.npy", diffusion_vp.cpu().numpy())
                np.save(np_dir / "vs_diffusion.npy", diffusion_vs.cpu().numpy())
                np.save(np_dir / "rho_diffusion.npy", diffusion_rho.cpu().numpy())

    rank_zero_logger.info("Generation completed!")


if __name__ == "__main__":
    main()
