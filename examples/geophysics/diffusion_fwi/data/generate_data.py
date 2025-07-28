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

import os
import logging
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

import deepwave
from deepwave import elastic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s"
)


def classify_lithology(
    vp: np.ndarray,
    vs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify lithology element-wise based on Vp, Vs, and Vp/Vs ratio.

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocity in m/s (2D array)
    vs : np.ndarray
        S-wave velocity in m/s (2D array)

    Returns
    -------
    lith : np.ndarray
        Array of rock type strings
    alpha : np.ndarray
        Gardner coefficients
    beta : np.ndarray
        Gardner coefficients
    salt_mask : np.ndarray
        Boolean mask to override density with fixed value

    .. note::

        Adapted from:
        - `Gardner et al. (1974), Geophysics, <https://doi.org/10.1190/1.1440465>`_
        - Mavko et al. (2009), "The Rock Physics Handbook"
        - Castagna et al. (1985), Geophysics, 50(4), 571-581
        - Gray & Head (2000), "Modeling, migration, and velocity analysis in salt", Geophysics
    """

    vpr = vp / vs  # Vp/Vs ratio
    lith = np.full(vp.shape, "Unknown", dtype=object)
    alpha = np.zeros_like(vp, dtype=float)
    beta = np.zeros_like(vp, dtype=float)

    # Lithology Masks
    shale_mask = vpr > 2.0  # Castagna et al., 1985

    sandstone_mask = (
        (vpr >= 1.6) & (vpr <= 2.2) & (vp >= 2500) & (vp <= 5000) & ~shale_mask
    )

    limestone_mask = (
        (vp >= 5000)
        & (vp <= 6500)
        & (vpr >= 1.7)
        & (vpr <= 1.95)
        & ~shale_mask
        & ~sandstone_mask
    )

    dolomite_mask = (
        (vp >= 5500)
        & (vp <= 7000)
        & (vpr >= 1.65)
        & (vpr <= 1.9)
        & ~shale_mask
        & ~sandstone_mask
        & ~limestone_mask
    )

    coal_mask = (
        (vp < 3600)
        & (vpr > 1.8)
        & ~shale_mask
        & ~sandstone_mask
        & ~limestone_mask
        & ~dolomite_mask
    )

    anhydrite_mask = (
        (vp >= 5800)
        & (vp <= 6800)
        & (vpr <= 1.8)
        & ~shale_mask
        & ~dolomite_mask
        & ~limestone_mask
    )

    # Salt: Vp ~4500 m/s, Vs ≈ 0 → large Vp/Vs
    salt_mask = (
        (vp >= 4300)
        & (vp <= 4700)
        & (vs < 700)
        & (vpr >= 6.0)  # Vs near zero  # Very high Vp/Vs
    )

    # Assign Lithologies and Gardner Parameters
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


def compute_density(
    vp: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    salt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute density using Gardner's rule.

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocity in m/s
    alpha : np.ndarray
        Gardner coefficients (same shape as vp)
    beta : np.ndarray
        Gardner coefficients (same shape as vp)
    salt_mask : np.ndarray | None
        Optional boolean mask to fix salt density

    Returns
    -------
    rho : np.ndarray
        Estimated density (g/cm³)

    .. note::

        For salt, we override Gardner with a fixed value:
        :math:`\rho = 2.15 \text{ g/cm}^3`
        (Gray & Head, 2000; Mavko et al.)
    """
    rho = alpha * vp**beta
    if salt_mask is not None:
        rho = np.where(salt_mask, 2.15, rho)
    return rho


# Core processing function for a single file
def process_file(filepath: str, device_id: int) -> tuple[str, str]:
    """
    Process a single file and save the results.

    Parameters
    ----------
    filepath : str
        Path to the input file.
    device_id : int
        GPU ID to use for processing. If value provided is not an integer,
        the function will run on CPU.

    Returns
    -------
    tuple[str, str]
        Tuple containing the output file path and a status message.
    """

    try:
        device = (
            torch.device(f"cuda:{device_id}")
            if isinstance(device_id, int)
            else torch.device("cpu")
        )

        original_data = np.load(filepath)
        vp_np = original_data["vp"]
        vs_np = original_data["vs"]

        if vp_np.ndim > 2:
            vp_np = vp_np.reshape(-1, 70, 70)[0]
        if vs_np.ndim > 2:
            vs_np = vs_np.reshape(-1, 70, 70)[0]

        # pr_np = compute_poisson_ratio(vp_np, vs_np)
        # rock_type = infer_rock_type(pr_np)
        # a, b = get_gardner_constants(rock_type)
        # rho_np = gardner_density(vp_np, a, b)

        lith, alpha, beta, salt_mask = classify_lithology(vp_np, vs_np)
        rho_np = compute_density(vp_np, alpha, beta, salt_mask)

        vp = torch.from_numpy(vp_np).float().to(device)
        vs = torch.from_numpy(vs_np).float().to(device)
        rho = torch.from_numpy(rho_np).float().to(device)

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

        receiver_locations = torch.zeros(
            n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
        )
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = torch.arange(n_receivers_per_shot).repeat(
            n_shots, 1
        )

        source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, 1, 1)
            .to(device)
            * 100000.0
        )

        receiver_amplitudes_z, receiver_amplitudes_x = elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            grid_spacing=dx,
            dt=dt,
            source_amplitudes_y=source_amplitudes,
            source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations,
            source_locations_x=source_locations,
            receiver_locations_y=receiver_locations,
            receiver_locations_x=receiver_locations,
            pml_freq=freq,
            pml_width=[20, 20, 20, 20],
        )[-2:]

        output_data = {
            "vp": vp_np,
            "vs": vs_np,
            "rho": rho_np,
            "vx": receiver_amplitudes_x.cpu().numpy(),
            "vz": receiver_amplitudes_z.cpu().numpy(),
        }

        output_dir = os.path.dirname(filepath).replace(
            f"{os.path.sep}samples", f"{os.path.sep}samples_new"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, os.path.basename(filepath))

        np.savez(output_filepath, **output_data)
        return (output_filepath, "Success")

    except Exception as e:
        logging.error(f"FAILED to process {filepath}: {e}", exc_info=True)
        return (filepath, f"Failed: {e}")


def process_file_wrapper(args_tuple):
    """
    Helper function to unpack arguments for use with multiprocessing.Pool's
    imap functions, which only accept a single argument.
    """
    return process_file(*args_tuple)


# Main function to discover files and distribute work
def main():
    """
    Finds all .npz files and distributes them, logging progress every 1000 files.
    """

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Process individual vs and vp samples in .npz files. "
        "Infers lithology and density for each sample and generate "
        "new .npz files with vx and vz."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset directory containing the .npz files.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.path) / "samples"
    file_list = glob.glob(os.path.join(dataset_path, "sample_*.npz"))

    if not file_list:
        logging.warning(f"No files found in {dataset_path}. Please check paths.")
        return

    total_files = len(file_list)
    logging.info(f"Found {total_files} files to process.")

    results = []
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        logging.warning("No GPUs found. Running on CPU. This will be very slow.")
        args = [(filepath, "cpu") for filepath in file_list]

        for i, arg in enumerate(args):
            results.append(process_file(*arg))
            if (i + 1) % 1000 == 0:
                logging.info(f"--- Processed {i + 1} / {total_files} files ---")
    else:
        logging.info(f"Found {num_gpus} GPUs. Starting parallel processing.")
        args = [(filepath, i % num_gpus) for i, filepath in enumerate(file_list)]

        with mp.get_context("spawn").Pool(processes=num_gpus) as pool:
            iterator = pool.imap_unordered(process_file_wrapper, args)

            for i, result in enumerate(iterator):
                results.append(result)
                if (i + 1) % 1000 == 0:
                    logging.info(f"--- Processed {i + 1} / {total_files} files ---")

    success_count = sum(1 for r in results if r[1] == "Success")
    logging.info(f"\n--- Processing Complete ---")
    logging.info(f"{success_count} / {total_files} files processed successfully.")

    failed_files = [r for r in results if r[1] != "Success"]
    if failed_files:
        logging.warning("\n--- Failed Files ---")
        for f, reason in failed_files:
            logging.warning(f"- {os.path.basename(f)}: {reason}")


if __name__ == "__main__":
    main()
