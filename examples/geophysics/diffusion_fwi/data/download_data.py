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

from pathlib import Path
import sys
import numpy as np
import argparse
import yaml
import math
import requests
import os
from tqdm import tqdm
import subprocess

# URLs for HuggingFace dataset parts
HF_BASE_URL = "https://huggingface.co/datasets/ashynf/EFWI/resolve/main"
DATASETS = {
    "CFB": [
        f"{HF_BASE_URL}/CFB/CFB_split.zip",
        f"{HF_BASE_URL}/CFB/CFB_split.z01",
        f"{HF_BASE_URL}/CFB/CFB_split.z02",
        f"{HF_BASE_URL}/CFB/CFB_split.z03",
    ],
    "CFA": [
        f"{HF_BASE_URL}/CFA/CFA_split.zip",
        f"{HF_BASE_URL}/CFA/CFA_split.z01",
        f"{HF_BASE_URL}/CFA/CFA_split.z02",
        f"{HF_BASE_URL}/CFA/CFA_split.z03",
    ],
    "CVA": [
        f"{HF_BASE_URL}/CVA/CVA_split.zip",
        f"{HF_BASE_URL}/CVA/CVA_split.z01",
        f"{HF_BASE_URL}/CVA/CVA_split.z02",
    ],
    "CVB": [
        f"{HF_BASE_URL}/CVB/CVB_split.zip",
        f"{HF_BASE_URL}/CVB/CVB_split.z01",
        f"{HF_BASE_URL}/CVB/CVB_split.z02",
    ],
    "FFA": [
        f"{HF_BASE_URL}/FFA/FFA_split.zip",
        f"{HF_BASE_URL}/FFA/FFA_split.z01",
        f"{HF_BASE_URL}/FFA/FFA_split.z02",
        f"{HF_BASE_URL}/FFA/FFA_split.z03",
    ],
    "FFB": [
        f"{HF_BASE_URL}/FFB/FFB_split.zip",
        f"{HF_BASE_URL}/FFB/FFB_split.z01",
        f"{HF_BASE_URL}/FFB/FFB_split.z02",
        f"{HF_BASE_URL}/FFB/FFB_split.z03",
    ],
    "FVA": [
        f"{HF_BASE_URL}/FVA/FVA_split.zip",
        f"{HF_BASE_URL}/FVA/FVA_split.z01",
        f"{HF_BASE_URL}/FVA/FVA_split.z02",
    ],
    "FVB": [
        f"{HF_BASE_URL}/FVB/FVB_split.zip",
        f"{HF_BASE_URL}/FVB/FVB_split.z01",
        f"{HF_BASE_URL}/FVB/FVB_split.z02",
    ],
}
DATASETS_INFO = {
    "CFB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "CFA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "FFB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "FFA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
    "CVA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "CVB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "FVA": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
    "FVB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 24000, "TEST_SAMPLES": 6000},
}


def download_file_from_url(url: str, local_filename: str) -> str:
    """
    Download a file from a direct URL and save it locally.

    Parameters
    ----------
    url : str
        The URL to download from.
    local_filename : str
        The path to save the file to.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    print(f"Downloading {os.path.basename(local_filename)} from {url}...")

    response = requests.head(url)
    file_size = int(response.headers.get("content-length", 0))

    progress = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(local_filename),
    )

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    progress.close()

    return local_filename


def download(name: str) -> None:
    """
    Download a dataset from Hugging Face.

    Parameters
    ----------
    name : str
        Name of the dataset to download.
    """
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")

    output_dir = Path(f"./{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all parts of the zip archive
    zip_parts = []
    for url in DATASETS[name]:
        filename = os.path.basename(url)
        output_path = output_dir / filename
        download_file_from_url(url, output_path)
        zip_parts.append(output_path)
    print(f"All parts of {name} dataset downloaded successfully.")

    # Combine multi-part zip archive into a single file
    print(f"Combining zip parts for {name} dataset...")
    combined_zip = output_dir / "_temp_combined.zip"
    try:
        subprocess.run(
            [
                "zip",
                "-s",
                "0",
                str(zip_parts[0]),
                "--out",
                str(combined_zip),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to combine zip parts") from exc

    # Remove original zip parts
    for zip_part in zip_parts:
        zip_part.unlink()

    # Extract the combined zip archive
    print(f"Extracting {name} dataset...")
    try:
        subprocess.run(
            ["unzip", str(combined_zip), "-d", str(output_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to extract combined zip archive") from exc

    # Cleanup combined archive
    combined_zip.unlink()

    print(f"Download and extraction of {name} dataset completed.")


def preprocess(
    dataset_names: list[str], clean: bool = False, shuffle: bool = False
) -> None:
    """
    Preprocess the dataset by:
    1. Reorganizing files into individual samples
    2. Computing statistics (mean, std, min, max) for training, test, and all data
    3. Saving statistics to a stats.yaml file

    Parameters
    ----------
    dataset_names : list[str]
        List of dataset names to preprocess.
    clean : bool, optional
        Whether to delete original files after processing.
    shuffle : bool, optional
        Whether to shuffle train and test samples.
    """
    # Process all datasets if 'all' is in the list
    if "all" in dataset_names:
        names = list(DATASETS.keys())
    else:
        # Validate dataset names
        for name in dataset_names:
            if name not in DATASETS:
                raise ValueError(f"Unsupported dataset: {name}")
        names = dataset_names

    for name in names:
        print(f"Preprocessing {name} dataset...")
        dataset_dir = Path(f"./{name}")
        samples_dir = dataset_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        # Get list of file indices by looking at vs files
        vs_files = sorted(
            dataset_dir.glob("vs_*.npy"), key=lambda x: int(x.stem.split("_")[-1])
        )

        if not vs_files:
            print(f"No files found for dataset {name}. Skipping preprocessing.")
            continue

        # Keep track of processed files to delete later
        processed_files = set()

        # Hardcoded sizes of the dataset
        train_samples = DATASETS_INFO[name]["TRAIN_SAMPLES"]
        test_samples = DATASETS_INFO[name]["TEST_SAMPLES"]
        total_samples = train_samples + test_samples
        samples_per_file = DATASETS_INFO[name]["SAMPLES_PER_FILE"]
        train_files = train_samples // samples_per_file

        # Setup shuffling
        if shuffle:
            print("Shuffling enabled.")
            np.random.seed(123)
            # Generate a random set of file indices from 0 to total_samples-1
            random_indices = np.random.permutation(total_samples).tolist()
        else:
            # If not shuffling, just use sequential indices
            random_indices = np.arange(total_samples).tolist()

        # Initialize dictionaries to accumulate data for statistics
        train_data = {
            "sum_vs": 0,
            "sum_vp": 0,
            "sum_ux": 0,
            "sum_uz": 0,
            "sum_vs2": 0,
            "sum_vp2": 0,
            "sum_ux2": 0,
            "sum_uz2": 0,
            "min_vs": float("inf"),
            "min_vp": float("inf"),
            "min_ux": float("inf"),
            "min_uz": float("inf"),
            "max_vs": float("-inf"),
            "max_vp": float("-inf"),
            "max_ux": float("-inf"),
            "max_uz": float("-inf"),
        }
        test_data = {
            "sum_vs": 0,
            "sum_vp": 0,
            "sum_ux": 0,
            "sum_uz": 0,
            "sum_vs2": 0,
            "sum_vp2": 0,
            "sum_ux2": 0,
            "sum_uz2": 0,
            "min_vs": float("inf"),
            "min_vp": float("inf"),
            "min_ux": float("inf"),
            "min_uz": float("inf"),
            "max_vs": float("-inf"),
            "max_vp": float("-inf"),
            "max_ux": float("-inf"),
            "max_uz": float("-inf"),
        }

        # Process each file
        for file_idx, vs_file in enumerate(
            tqdm(vs_files, desc="Processing files", unit="file")
        ):
            file_num = int(vs_file.stem.split("_")[-1])

            # Define file paths
            vs_file_path = dataset_dir / f"vs_{file_num}.npy"
            vp_file_path = dataset_dir / f"vp_{file_num}.npy"
            data_x_file_path = dataset_dir / f"data_x_{file_num}.npy"
            data_z_file_path = dataset_dir / f"data_z_{file_num}.npy"

            # Add files to processed list
            processed_files.add(vs_file_path)
            processed_files.add(vp_file_path)
            processed_files.add(data_x_file_path)
            processed_files.add(data_z_file_path)

            # Load all quantities for this file index
            try:
                vs_data = np.load(vs_file_path, mmap_mode="r")
                vp_data = np.load(vp_file_path, mmap_mode="r")
                data_x = np.load(data_x_file_path, mmap_mode="r")
                data_z = np.load(data_z_file_path, mmap_mode="r")
            except FileNotFoundError as e:
                print(f"Error loading files for index {file_num}: {e}")
                continue

            # Get number of samples in this file
            num_samples = len(vs_data)

            # Create individual files for each sample
            for sample_idx in range(num_samples):
                sample_data = {
                    "vs": vs_data[sample_idx],
                    "vp": vp_data[sample_idx],
                    "ux": data_x[sample_idx],
                    "uz": data_z[sample_idx],
                }

                # Get next file index and remove it
                rnd_global_sample_idx = random_indices.pop(0)

                # Save the combined sample data
                sample_file = samples_dir / f"sample_{rnd_global_sample_idx}.npz"
                np.savez(sample_file, **sample_data)

                # Determine if the sample is for training or testing
                is_train = rnd_global_sample_idx < train_samples

                # Collect data for statistics
                data_dict = train_data if is_train else test_data

                for var, value in zip(
                    ("vs", "vp", "ux", "uz"),
                    (
                        sample_data["vs"],
                        sample_data["vp"],
                        sample_data["ux"],
                        sample_data["uz"],
                    ),
                ):
                    nb_points = math.prod(value.shape)
                    data_dict[f"sum_{var}"] += np.sum(value) / nb_points
                    data_dict[f"sum_{var}2"] += np.sum(value**2) / nb_points
                    data_dict[f"min_{var}"] = min(
                        data_dict[f"min_{var}"], np.amin(value)
                    )
                    data_dict[f"max_{var}"] = max(
                        data_dict[f"max_{var}"], np.amax(value)
                    )

                del sample_data

            # Explicitly delete memory-mapped arrays to free resources
            del vs_data, vp_data, data_x, data_z

            # Delete the original files after processing only if clean is True
            if clean:
                vs_file_path.unlink()
                vp_file_path.unlink()
                data_x_file_path.unlink()
                data_z_file_path.unlink()

                # Find and delete other files with the same index pattern
                # (like pm_* and pr_*)
                other_pattern_files = list(dataset_dir.glob(f"*_{file_num}.npy"))
                for file_path in other_pattern_files:
                    if file_path not in processed_files:
                        try:
                            file_path.unlink()
                        except FileNotFoundError:
                            pass
                print("All original files have been deleted.")
        print(f"Reorganization of {name} dataset completed.")

        # Compute statistics for this dataset
        print("Computing statistics...")
        stats = {}
        for var in ["vs", "vp", "ux", "uz"]:
            # Create nested dictionary for each variable
            train_mean = float(train_data[f"sum_{var}"] / train_samples)
            train_std = float(
                math.sqrt(train_data[f"sum_{var}2"] / train_samples - train_mean**2)
            )

            test_mean = float(test_data[f"sum_{var}"] / test_samples)
            test_std = float(
                math.sqrt(test_data[f"sum_{var}2"] / test_samples - test_mean**2)
            )

            all_samples = train_samples + test_samples
            all_mean = float(
                (train_data[f"sum_{var}"] + test_data[f"sum_{var}"]) / all_samples
            )
            all_std = float(
                math.sqrt(
                    (train_data[f"sum_{var}2"] + test_data[f"sum_{var}2"]) / all_samples
                    - all_mean**2
                )
            )

            stats[var] = {
                "train": {
                    "min": float(train_data[f"min_{var}"]),
                    "max": float(train_data[f"max_{var}"]),
                    "mean": train_mean,
                    "std": train_std,
                },
                "test": {
                    "min": float(test_data[f"min_{var}"]),
                    "max": float(test_data[f"max_{var}"]),
                    "mean": test_mean,
                    "std": test_std,
                },
                "all": {
                    "min": float(
                        min(train_data[f"min_{var}"], test_data[f"min_{var}"])
                    ),
                    "max": float(
                        max(train_data[f"max_{var}"], test_data[f"max_{var}"])
                    ),
                    "mean": all_mean,
                    "std": all_std,
                },
            }

        # Save statistics to YAML file
        stats_file = dataset_dir / "stats.yaml"
        print(f"Saving statistics to {stats_file}...")

        with open(stats_file, "w") as f:
            yaml.dump(stats, f, default_flow_style=False)

        print(f"Preprocessing of {name} dataset completed successfully!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess EFWI datasets."
    )

    parser.add_argument("--download", action="store_true", help="Download the dataset")

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess the dataset: reorganize files and compute statistics",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete original files after processing",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle train and test samples using fixed random seed 123",
    )

    parser.add_argument(
        "--name",
        nargs="+",
        default=["all"],
        help="Names of datasets to process (default: all)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate dataset names
    if "all" not in args.name:
        for name in args.name:
            if name not in DATASETS:
                print(f"Error: Unknown dataset '{name}'")
                print(f"Available datasets: {', '.join(DATASETS.keys())}")
                sys.exit(1)

    # Check if at least one action is specified
    if not (args.download or args.preprocess):
        print("Error: No action specified. Use --download or --preprocess")
        sys.exit(1)

    # Download if requested
    if args.download:
        if "all" in args.name:
            for dataset in DATASETS.keys():
                download(dataset)
        else:
            for dataset in args.name:
                download(dataset)

    # Preprocess if requested
    if args.preprocess:
        preprocess(args.name, args.clean, args.shuffle)
