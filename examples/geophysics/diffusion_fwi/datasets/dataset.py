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


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

# Constants for each dataset
_DATASETS = {
    "CFB": {"SAMPLES_PER_FILE": 500, "TRAIN_SAMPLES": 48000, "TEST_SAMPLES": 6000},
}


def _get_list_files(dir: Path, pattern: str):
    return sorted(dir.glob(pattern), key=lambda x: int(x.stem.split("_")[-1]))


def _read_npz_sample(filename: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Read a single .npz file containing multiple fields (vs, vp, ux, uz).

    Args:
        filename: Path to the .npz file

    Returns:
        Dictionary containing the data fields
    """
    with np.load(filename) as data:
        # Create a copy of the data to avoid issues with the file being closed
        return {key: data[key] for key in data.keys()}


def _read_stats_file(
    filename: Union[str, Path]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Read a stats YAML file containing statistics for the dataset.
    """
    with open(filename, "r") as f:
        stats = yaml.safe_load(f)
    return stats


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "EFWIDatapipe"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


# TODO: add option for 2D/3D
# TODO: any data augmentation possible? Maybe symmetry?
# TODO: implement data loading + getitem for multiple datasets (e.g. CFB, CVA,
# etc... combined into one dataset) --> Probably better to have higher level
# DatasetMerger(dset1, dset2, ...)
class EFWIDataset(Dataset):
    """
    Dataset for E-FWI.
    """

    def __init__(
        self,
        name: Literal["CFB"],
        data_dir: Union[str, Path],
        phase: Literal["train", "test"],
    ) -> None:
        # Safety checks and input pre-processing
        if name not in _DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir.expanduser() / name / "samples"
        if not self.data_dir.exists():
            raise AssertionError(f"Path {self.data_dir} does not exist")
        if not self.data_dir.is_dir():
            raise AssertionError(f"Path {self.data_dir} is not a directory")
        if phase not in ["train", "test"]:
            raise AssertionError(
                f"phase should be one of ['train', 'test'], got {phase}"
            )
        self.name = name
        self.phase = phase
        # Some constants
        self._SAMPLES_PER_FILE = _DATASETS[self.name]["SAMPLES_PER_FILE"]
        self._TRAIN_SAMPLES = _DATASETS[self.name]["TRAIN_SAMPLES"]
        self._TEST_SAMPLES = _DATASETS[self.name]["TEST_SAMPLES"]

        # Get list of sample files
        if not self.data_dir.exists():
            raise AssertionError(
                f"Samples directory {self.data_dir} does not exist. "
                f"Please run the reorganize script first."
            )

        all_sample_files = sorted(
            self.data_dir.glob("sample_*.npz"), key=lambda x: int(x.stem.split("_")[-1])
        )

        # Select train or test samples
        if self.phase == "train":
            self.sample_files = all_sample_files[: self._TRAIN_SAMPLES]
        else:
            self.sample_files = all_sample_files[
                self._TRAIN_SAMPLES : self._TRAIN_SAMPLES + self._TEST_SAMPLES
            ]

        # Get first element to get the keys
        self.variables = list(self[0].keys())

        # Load dataset statistics
        stats_file = self.data_dir.parent / "stats.yaml"
        if stats_file.exists():
            self.stats = _read_stats_file(stats_file)
        else:
            self.stats = None
            print(
                f"Warning: Stats file {stats_file} not found. Normalization will not be available."
            )

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Read and load samples from .npz files.
        Each .npz file contains multiple fields (vs, vp, ux, uz).

        Args:
            idx: Index or indices of samples to load

        Returns:
            Dictionary of tensors with keys 'vs', 'vp', 'ux', 'uz'
        """

        # Initialize data dictionary
        data = {}

        # Load the sample file
        sample_data = _read_npz_sample(self.sample_files[idx])

        # Initialize data arrays on first iteration
        for key, value in sample_data.items():
            data[key] = torch.tensor(value, dtype=torch.float32, device="cpu")

        return data


class EFWIDatapipe(Datapipe, DataLoader):
    """
    Datapipe for E-FWI
    """

    def __init__(
        self,
        name: Literal["CFB"],
        data_dir: Union[str, Path],
        phase: Literal["train", "test"],
        batch_size_per_device: int,
        seed: int = 0,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        prefetch_factor: Optional[int] = 2,
        use_sharding: Optional[bool] = None,
    ) -> None:

        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        super().__init__(meta=MetaData())
        dataset = EFWIDataset(name=name, data_dir=data_dir, phase=phase)
        self.phase = phase

        # Determine whether to use sharding
        should_shard = use_sharding if use_sharding is not None else True

        if should_shard and world_size > 1:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=process_rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
            shuffle = None
        else:
            sampler = None

        DataLoader.__init__(
            self,
            dataset=dataset,
            batch_size=batch_size_per_device,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(pin_memory := device.type == "cuda"),
            shuffle=shuffle,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
            prefetch_factor=prefetch_factor,
            pin_memory_device=(str(self.device) if pin_memory else ""),
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for the datapipe. Used for shuffling in distributed
        training.
        """
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def get_stats(
        self, metric: str, phase: Literal["train", "test"] = "train"
    ) -> Dict[str, float]:
        """Return training statistics for each variable.

        Parameters
        ----------
        - metric : str
            One of "mean", "std", "min", "max" corresponding to the
            statistics stored in the YAML file.
        - phase : Literal["train", "test"]
            The phase to get the statistics for.
        """

        if self.dataset.stats is None:
            raise RuntimeError("Statistics file not available for dataset.")

        if metric not in {"mean", "std", "min", "max"}:
            raise ValueError(f"Unknown metric '{metric}'.")

        if phase not in {"train", "test"}:
            raise ValueError(f"Unknown phase '{phase}'.")

        return {k: v[phase][metric] for k, v in self.dataset.stats.items()}

    def __iter__(self):
        for data in super().__iter__():
            for key in data.keys():
                data[key] = data[key].to(self.device)
            yield data
