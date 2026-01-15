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

import asyncio
import datetime
import csv
import os
import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

from physicsnemo.utils.zenith_angle import cos_zenith_angle


class HRRRSurfaceDataset(Dataset):
    """HRRR Surface dataset on S3

    Parameters
    ----------
    zarr_url : str
        URL to Zarr group (e.g., s3://bucket/path)
    storage_options : dict, optional
        Backend/storage kwargs passed to Zarr opener (e.g., endpoint_url)
    time_indices : np.array
        Index array of times to use as part of dataset
    stats_csv : str, optional
        Stats CSV location, by default "stats/stats.csv"
    """

    VARIABLES = [
        "u10m",
        "v10m",
        "u80m",
        "v80m",
        "t2m",
        "d2m",
        "q2m",
        "sp",
        "fg10m",
        "tcc",
        "sde",
        "snowc",
        "refc",
        "rsds",
        "tp",
        "aerot",
    ]
    LOG_VARIABLES = ("tp", "aerot")  # Make sure is consistent with stats CSV
    EPSILON = 1e-8

    def __init__(
        self,
        zarr_url: str,
        time_indices: np.array,
        stats_csv: str = "stats/stats.csv",
        storage_options: dict | None = None,
    ):
        self.zarr_url = zarr_url
        self.storage_options = storage_options or {}
        self.idx = np.asarray(time_indices, dtype=int).ravel()

        # Verify bounds against available time coordinate in zarr
        _root = zarr.open_group(
            store=self.zarr_url, mode="r", storage_options=self.storage_options
        )
        n_time = _root["time"].size
        if np.any((self.idx < 0) | (self.idx >= n_time)):
            invalid_values = np.unique(self.idx[out_of_bounds_mask])
            raise IndexError(
                "time_indices contain out-of-bounds values for zarr_root['time']"
            )

        # Load normalization stats and log-scaling flags from summary_stats.csv
        stats_csv = os.path.join(os.path.dirname(__file__), stats_csv)
        means = []
        stds = []
        stats_map = {}
        with open(stats_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                var = row.get("variable")
                mu = float(row.get("mean", "nan"))
                sd = float(row.get("std", "nan"))
                stats_map[var] = (mu, sd)

        # Order based on VARIABLES
        for var in self.VARIABLES:
            mu, sd = stats_map[var]
            means.append(mu)
            stds.append(sd)

        # Instance-level overrides for normalization and log variables
        self.target_means = (
            torch.tensor(means, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        )
        self.target_stds = (
            torch.tensor(stds, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        )
        # Save zarr coords to memory for use
        self.grid_lat = _root["lat"][:]
        self.grid_lon = _root["lon"][:]
        self.time_array = _root["time"][:]

    def __len__(self):
        return self.idx.shape[0]

    async def _zarr_read(
        self,
        root,
        array_name: str,
        array_idx: int,
        time_idx: int,
        data_arrays: np.array,
    ):
        arr = await root.get(array_name)
        arr = await arr.getitem((time_idx, slice(None), slice(None)))
        if array_name in self.LOG_VARIABLES:
            data_arrays[array_idx] = np.log(arr + self.EPSILON)
        else:
            data_arrays[array_idx] = arr

    async def _get_array(self, idx):
        root = await zarr.api.asynchronous.open_group(
            self.zarr_url, mode="r", storage_options=self.storage_options
        )

        time_idx = self.idx[idx]
        data_arrays = np.empty(
            (len(self.VARIABLES), self.grid_lat.shape[0], self.grid_lat.shape[1])
        )
        jobs = []
        for i, t in enumerate(self.VARIABLES):
            jobs.append(self._zarr_read(root, t, i, time_idx, data_arrays))
        await asyncio.gather(*jobs)
        return data_arrays

    def __getitem__(self, idx):
        time_idx = self.idx[idx]
        time_stamp = self.time_array[time_idx]
        data_arrays = asyncio.run(self._get_array(idx))

        target = torch.Tensor(data_arrays)
        target = (target - self.target_means) / self.target_stds
        # Conditional encoding
        data_arrays = np.empty(
            (4, self.grid_lat.shape[0], self.grid_lat.shape[1]), dtype=np.float32
        )
        ts = (time_stamp - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(
            1, "s"
        )
        data_arrays[0] = cos_zenith_angle(
            datetime.datetime.utcfromtimestamp(ts), self.grid_lat, self.grid_lon
        )
        data_arrays[1][:] = np.sin(
            2 * np.pi *int(
                (
                    time_stamp.astype("datetime64[D]")
                    - time_stamp.astype("datetime64[Y]")
                ).astype(int)
                + 1
            )
            / 365.0
        )
        data_arrays[2] = self.grid_lat / 90.0
        data_arrays[3] = self.grid_lon / 360.0

        condition = torch.Tensor(data_arrays)
        return condition, target


if __name__ == "__main__":
    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v2",
        mode="r",
        storage_options={"endpoint_url": "https://pdx.s8k.io"},
    )
    time = root["time"][:]
    sidx = np.where(time == np.datetime64("2023-01-01T00:00:00"))[0][0]
    eidx = np.where(time == np.datetime64("2023-02-01T00:00:00"))[0][0]

    time_idx = np.arange(sidx, eidx)

    dataset = HRRRSurfaceDataset(root, time_idx)
    cond, target = dataset[30]

    print(cond.shape)
    print(target)
