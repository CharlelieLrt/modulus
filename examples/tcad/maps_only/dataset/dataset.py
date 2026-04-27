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

import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from physicsnemo.diffusion.utils.utils import InfiniteSampler

# Maps directory name to thickness in meters
_THICKNESS_METERS: dict[str, float] = {
    "2nm": 2e-9,
    "3nm": 3e-9,
    "4nm": 4e-9,
}


class TCADMapsDataset(Dataset):
    """Sample windows of consecutive TCAD timesteps for autoregressive modeling.

    Use this to train a rollout model on TCAD temperature/potential field maps.
    Each sample is a contiguous window of ``n_steps`` timesteps from one
    simulation. Values are returned in raw physical units (K, V, m, s); apply
    z-score normalization in your training loop using stats from
    :meth:`get_stats`. For batching with DDP-aware infinite sampling, wrap with
    :class:`TCADMapsDataPipe`.

    Parameters
    ----------
    data_dir : path
        Path to the maps_only/data/ directory.
    n_steps : int, optional
        Number of consecutive timesteps per sample window. Default ``2``.
    stats_file : path or None, optional
        Path to a stats.json produced by ``compute_stats.py``. Default ``None``
        (``get_stats()`` then raises).
    thickness : str or None, optional
        If set to one of ``"2nm"``, ``"3nm"``, ``"4nm"``, only simulations
        under that thickness subdirectory are loaded. ``None`` (default) loads
        every available thickness.

    Examples
    --------
    >>> ds = TCADMapsDataset("data/", n_steps=3, stats_file="data/stats.json")
    >>> sample, meta = ds[0]
    >>> sample["variables"].shape          # (S=3 steps, V=2 vars, N points)
    torch.Size([3, 2, 2816])
    >>> T_mean, T_std = ds.get_stats("temperature")
    """

    # Ordered mapping from axis index to variable name
    VARIABLES: tuple[str, ...] = ("temperature", "potential")

    def __init__(
        self,
        data_dir: Path | str,
        n_steps: int = 2,
        stats_file: Path | str | None = None,
        thickness: str | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._n_steps = n_steps

        if thickness is not None and thickness not in _THICKNESS_METERS:
            raise ValueError(
                f"thickness={thickness!r} is not one of "
                f"{sorted(_THICKNESS_METERS)}; pass None to load every thickness."
            )
        self._thickness_filter = thickness

        # Load time arrays from i-vs-time/ (source of truth for sims + timestep counts)
        self._time_arrays: dict[tuple[str, int], np.ndarray] = {}
        self._load_time_files()

        # Validate that all required TSV files exist
        self._validate_tsv_files()

        # Build flat sample index: [(thickness, sim_id, ts_id), ...]
        self._samples: list[tuple[str, int, int]] = []
        for (thickness, sim_id), time_arr in sorted(self._time_arrays.items()):
            n_ts = len(time_arr)
            for ts_id in range(n_ts - n_steps + 1):
                self._samples.append((thickness, sim_id, ts_id))

        # Cache XYZ mesh positions per thickness (identical across all sims and timesteps)
        self._positions: dict[str, Float[Tensor, "N 3"]] = {}
        for thickness in {t for t, _ in self._time_arrays}:
            self._positions[thickness] = self._load_positions(thickness)

        # Load stats (raw tensors are returned unchanged; normalization is the
        # responsibility of the training loop)
        if stats_file is not None:
            with open(stats_file, "r") as f:
                self._stats: dict | None = json.load(f)
        else:
            self._stats = None
            warnings.warn(
                "TCADMapsDataset instantiated without a stats_file; "
                "get_stats() will raise. Run compute_stats.py first.",
                stacklevel=2,
            )

    def _load_time_files(self) -> None:
        """Populate _time_arrays from i-vs-time/{thickness}/I_time_t_{sim_id}.txt."""
        ivstime_root = self._data_dir / "i-vs-time"
        if not ivstime_root.exists():
            raise FileNotFoundError(f"i-vs-time directory not found: {ivstime_root}")

        for thickness_dir in sorted(ivstime_root.iterdir()):
            thickness = thickness_dir.name
            if thickness not in _THICKNESS_METERS:
                continue
            if (
                self._thickness_filter is not None
                and thickness != self._thickness_filter
            ):
                continue
            for time_file in sorted(thickness_dir.glob("I_time_t_*.txt")):
                m = re.fullmatch(r"I_time_t_(\d+)\.txt", time_file.name)
                if m is None:
                    continue
                sim_id = int(m.group(1))
                rows = np.loadtxt(time_file, delimiter="\t", skiprows=1)
                # rows: (n_ts, 2) — columns are [Time (s), Current (A)]
                self._time_arrays[(thickness, sim_id)] = rows[:, 0].astype(np.float32)

        if not self._time_arrays:
            extra = (
                f" (thickness filter = {self._thickness_filter!r})"
                if self._thickness_filter is not None
                else ""
            )
            raise FileNotFoundError(
                f"No I_time_t_*.txt files found under {ivstime_root}{extra}. "
                "Expected subdirectories named after thicknesses (e.g. '2nm')."
            )

    def _validate_tsv_files(self) -> None:
        """Verify all required TSV files exist for each (thickness, sim_id)."""
        for (thickness, sim_id), time_arr in sorted(self._time_arrays.items()):
            n_ts = len(time_arr)
            for var_name in self.VARIABLES:
                var_dir = self._data_dir / f"{var_name}_data" / thickness
                if not var_dir.exists():
                    raise FileNotFoundError(f"Variable directory not found: {var_dir}")
                for ts_id in range(n_ts):
                    file_path = var_dir / f"{var_name}_{sim_id}_{ts_id}.tsv"
                    if not file_path.exists():
                        raise ValueError(
                            f"Missing {var_name} file for {thickness}/sim_{sim_id} "
                            f"at ts_id={ts_id}: expected {file_path}"
                        )

    def _load_positions(self, thickness: str) -> Float[Tensor, "N 3"]:
        """Read XYZ mesh coordinates from one representative TSV file."""
        var_dir = self._data_dir / "temperature_data" / thickness
        representative = next(var_dir.glob("temperature_*.tsv"))
        raw = np.loadtxt(representative, delimiter="\t", skiprows=1)
        # TSV column order: Z, Y, X, value → reorder to X, Y, Z
        xyz = raw[:, [2, 1, 0]].astype(np.float32)
        return torch.from_numpy(xyz)

    def _read_field(self, path: Path) -> Float[Tensor, " N"]:
        """Read scalar field values (last TSV column) from a file."""
        raw = np.loadtxt(path, delimiter="\t", skiprows=1)
        return torch.from_numpy(raw[:, 3].astype(np.float32))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[TensorDict, dict]:
        thickness, sim_id, ts_id = self._samples[idx]
        n_pts = self._positions[thickness].shape[0]

        variables = torch.empty(
            self._n_steps, len(self.VARIABLES), n_pts, dtype=torch.float32
        )
        source_files: list[str] = []

        # Load n_steps × 2 TSV files (raw, no normalization)
        for step in range(self._n_steps):
            for v_idx, var_name in enumerate(self.VARIABLES):
                path = (
                    self._data_dir
                    / f"{var_name}_data"
                    / thickness
                    / f"{var_name}_{sim_id}_{ts_id + step}.tsv"
                )
                variables[step, v_idx] = self._read_field(path)
                source_files.append(str(path))

        time_arr = self._time_arrays[(thickness, sim_id)]
        time = torch.from_numpy(time_arr[ts_id : ts_id + self._n_steps].copy())

        sample = TensorDict(
            {
                "positions": self._positions[thickness],  # (N, 3)
                "variables": variables,  # (S, V, N)
                "time": time,  # (S,)
                "thickness": torch.tensor(
                    [_THICKNESS_METERS[thickness]], dtype=torch.float32
                ),
            },
            batch_size=[],
        )
        metadata = {
            "sim_id": sim_id,
            "thickness_str": thickness,
            "ts_id": ts_id,
            "source_files": source_files,
        }
        return sample, metadata

    def get_stats(self, var_name: str) -> tuple[float, float]:
        """Return (mean, std) from the loaded stats.json for ``var_name``.

        Available keys: ``coords``, ``temperature``, ``potential``, ``t``.
        """
        if self._stats is None:
            raise RuntimeError(
                "No stats file was provided to TCADMapsDataset; "
                "call compute_stats.py first and pass stats_file=..."
            )
        entry = self._stats[var_name]
        return float(entry["mean"]), float(entry["std"])

    def get_sim_indices(self, thickness_str: str, sim_id: int) -> list[int]:
        """Return ordered flat sample indices for a given (thickness, sim_id).

        Raises ``ValueError`` if the pair does not exist in this dataset.
        """
        indices = [
            i
            for i, (t, s, _) in enumerate(self._samples)
            if t == thickness_str and s == sim_id
        ]
        if not indices:
            available = sorted({(t, s) for t, s, _ in self._samples})
            raise ValueError(
                f"No samples found for (thickness={thickness_str!r}, sim_id={sim_id}). "
                f"Dataset contains {len(available)} (thickness, sim_id) pairs."
            )
        return indices

    def get_sim_ids(self, thickness_str: str) -> list[int]:
        """Return the sorted list of ``sim_id`` values available for ``thickness_str``.

        Raises ``ValueError`` if no simulation exists for that thickness.
        """
        sim_ids = sorted(s for t, s in self._time_arrays if t == thickness_str)
        if not sim_ids:
            available = sorted({t for t, _ in self._time_arrays})
            raise ValueError(
                f"No simulations found for thickness={thickness_str!r}. "
                f"Available thicknesses: {available}"
            )
        return sim_ids


def _collate(
    batch: list[tuple[TensorDict, dict]],
) -> tuple[TensorDict, list[dict]]:
    """Collate a list of (TensorDict, metadata) samples into a batched TensorDict."""
    tds = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    return torch.stack(tds, dim=0), metas


class TCADMapsDataPipe(DataLoader):
    """DDP-aware infinite DataLoader for TCAD map windows with resume support.

    Wraps :class:`TCADMapsDataset` with an :class:`InfiniteSampler` so training
    is sample-count based rather than epoch based. Pass ``start_idx`` to resume
    from an arbitrary position after a checkpoint reload. ``get_stats()``
    proxies to the inner dataset so training code can fetch normalization
    statistics from the same object that produces batches.

    Parameters
    ----------
    data_dir : path
        Path to the maps_only/data/ directory.
    batch_size_per_device : int
        Per-rank batch size. The effective global batch is
        ``batch_size_per_device * world_size``.
    n_steps : int, optional
        Number of consecutive timesteps per sample window. Default ``2``.
    stats_file : path or None, optional
        Path to a stats.json produced by ``compute_stats.py``. Default ``None``
        (``get_stats()`` then raises).
    thickness : str or None, optional
        Forwarded to :class:`TCADMapsDataset`. ``"2nm"``, ``"3nm"``, ``"4nm"``,
        or ``None`` (default) for every thickness.
    shuffle : bool, optional
        Whether the sampler shuffles indices. Default ``True``.
    num_workers : int, optional
        Number of DataLoader worker processes. Default ``4``.
    prefetch_factor : int, optional
        Number of samples loaded in advance per worker (only used when
        ``num_workers > 0``). Default ``4``.
    process_rank : int, optional
        Rank of this process in the DDP group. Default ``0``.
    world_size : int, optional
        Total number of DDP ranks. Default ``1``.
    start_idx : int, optional
        Sample offset to resume from (used after a checkpoint reload).
        Default ``0``.
    seed : int, optional
        Seed for the :class:`InfiniteSampler` shuffle. Default ``0``.

    Examples
    --------
    >>> loader = TCADMapsDataPipe(
    ...     data_dir="data/",
    ...     batch_size_per_device=8,
    ...     n_steps=3,
    ...     stats_file="data/stats.json",
    ...     process_rank=dist.rank,
    ...     world_size=dist.world_size,
    ... )
    >>> for sample, metas in loader:
    ...     # sample is a batched TensorDict; metas is a list[dict]
    ...     ...
    """

    def __init__(
        self,
        data_dir: Path | str,
        batch_size_per_device: int,
        n_steps: int = 2,
        stats_file: Path | str | None = None,
        thickness: str | None = None,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        process_rank: int = 0,
        world_size: int = 1,
        start_idx: int = 0,
        seed: int = 0,
    ) -> None:
        dataset = TCADMapsDataset(
            data_dir=data_dir,
            n_steps=n_steps,
            stats_file=stats_file,
            thickness=thickness,
        )
        sampler = InfiniteSampler(
            dataset=dataset,
            rank=process_rank,
            num_replicas=world_size,
            shuffle=shuffle,
            seed=seed,
            start_idx=start_idx,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size_per_device,
            sampler=sampler,
            collate_fn=_collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=0,
            persistent_workers=False,
        )
        # prefetch_factor is only valid when num_workers > 0
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(**loader_kwargs)

    def get_stats(self, var_name: str) -> tuple[float, float]:
        """Proxy to :meth:`TCADMapsDataset.get_stats`."""
        return self.dataset.get_stats(var_name)
