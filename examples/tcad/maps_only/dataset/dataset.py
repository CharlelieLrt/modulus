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
from typing import Iterator

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

    def compute_sample_weights(self) -> Tensor:
        """Per-sample importance weights from the standardized starting state.

        For each sample ``(thickness, sim_id, ts_id)``, this loads the
        temperature and potential fields at the *starting* timestep
        ``ts_id``, standardizes them with the dataset stats, and combines
        the two extremes:

        .. math::

            w_i = \\max_n \\left| \\frac{T_n(t_{\\rm s}) - T_\\mathrm{mean}}{T_\\mathrm{std}} \\right|
                + \\max_n \\left| \\frac{V_n(t_{\\rm s}) - V_\\mathrm{mean}}{V_\\mathrm{std}} \\right|.

        Samples whose starting state is unusual (hot spot, high field) get
        higher weight; quiescent early states (close to dataset mean) get
        small weight. The result is unit-free and balanced across T and V.

        Use the returned tensor as input to :func:`apply_weight_schedule`,
        then pass the schedule output to :class:`InfiniteWeightedSampler`.

        Notes
        -----
        Reads every TSV file in the dataset once (cached per
        ``(thickness, sim_id)``). For a few hundred sims this is on the
        order of seconds and only happens at the start of training.

        Returns
        -------
        Tensor
            1D tensor of length ``len(self)`` with non-negative weights.
        """
        if self._stats is None:
            raise RuntimeError(
                "stats_file is required to compute sample weights "
                "(used to standardize T and V before combining)."
            )
        T_mean, T_std = self.get_stats("temperature")
        V_mean, V_std = self.get_stats("potential")

        # Per (thickness, sim_id) -> (T_zmax_per_ts, V_zmax_per_ts), each
        # of shape (n_ts,). Computed once per sim by loading every
        # timestep once.
        norms_per_sim: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}
        for (thickness, sim_id), time_arr in self._time_arrays.items():
            n_ts = len(time_arr)
            T_steps: list[Tensor] = []
            V_steps: list[Tensor] = []
            for ts_id in range(n_ts):
                T_path = (
                    self._data_dir
                    / "temperature_data"
                    / thickness
                    / f"temperature_{sim_id}_{ts_id}.tsv"
                )
                V_path = (
                    self._data_dir
                    / "potential_data"
                    / thickness
                    / f"potential_{sim_id}_{ts_id}.tsv"
                )
                T_steps.append(self._read_field(T_path))
                V_steps.append(self._read_field(V_path))
            T_arr = torch.stack(T_steps, dim=0)  # (n_ts, N)
            V_arr = torch.stack(V_steps, dim=0)  # (n_ts, N)
            T_zmax = ((T_arr - T_mean) / T_std).abs().amax(dim=-1)  # (n_ts,)
            V_zmax = ((V_arr - V_mean) / V_std).abs().amax(dim=-1)  # (n_ts,)
            norms_per_sim[(thickness, sim_id)] = (T_zmax, V_zmax)

        weights = torch.empty(len(self), dtype=torch.float32)
        for i, (thickness, sim_id, ts_id) in enumerate(self._samples):
            T_zmax, V_zmax = norms_per_sim[(thickness, sim_id)]
            weights[i] = float(T_zmax[ts_id] + V_zmax[ts_id])
        return weights

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


def apply_weight_schedule(
    weights: Tensor,
    weight_percentile: float,
    sampling_probability: float,
) -> Tensor:
    """Two-tier discontinuous rank-based weight schedule.

    Splits the samples into a *top* tier (target size ``round(X * N)``
    where ``X = weight_percentile``) and a *bottom* tier of the rest,
    then assigns each tier a uniform per-sample probability so that:

    - top tier collectively integrates to ``Y = sampling_probability``;
    - bottom tier collectively integrates to ``1 - Y``.

    Tied input weights receive equal probability: per-position
    probabilities are computed first from the argsort positions, and
    then averaged within each tie group. As a consequence, when a tie
    group straddles the percentile boundary, all of its members get the
    same blended probability. This preserves the invariant
    "identical weights => identical probability" without breaking the
    ``X == Y`` -> uniform identity (which would fail if all members of
    a large bottom-tied group were force-promoted into a single tier).

    Setting ``Y == X`` produces a uniform distribution. ``Y > X``
    over-samples the top tier; ``Y < X`` would under-sample it (we
    require ``Y >= X``).

    Parameters
    ----------
    weights : Tensor
        1D non-negative tensor (e.g. from
        :meth:`TCADMapsDataset.compute_sample_weights`). Used only for
        ranking — magnitudes do not affect the output beyond rank order.
    weight_percentile : float
        Fraction of samples in the top tier, in (0, 1).
    sampling_probability : float
        Total probability mass assigned to the top tier, in (0, 1).
        Must be ``>= weight_percentile``.

    Returns
    -------
    Tensor
        1D float tensor of length ``len(weights)`` summing to 1.
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {tuple(weights.shape)}")
    if not 0.0 < weight_percentile < 1.0:
        raise ValueError(
            f"weight_percentile must be in (0, 1), got {weight_percentile}"
        )
    if not 0.0 < sampling_probability < 1.0:
        raise ValueError(
            f"sampling_probability must be in (0, 1), got {sampling_probability}"
        )
    if sampling_probability < weight_percentile:
        raise ValueError(
            "sampling_probability must be >= weight_percentile so the top "
            f"tier is at least as dense as the bottom tier; got "
            f"sampling_probability={sampling_probability}, "
            f"weight_percentile={weight_percentile}."
        )
    n = weights.numel()
    if n < 2:
        raise ValueError("apply_weight_schedule requires at least 2 samples.")

    X = weight_percentile
    Y = sampling_probability

    # Stable ascending sort. sort_position[i] = sort rank of sample i.
    sort_idx = torch.argsort(weights, stable=True)
    sort_position = torch.empty(n, dtype=torch.long, device=weights.device)
    sort_position[sort_idx] = torch.arange(n, device=weights.device)

    # Target tier sizes (positions >= threshold_pos are top tier before
    # tie-aware averaging).
    n_top = max(1, min(int(round(X * n)), n - 1))
    n_bot = n - n_top
    threshold_pos = n_bot
    p_top = Y / n_top
    p_bot = (1.0 - Y) / n_bot

    # Raw per-position probability.
    p_raw = torch.where(
        sort_position >= threshold_pos,
        torch.full_like(weights, p_top, dtype=torch.float64),
        torch.full_like(weights, p_bot, dtype=torch.float64),
    )

    # Average within each tie group so identical weights get identical
    # probability. A tie group is a maximal run of equal weights in the
    # sorted order; its averaged probability is the mass-weighted blend
    # of p_top / p_bot proportional to how much of the group lies above
    # / below the threshold position.
    sorted_w = weights[sort_idx]
    sorted_p = p_raw[sort_idx]
    is_new = torch.ones(n, dtype=torch.long, device=weights.device)
    is_new[1:] = (sorted_w[1:] != sorted_w[:-1]).long()
    group_ids = is_new.cumsum(0) - 1
    n_groups = int(group_ids[-1].item()) + 1
    sums = torch.zeros(n_groups, dtype=torch.float64, device=weights.device)
    counts = torch.zeros(n_groups, dtype=torch.float64, device=weights.device)
    sums.scatter_add_(0, group_ids, sorted_p)
    counts.scatter_add_(
        0, group_ids, torch.ones(n, dtype=torch.float64, device=weights.device)
    )
    group_avg_p = sums / counts
    sorted_p_avg = group_avg_p[group_ids]

    p = torch.empty(n, dtype=torch.float64, device=weights.device)
    p[sort_idx] = sorted_p_avg
    return p.to(weights.dtype)


class InfiniteWeightedSampler(InfiniteSampler):
    """Like :class:`InfiniteSampler` but draws indices via per-sample weights.

    Each draw is independent and made with replacement using ``weights`` as
    relative probabilities. Weights are normalized to a probability vector
    internally. A scalar ``weights`` value broadcasts to a uniform
    distribution over all dataset indices, which is statistically equivalent
    to non-weighted uniform sampling -- useful for non-regression tests of
    training recipes that opt into the weighted code path.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to sample from.
    weights : Tensor or np.ndarray or scalar
        Per-sample relative weights, length ``len(dataset)``. A scalar is
        broadcast to a uniform vector. Must be 1-D, finite, non-negative,
        and sum to a positive value.
    rank : int, default 0
        Rank of the current process within ``num_replicas`` ranks.
    num_replicas : int, default 1
        Total number of distributed ranks.
    shuffle : bool, default True
        Accepted for API parity with :class:`InfiniteSampler` but ignored:
        weighted sampling is always with replacement.
    seed : int, default 0
        Seed for ``np.random.RandomState``.
    start_idx : int, default 0
        Sample offset used to fast-forward the RNG when resuming a
        checkpoint. The RNG is advanced by drawing (and discarding)
        ``start_idx`` samples.

    Notes
    -----
    The ``window_size`` argument from :class:`InfiniteSampler` is unused --
    weighted draws are independent per index. Resume support is exact: every
    rank advances its RNG by ``start_idx`` draws before yielding, so the
    rank-disjoint sample sequence is identical to a fresh run.
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Tensor | np.ndarray | float,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        start_idx: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            rank=rank,
            num_replicas=num_replicas,
            shuffle=shuffle,
            seed=seed,
            start_idx=start_idx,
        )
        n = len(dataset)
        if isinstance(weights, (int, float)):
            w = np.full(n, float(weights), dtype=np.float64)
        else:
            if isinstance(weights, Tensor):
                w = weights.detach().cpu().numpy().astype(np.float64)
            else:
                w = np.asarray(weights, dtype=np.float64)
            if w.ndim != 1:
                raise ValueError(f"weights must be 1D, got shape {tuple(w.shape)}")
            if len(w) != n:
                raise ValueError(f"weights length {len(w)} != dataset length {n}")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite (no NaN/Inf)")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        self._probs: np.ndarray = w / total

    def __iter__(self) -> Iterator[int]:
        rnd = np.random.RandomState(self.seed)
        n = len(self.dataset)
        # Fast-forward the RNG to the correct state for resume.
        if self.start_idx > 0:
            _ = rnd.choice(n, size=self.start_idx, p=self._probs)
        idx = self.start_idx
        while True:
            sample_idx = int(rnd.choice(n, p=self._probs))
            if idx % self.num_replicas == self.rank:
                yield sample_idx
            idx += 1


class TCADMapsDataPipe(DataLoader):
    """DDP-aware infinite DataLoader for TCAD map windows with resume support.

    Wraps :class:`TCADMapsDataset` with an :class:`InfiniteSampler` so training
    is sample-count based rather than epoch based. Pass ``start_idx`` to resume
    from an arbitrary position after a checkpoint reload. ``get_stats()``
    proxies to the inner dataset so training code can fetch normalization
    statistics from the same object that produces batches.

    Pass a ``weights`` tensor to switch to :class:`InfiniteWeightedSampler`
    -- the data pipe does not interpret the values and does not rescale
    them, it just hands them to the sampler. Build the tensor with
    :meth:`TCADMapsDataset.compute_sample_weights` (rescales deltas by
    per-variable std) and :func:`apply_weight_schedule` (rank-based
    reshaping). With ``weights=None`` (default) the loader falls back to
    the uniform :class:`InfiniteSampler`, identical to its pre-existing
    behavior.

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
    weights : Tensor or None, optional
        Per-sample weights for :class:`InfiniteWeightedSampler`. Length
        must equal ``len(self.dataset)``. ``None`` (default) means uniform
        :class:`InfiniteSampler`.

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
        weights: Tensor | None = None,
    ) -> None:
        dataset = TCADMapsDataset(
            data_dir=data_dir,
            n_steps=n_steps,
            stats_file=stats_file,
            thickness=thickness,
        )
        if weights is not None:
            sampler: InfiniteSampler = InfiniteWeightedSampler(
                dataset=dataset,
                weights=weights,
                rank=process_rank,
                num_replicas=world_size,
                shuffle=shuffle,
                seed=seed,
                start_idx=start_idx,
            )
        else:
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
