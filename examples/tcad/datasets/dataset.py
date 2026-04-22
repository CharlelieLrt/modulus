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


import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from physicsnemo.datapipes.meta import DatapipeMetaData

# Global constants for node attributes
NODE_TYPE = {
    "defect": 0,
    "boundary-dirichlet": 2,
}

NODE_STATUS = {
    "no_status": -1,
    "active": 0,
    "new": 1,
}

MATERIAL_ID = {
    "material_0": 0,
    "material_1": 1,
}

EDGE_TYPE = {
    "intra-defects": 0,
    "intra-boundary": 1,
    "inter-defects-boundary": 2,
}


def compute_edge_index(
    graph: PyGData,
    radius_defects: float,
    radius_boundary: float,
) -> None:
    """Computes graph connectivity based on pairwise distance.

    Parameters
    ----------
    graph : PyGData
        Input graph with pos and node_type attributes
    radius_defects : float
        Connectivity radius for defects nodes
    radius_boundary : float
        Connectivity radius for boundary nodes

    Returns
    -------
    None
        The graph is updated in place by modifying the `edge_index` attribute.
    """
    if graph.pos is None or graph.node_type is None:
        raise ValueError("Graph must have pos and node_type attributes")

    pos = graph.pos
    node_type = graph.node_type

    # Identify defect and boundary nodes
    defect_indices = torch.where(node_type == NODE_TYPE["defect"])[0]
    boundary_indices = torch.where(node_type >= NODE_TYPE["boundary-dirichlet"])[0]

    # Intra-connectivity for defect nodes
    # Only compute distances between defect nodes
    if len(defect_indices) > 0:
        pos_defects = pos[defect_indices]
        distances_defects = torch.cdist(pos_defects, pos_defects, p=2)
        mask_defect = distances_defects < radius_defects
        edges_local = torch.nonzero(mask_defect).t()
        # Map back to global node indices
        edges_defect_defect = defect_indices[edges_local]
    else:
        edges_defect_defect = torch.empty((2, 0), dtype=torch.long, device=pos.device)

    # Intra-connectivity for boundary nodes
    # Only compute distances between boundary nodes
    if len(boundary_indices) > 0:
        pos_boundary = pos[boundary_indices]
        distances_boundary = torch.cdist(pos_boundary, pos_boundary, p=2)
        mask_boundary = distances_boundary < radius_boundary
        edges_local = torch.nonzero(mask_boundary).t()
        # Map back to global node indices
        edges_boundary_boundary = boundary_indices[edges_local]
    else:
        edges_boundary_boundary = torch.empty(
            (2, 0), dtype=torch.long, device=pos.device
        )

    # Inter-connectivity between defects and boundary nodes
    # Only compute distances between defect and boundary nodes
    if len(defect_indices) > 0 and len(boundary_indices) > 0:
        pos_defects = pos[defect_indices]
        pos_boundary = pos[boundary_indices]
        distances_inter = torch.cdist(pos_defects, pos_boundary, p=2)
        mask_inter = distances_inter < radius_defects
        edges_local = torch.nonzero(mask_inter).t()
        # Map back to global node indices
        edges_inter_src = defect_indices[edges_local[0]]
        edges_inter_dst = boundary_indices[edges_local[1]]
        edges_inter = torch.stack([edges_inter_src, edges_inter_dst], dim=0)
    else:
        edges_inter = torch.empty((2, 0), dtype=torch.long, device=pos.device)

    # Merge all edge indices
    edge_index = torch.cat(
        [edges_defect_defect, edges_boundary_boundary, edges_inter], dim=1
    )

    graph.edge_index = edge_index
    return


def compute_edge_attr(
    graph: PyGData,
    radius_defects: float,
) -> None:
    """Computes edge attributes (displacement and distance).

    Parameters
    ----------
    graph : PyGData
        Input graph
    radius_defects : float
        Radius for distance calculation for defects nodes

    Returns
    -------
    None
        The graph is updated in place by modifying the `edge_attr` and
        `edge_type` attributes.
    """
    if graph.edge_index is None or graph.pos is None:
        raise ValueError("Graph must have edge_index and pos attributes")

    edge_index = graph.edge_index
    displacement = graph.pos[edge_index[1]] - graph.pos[edge_index[0]]
    distance = torch.pairwise_distance(
        graph.pos[edge_index[0]],
        graph.pos[edge_index[1]],
        keepdim=True,
    )

    # Apply exponential transform to distance
    distance = torch.exp(-(distance**2) / radius_defects**2)

    # Determine edge type for each edge
    node_type_src = graph.node_type[edge_index[0]]
    node_type_dst = graph.node_type[edge_index[1]]
    is_defect_src = node_type_src == NODE_TYPE["defect"]
    is_defect_dst = node_type_dst == NODE_TYPE["defect"]
    is_boundary_src = node_type_src >= NODE_TYPE["boundary-dirichlet"]
    is_boundary_dst = node_type_dst >= NODE_TYPE["boundary-dirichlet"]

    edge_type = torch.zeros(
        edge_index.shape[1], dtype=torch.long, device=graph.pos.device
    )
    edge_type[is_defect_src & is_defect_dst] = EDGE_TYPE["intra-defects"]
    edge_type[is_boundary_src & is_boundary_dst] = EDGE_TYPE["intra-boundary"]
    edge_type[(is_defect_src & is_boundary_dst) | (is_boundary_src & is_defect_dst)] = (
        EDGE_TYPE["inter-defects-boundary"]
    )

    # Store edge attributes
    graph.edge_attr = torch.cat((displacement, distance), dim=-1)
    graph.edge_type = edge_type

    return


def graph_add_z_boundary(
    graph: PyGData,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    material_id_lower: int = 1,
    material_id_upper: int = 1,
    node_type_lower: int = 3,
    node_type_upper: int = 3,
    res_xy: int = 10,
    res_z: int = 2,
    dz: float = 0.1,
) -> None:
    """Adds boundary nodes to the graph.

    Parameters
    ----------
    graph : PyGData
        Input graph. Should only contain the defect nodes and no connectivity.
    x_range : Tuple[float, float]
        Range of the x coordinates for the boundary nodes. Defines the length
        of the domain in the x direction.
    y_range : Tuple[float, float]
        Range of the y coordinates for the boundary nodes.
    z_range : Tuple[float, float], optional
        Range of z coordinates (lower and upper boundary locations)
    material_id_lower : int
        Material id for the lower boundary nodes.
    material_id_upper : int
        Material id for the upper boundary nodes.
    node_type_lower : int
        Node type for the lower boundary nodes; determines the type of the
        boundary condition (Dirichlet, Neumann, etc...).
    node_type_upper : int
        Node type for the upper boundary nodes; determines the type of the
        boundary condition (Dirichlet, Neumann, etc...).
    res_xy : int
        Resolution of the boundary nodes in the transversal direction (x, y).
    res_z : int
        Number of layers of boundary nodes to add in the z direction.
    dz : float
        Total thickness of the layers of boundary nodes in the z direction.

    Returns
    -------
    None
        The graph is updated in place.
    """
    # Get device and dtype from existing graph attributes
    if graph.pos is None:
        raise ValueError("Graph must have pos attribute")
    device = graph.pos.device
    dtype = graph.pos.dtype

    # Create 2D grid for x-y plane
    x_coords = torch.linspace(x_range[0], x_range[1], res_xy, device=device)
    y_coords = torch.linspace(y_range[0], y_range[1], res_xy, device=device)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    num_nodes_per_layer = len(x_grid)

    # Create z coordinates for lower boundary layers
    z_lower = z_range[0]
    z_upper = z_range[1]
    z_coords_lower = torch.linspace(
        z_lower - dz, z_lower, res_z, device=device, dtype=dtype
    )

    pos_lower = []
    for z_val in z_coords_lower:
        z_layer = torch.full((num_nodes_per_layer,), z_val, device=device)
        layer_pos = torch.stack([x_grid, y_grid, z_layer], dim=-1)
        pos_lower.append(layer_pos)
    pos_lower = torch.cat(pos_lower, dim=0).to(dtype)

    # Create upper boundary by duplicating and translating
    pos_upper = pos_lower.clone()
    pos_upper[:, 2] += z_upper - z_lower + dz  # Translate in z direction

    # Total boundary nodes
    num_boundary = pos_lower.shape[0]
    num_boundary_total = 2 * num_boundary

    # Create attributes for lower boundary nodes (all features set to zero except voltage/temp)
    defect_ion_ene_lower = torch.zeros(num_boundary, device=device, dtype=dtype)
    defect_invsoftplus_gen_delay_lower = torch.zeros(
        num_boundary, device=device, dtype=dtype
    )
    voltage_lower = torch.ones(num_boundary, device=device, dtype=dtype)
    temperature_lower = torch.ones(num_boundary, device=device, dtype=dtype)

    defect_id_lower = torch.full((num_boundary,), -1, device=device, dtype=torch.long)
    material_id_lower_tensor = torch.full(
        (num_boundary,), material_id_lower, device=device, dtype=torch.long
    )
    node_type_lower_tensor = torch.full(
        (num_boundary,), node_type_lower, device=device, dtype=torch.long
    )
    node_status_lower = torch.full(
        (num_boundary,), NODE_STATUS["no_status"], device=device, dtype=torch.long
    )

    # Create attributes for upper boundary nodes by cloning
    defect_ion_ene_upper = defect_ion_ene_lower.clone()
    defect_invsoftplus_gen_delay_upper = defect_invsoftplus_gen_delay_lower.clone()
    voltage_upper = voltage_lower.clone()
    temperature_upper = temperature_lower.clone()

    defect_id_upper = defect_id_lower.clone()
    node_status_upper = node_status_lower.clone()

    # Material ID and node type may differ for upper boundary
    material_id_upper_tensor = torch.full(
        (num_boundary,), material_id_upper, device=device, dtype=torch.long
    )
    node_type_upper_tensor = torch.full(
        (num_boundary,), node_type_upper, device=device, dtype=torch.long
    )

    # Concatenate boundary positions and features
    pos_boundary = torch.cat([pos_lower, pos_upper], dim=0)
    defect_ion_ene_boundary = torch.cat(
        [defect_ion_ene_lower, defect_ion_ene_upper], dim=0
    )
    defect_invsoftplus_gen_delay_boundary = torch.cat(
        [defect_invsoftplus_gen_delay_lower, defect_invsoftplus_gen_delay_upper], dim=0
    )
    voltage_boundary = torch.cat([voltage_lower, voltage_upper], dim=0)
    temperature_boundary = torch.cat([temperature_lower, temperature_upper], dim=0)

    defect_id_boundary = torch.cat([defect_id_lower, defect_id_upper], dim=0)
    material_id_boundary = torch.cat(
        [material_id_lower_tensor, material_id_upper_tensor], dim=0
    )
    node_type_boundary = torch.cat(
        [node_type_lower_tensor, node_type_upper_tensor], dim=0
    )
    node_status_boundary = torch.cat([node_status_lower, node_status_upper], dim=0)

    # Concatenate all nodes (defects + boundary)
    graph.pos = torch.cat([graph.pos, pos_boundary], dim=0)
    graph.defect_ion_ene = torch.cat(
        [graph.defect_ion_ene, defect_ion_ene_boundary], dim=0
    )
    graph.defect_invsoftplus_gen_delay = torch.cat(
        [graph.defect_invsoftplus_gen_delay, defect_invsoftplus_gen_delay_boundary],
        dim=0,
    )
    graph.voltage = torch.cat([graph.voltage, voltage_boundary], dim=0)
    graph.temperature = torch.cat([graph.temperature, temperature_boundary], dim=0)
    graph.defect_id = torch.cat([graph.defect_id, defect_id_boundary], dim=0)
    graph.material_id = torch.cat([graph.material_id, material_id_boundary], dim=0)
    graph.node_type = torch.cat([graph.node_type, node_type_boundary], dim=0)
    graph.node_status = torch.cat([graph.node_status, node_status_boundary], dim=0)

    # Update total number of nodes
    graph.num_nodes = graph.num_defects + num_boundary_total

    return


def collate_snapshots(batch):
    """Collate a batch of snapshot sequences into a single batch.

    Parameters
    ----------
    batch : list
        List of snapshot sequences, where each sequence is a list of
        num_steps PyGData objects

    Returns
    -------
    list[PyGData]
        List of num_steps batched PyGData objects
    """

    # batch is a list of lists: [[graph_0_t0, graph_0_t1, ...], [graph_1_t0, ...], ...]
    # We want to collate timestep-wise: [Batch(graph_0_t0, graph_1_t0, ...), ...]

    if len(batch) == 0:
        return []

    # Use zip to transpose: [[g0_t0, g0_t1], [g1_t0, g1_t1]] -> [(g0_t0, g1_t0), (g0_t1, g1_t1)]
    batched_snapshots = [
        PyGBatch.from_data_list(graphs_at_t) for graphs_at_t in zip(*batch)
    ]

    return batched_snapshots


@dataclass
class MetaData(DatapipeMetaData):
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class TCADDataset(PyGDataset):
    """
    In-memory dataset for TCAD data.

    Parameters
    ----------
    data_dir : str | Path
        Path to the dataset directory containing sim_<sim_id> subdirectories
    num_steps : int
        Number of consecutive timesteps to load for each sample
    radius_defects : float, optional
        Connectivity radius for defect nodes, default=0.015
    compute_connectivity : bool, optional
        Whether to compute graph connectivity, default=True
    add_boundary : bool, optional
        Whether to add boundary nodes, default=True
    normalize : bool, optional
        Whether to normalize node features and coordinates, default=False
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_steps: int,
        radius_defects: float,
        compute_connectivity: bool = True,
        add_boundary: bool = True,
        normalize: bool = False,
    ) -> None:
        # Parameters validation and input pre-processing
        if isinstance(data_dir, str):
            data_dir: Path = Path(data_dir)
        self.data_dir: Path = data_dir.expanduser()

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        self.num_steps = num_steps
        self.radius_defects = radius_defects
        self._compute_connectivity = compute_connectivity
        self._add_boundary = add_boundary
        self._normalize = normalize

        # Define node feature variable names (stored as separate attributes)
        self.variables = [
            "ion_ene",
            "invsoftplus_gen_delay",
            "voltage",
            "temperature",
        ]

        # Define boundary configuration
        self._boundary_res_xy = 10
        self._boundary_x_range = (-1.5, 1.5)
        self._boundary_y_range = (-1.5, 1.5)
        # Compute grid spacing
        dx = (self._boundary_x_range[1] - self._boundary_x_range[0]) / (
            self._boundary_res_xy - 1
        )
        dy = (self._boundary_y_range[1] - self._boundary_y_range[0]) / (
            self._boundary_res_xy - 1
        )
        self._boundary_dxy = min(dx, dy)
        # Boundary radius based on grid spacing
        self.radius_boundary = 1.2 * self._boundary_dxy

        # Load dataset statistics
        stats_file = self.data_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.stats = json.load(f)
            # Override stats for invsoftplus_time to use same as invsoftplus_gen_delay
            if "invsoftplus_gen_delay" in self.stats:
                self.stats["invsoftplus_time"] = self.stats["invsoftplus_gen_delay"]
        else:
            if normalize:
                raise ValueError(
                    f"Normalization requested but stats file not found: {stats_file}"
                )
            self.stats = None
            warnings.warn(
                f"Stats file {stats_file} not found. "
                f"Normalization will not be available."
            )

        # Find all sim_<sim_id> subdirectories
        sim_dirs = sorted(
            [
                d
                for d in self.data_dir.iterdir()
                if d.is_dir() and d.name.startswith("sim_")
            ],
            key=lambda x: int(x.name.split("_")[1]),
        )

        if not sim_dirs:
            raise ValueError(f"No sim_* subdirectories found in {self.data_dir}")

        # Load all data and metadata
        # {(sim_id, timestamp_id): tensors_dict}
        self.data: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        # {sim_id: info_dict}
        self.metadata: Dict[int, Dict[str, Any]] = {}

        for sim_dir in sim_dirs:
            sim_id = int(sim_dir.name.split("_")[1])

            # Load info.json metadata
            info_file = sim_dir / "info.json"
            if info_file.exists():
                with open(info_file, "r") as f:
                    self.metadata[sim_id] = json.load(f)
            else:
                raise ValueError(f"No info.json found for {sim_dir.name}")

            # Load all sample files
            sample_files = sorted(
                sim_dir.glob("sample_*.npz"), key=lambda x: int(x.stem.split("_")[1])
            )

            for sample_file in sample_files:
                timestamp_id = int(sample_file.stem.split("_")[1])

                # Load NPZ file and convert to torch tensors
                with np.load(sample_file) as data_npz:
                    tensors: Dict[str, torch.Tensor] = {}
                    for key in data_npz.keys():
                        tensors[key] = torch.from_numpy(data_npz[key]).to("cpu")
                    self.data[(sim_id, timestamp_id)] = tensors

        # Create index mapping
        # Each sample consists of N_steps consecutive timesteps
        # We can only create samples where we have N_steps available
        self.sample_keys: Dict[int, Tuple[int, int]] = {}
        for idx, sim_id in enumerate(sorted(self.metadata.keys())):
            # Get all timestamps for this simulation
            sim_timestamps = sorted([ts for (s, ts) in self.data.keys() if s == sim_id])

            # Create samples: each starts at a different timestamp
            # Last valid starting point is (max_ts - N_steps + 1)
            num_valid_samples = len(sim_timestamps) - self.num_steps + 1
            if num_valid_samples <= 0:
                raise ValueError(
                    f"Simulation {sim_id} has {len(sim_timestamps)} "
                    f"timesteps, need at least {self.num_steps}."
                )

            for start_idx in range(num_valid_samples):
                start_ts = sim_timestamps[start_idx]
                self.sample_keys[idx] = (sim_id, start_ts)

        # Compute the maximum number of new defects at each timestep over all
        # simulations.
        self.max_new_defects = max(
            [
                self.metadata[sim_id]["max_new_defects"]
                for sim_id in self.metadata.keys()
            ]
        )

        self.num_samples = len(self.sample_keys)

    def __len__(self) -> int:
        return self.num_samples

    def normalize(self, variable: str, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize a variable using statistics from stats.json.

        Parameters
        ----------
        variable : str
            Variable name (must exist in self.stats)
        data : torch.Tensor
            Data to normalize

        Returns
        -------
        torch.Tensor
            Normalized data
        """
        if self.stats is None:
            raise ValueError("No statistics available for normalization")

        if variable not in self.stats:
            raise ValueError(f"Variable '{variable}' not found in statistics")

        mean = self.stats[variable]["mean"]
        std = self.stats[variable]["std"]
        return (data - mean) / std

    def denormalize(self, variable: str, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a variable using statistics from stats.json.

        Parameters
        ----------
        variable : str
            Variable name (must exist in self.stats)
        data : torch.Tensor
            Normalized data

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        if self.stats is None:
            raise ValueError("No statistics available for denormalization")

        if variable not in self.stats:
            raise ValueError(f"Variable '{variable}' not found in statistics")

        mean = self.stats[variable]["mean"]
        std = self.stats[variable]["std"]
        return data * std + mean

    def graph_update(self, graph: PyGData) -> None:
        """Updates graph structure by reconstructing edges based on positions.

        Parameters
        ----------
        graph : PyGData
            Input graph

        Returns
        -------
        None
            The graph is updated in place by modifying the `edge_index` and
            `edge_attr` attributes.
        """
        if graph.pos is None or graph.node_type is None:
            raise ValueError("Graph must have pos and node_type attributes")

        compute_edge_index(graph, self.radius_defects, self.radius_boundary)
        compute_edge_attr(graph, self.radius_defects)
        return

    def __getitem__(self, idx: int) -> list[PyGData]:
        """
        Load a sample consisting of num_steps consecutive timesteps.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        list[PyGData]
            List of num_steps graph objects
        """
        # Get the (sim_id, start_timestamp) for this sample
        sim_id, start_ts = self.sample_keys[idx]

        snapshots: list[PyGData] = []

        # Loop over num_steps consecutive timesteps
        for step_idx in range(self.num_steps):
            timestamp_id = start_ts + step_idx

            # Get data for this timestep
            data_dict = self.data[(sim_id, timestamp_id)]

            # Extract node positions (x, y, z coordinates)
            x_coord = data_dict["x"]
            y_coord = data_dict["y"]
            z_coord = data_dict["z"]

            # Normalize coordinates if requested
            if self._normalize:
                x_coord = self.normalize("x", x_coord)
                y_coord = self.normalize("y", y_coord)
                z_coord = self.normalize("z", z_coord)

            pos_defects = torch.stack(
                [x_coord, y_coord, z_coord], dim=-1
            ).float()  # Shape: (num_defects, 3)

            num_defects = pos_defects.shape[0]

            # Extract and optionally normalize node features
            ion_ene = data_dict["ion_ene"]
            invsoftplus_gen_delay = data_dict["invsoftplus_gen_delay"]
            invsoftplus_time = data_dict["invsoftplus_time"]

            if self._normalize:
                ion_ene = self.normalize("ion_ene", ion_ene)
                invsoftplus_gen_delay = self.normalize(
                    "invsoftplus_gen_delay", invsoftplus_gen_delay
                )
                invsoftplus_time = self.normalize("invsoftplus_time", invsoftplus_time)

            # Store node features as separate attributes with "defect_" prefix
            graph = PyGData(pos=pos_defects)

            # Set individual node feature attributes for defect nodes
            graph.defect_ion_ene = ion_ene.float()
            graph.defect_invsoftplus_gen_delay = invsoftplus_gen_delay.float()
            graph.voltage = torch.zeros(num_defects, dtype=torch.float32)
            graph.temperature = torch.zeros(num_defects, dtype=torch.float32)

            # Set graph-level attribute (single value per graph)
            graph.invsoftplus_time = invsoftplus_time.float()

            # Node-level labels for defect nodes
            graph.defect_id = data_dict["id"].long()
            graph.material_id = torch.full(
                (num_defects,), MATERIAL_ID["material_0"], dtype=torch.long
            )
            graph.node_type = torch.full(
                (num_defects,), NODE_TYPE["defect"], dtype=torch.long
            )
            graph.node_status = torch.full(
                (num_defects,), NODE_STATUS["active"], dtype=torch.long
            )

            # Add graph-level metadata as attributes
            graph.sim_id = torch.tensor([sim_id], dtype=torch.long)
            graph.timestep = torch.tensor([step_idx], dtype=torch.long)
            graph.time = data_dict["time"].float()

            # Normalize thickness if requested
            thickness = data_dict["thickness"].float()
            if self._normalize:
                thickness = self.normalize("z", thickness)
            graph.thickness = thickness
            graph.num_defects = data_dict["num_defects"]

            # Add boundary nodes if requested
            if self._add_boundary:
                z_range_norm = (0.0, thickness.item())

                graph_add_z_boundary(
                    graph,
                    x_range=self._boundary_x_range,
                    y_range=self._boundary_y_range,
                    z_range=z_range_norm,
                    res_xy=self._boundary_res_xy,
                )

            # Build graph connectivity and edge attributes if requested
            if self._compute_connectivity:
                type(self).graph_update(
                    graph, self.radius_defects, self.radius_boundary
                )

            snapshots.append(graph)

        return snapshots


class TCADDatapipe(PyGDataLoader):
    """
    Datapipe for TCAD dataset.

    Parameters
    ----------
    data_dir : str | Path
        Path to the dataset directory containing sim_* subdirectories
    num_steps : int
        Number of consecutive timesteps per sample
    batch_size_per_device : int
        Batch size per device
    radius_defects : float
        Connectivity radius for defect nodes
    compute_connectivity : bool, optional
        Whether to compute graph connectivity, default=True
    add_boundary : bool, optional
        Whether to add boundary nodes, default=True
    normalize : bool, optional
        Whether to normalize node features and coordinates, default=False
    seed : int, optional, default=0
        Random seed for shuffling
    shuffle : bool, optional, default=True
        Whether to shuffle the dataset
    drop_last : bool, optional, default=False
        Whether to drop the last incomplete batch
    num_workers : int, optional, default=1
        Number of workers for data loading
    pin_memory : bool, optional, default=True
        Whether to use pinned memory
    device : str | torch.device, optional, default="cuda"
        Device to move data to
    process_rank : int, optional, default=0
        Rank of the process (for distributed training)
    world_size : int, optional, default=1
        Total number of processes (for distributed training)
    prefetch_factor : int, optional, default=2
        Number of batches to prefetch
    use_sharding : bool, optional, default=None
        Whether to use sharding. If None, sharding is used if world_size > 1
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_steps: int,
        batch_size_per_device: int,
        radius_defects: float,
        compute_connectivity: bool = True,
        add_boundary: bool = True,
        normalize: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        prefetch_factor: Optional[int] = 2,
        use_sharding: Optional[bool] = None,
    ) -> None:
        if isinstance(device, str):
            device: torch.device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device: torch.device = torch.device("cuda:0")
        self.device = device

        # Create dataset
        dataset = TCADDataset(
            data_dir=data_dir,
            num_steps=num_steps,
            radius_defects=radius_defects,
            compute_connectivity=compute_connectivity,
            add_boundary=add_boundary,
            normalize=normalize,
        )

        # Determine whether to use sharding
        should_shard: bool = use_sharding if use_sharding is not None else True

        # Create sampler for distributed training
        if should_shard and world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=process_rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
            shuffle = None
            generator = None
        else:
            sampler = None
            generator = torch.Generator("cpu").manual_seed(seed)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size_per_device,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_snapshots,
            generator=generator,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for the datapipe. Used for shuffling in distributed
        training.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def graph_update(self, graph: PyGData) -> None:
        """
        Updates graph structure by reconstructing edges based on positions.

        Parameters
        ----------
        graph : PyGData
            Input graph

        Returns
        -------
        None
            The graph is updated in place by modifying the `edge_index` and
            `edge_attr` attributes.
        """
        self.dataset.graph_update(graph)
        return

    def normalize(self, variable: str, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize a variable using statistics from stats.json.

        Parameters
        ----------
        variable : str
            Variable name
        data : torch.Tensor
            Data to normalize

        Returns
        -------
        torch.Tensor
            Normalized data
        """
        return self.dataset.normalize(variable, data)

    def denormalize(self, variable: str, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a variable using statistics from stats.json.

        Parameters
        ----------
        variable : str
            Variable name
        data : torch.Tensor
            Normalized data

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        return self.dataset.denormalize(variable, data)

    def get_stats(
        self,
        metric: str,
    ) -> Dict[str, float]:
        """Return statistics for each variable.

        Parameters
        ----------
        metric : str
            One of "mean", "std", "min", "max" corresponding to the
            statistics stored in the stats.json file.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping variable names to their statistics
        """
        if self.dataset.stats is None:
            raise RuntimeError("Statistics file not available for dataset.")

        if metric not in {"mean", "std", "min", "max"}:
            raise ValueError(f"Unknown metric '{metric}'.")

        return {k: v[metric] for k, v in self.dataset.stats.items()}

    def __iter__(self):
        """
        Iterate over batches and move graph data to device.
        """
        for batch_snapshots in super().__iter__():
            # batch_snapshots is a list of Batch objects (one per timestep)
            # Move each batched graph to the device
            for batched_graph in batch_snapshots:
                batched_graph.to(self.device)
            yield batch_snapshots
