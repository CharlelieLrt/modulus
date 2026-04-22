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

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Columns to keep from the TSV files
COLUMNS_TO_KEEP = {
    "ID of the defect/ion": "id",
    "X coordinate (nm)": "x",
    "Y coordinate (nm)": "y",
    "Z coordinate (nm)": "z",
    "State D(Q-1) thermal ionization energy (eV)": "ion_ene",
    "Generation time (s)": "gen_time",
}


def parse_defect_filename(filename: str) -> Tuple[int, int]:
    """
    Parse defect data filename to extract simulation ID and timestamp ID.

    Parameters
    ----------
    filename : str
        Filename in format 'defect_data_i_j.tsv'

    Returns
    -------
    Tuple[int, int]
        Tuple of (simulation_id, timestamp_id)
    """
    match = re.match(r"defect_data_(\d+)_(\d+)\.tsv", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    sim_id = int(match.group(1))
    timestamp_id = int(match.group(2))
    return sim_id, timestamp_id


def parse_thickness_from_dirname(dirname: str) -> float:
    """
    Parse thickness value from directory name.

    Parameters
    ----------
    dirname : str
        Directory name in format 't_xxx'

    Returns
    -------
    float
        Thickness value
    """
    match = re.match(r"t_(\d+)", dirname)
    if not match:
        raise ValueError(f"Invalid directory name format: {dirname}")
    return float(match.group(1))


def read_time_data(time_file: Path) -> Dict[int, np.ndarray]:
    """
    Read time data from I_time_t_x.txt file.

    The file contains multiple simulations separated by "#END" markers.
    Each simulation has its own time series.

    Parameters
    ----------
    time_file : Path
        Path to the time data file

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping simulation index to array of time values
    """
    time_data = {}
    current_sim_idx = 0
    current_times = []

    with open(time_file, "r") as f:
        for line in f:
            line = line.strip()

            # Check for simulation end marker
            if line == "#END":
                if current_times:
                    time_data[current_sim_idx] = np.array(current_times)
                    current_sim_idx += 1
                    current_times = []
                continue

            # Skip empty lines and header lines
            if line == "" or line.startswith("Time") or line.startswith("#"):
                continue

            # Parse time value
            parts = line.split("\t")
            if len(parts) == 2:
                try:
                    time_val = float(parts[0])
                    current_times.append(time_val)
                except ValueError:
                    pass

    # Add the last simulation if file doesn't end with #END
    if current_times:
        time_data[current_sim_idx] = np.array(current_times)

    return time_data


def get_global_metadata(input_dir: Path) -> Tuple[int, int]:
    """
    Get global metadata by scanning all subdirectories.

    Parameters
    ----------
    input_dir : Path
        Input directory containing defect_data/ subdirectory

    Returns
    -------
    Tuple[int, int]
        Tuple of (total_simulations, total_files)
    """
    logging.info("Scanning input directory for metadata...")

    total_simulations = 0
    total_files = 0
    total_time_rows = 0

    # Look for defect_data subdirectory
    defect_data_dir = input_dir / "defect_data"
    if not defect_data_dir.exists():
        raise ValueError(f"defect_data subdirectory not found in {input_dir}")

    # Find all t_xxx subdirectories inside defect_data/
    subdirs = sorted(
        [d for d in defect_data_dir.iterdir() if d.is_dir() and d.name.startswith("t_")]
    )

    if not subdirs:
        raise ValueError(f"No t_xxx subdirectories found in {defect_data_dir}")

    for subdir in subdirs:
        logging.info(f"Scanning {subdir.name}...")

        # Find the specific time data file for this subdirectory
        # For subdir "t_4", this looks for "I_time_t_4.txt"
        time_file_path = input_dir / f"I_time_{subdir.name}.txt"

        if not time_file_path.exists():
            logging.warning(
                f"  Time data file I_time_{subdir.name}.txt not found, skipping"
            )
            continue

        # Read time data for all simulations in this subdirectory
        time_data = read_time_data(time_file_path)

        # Count simulations
        num_sims_in_dir = len(time_data)
        total_simulations += num_sims_in_dir

        # Count total time rows (timestamps) across all simulations
        num_time_rows = sum(len(time_array) for time_array in time_data.values())
        total_time_rows += num_time_rows

        # Count all defect data files in this subdirectory
        defect_files = list(subdir.glob("defect_data_*.tsv"))
        num_files_in_dir = len(defect_files)
        total_files += num_files_in_dir

        logging.info(
            f"  Found {num_sims_in_dir} simulations, "
            f"{num_files_in_dir} defect files, "
            f"{num_time_rows} time rows"
        )

    # Safety check: verify counts match
    if total_files != total_time_rows:
        raise ValueError(
            f"Data mismatch detected!\n"
            f"  Number of defect data files: {total_files}\n"
            f"  Number of time data rows: {total_time_rows}\n"
            f"These counts should match. Each timestamp should have a "
            f"corresponding defect file."
        )

    logging.info("  Data consistency check passed!")

    return total_simulations, total_files


def process_defect_file(
    file_path: Path,
    thickness: float,
    time_value: float,
    output_sim_dir: Path,
    shifted_timestamp_id: int,
    num_new_defects: int,
    new_defects_ids: set,
) -> None:
    """
    Process a single defect data file and save as NPZ.

    Parameters
    ----------
    file_path : Path
        Path to the TSV file
    thickness : float
        Thickness value from directory name
    time_value : float
        Time value for this timestamp
    output_sim_dir : Path
        Output directory for this simulation's files
    shifted_timestamp_id : int
        Timestamp ID shifted to start at 0
    num_new_defects : int
        Number of new defects compared to previous timestep
    new_defects_ids : set
        Set of defect IDs that are new at this timestep
    """

    # # DEBUG
    # if num_new_defects not in [0, 2]:
    #     print(f"New defects: {num_new_defects}, new defects ids: {new_defects_ids}, file: {file_path}")

    # Read the TSV file
    df = pd.read_csv(file_path, sep="\t", comment="#", skiprows=2)

    # Check if file is empty
    if len(df) == 0:
        logging.warning(f"Empty file: {file_path}")
        return

    # Extract and rename columns
    data_dict = {}
    for old_name, new_name in COLUMNS_TO_KEEP.items():
        if old_name in df.columns:
            data_dict[new_name] = df[old_name].values
        else:
            logging.warning(f"Column '{old_name}' not found in {file_path}")
            return

    # Compute gen_delay (difference in generation time between rows)
    gen_time = data_dict["gen_time"]
    gen_delay = np.zeros_like(gen_time)
    gen_delay[1:] = gen_time[1:] - gen_time[:-1]

    # Compute transformed generation delay
    eps, beta = 1e-12, 2.0
    data_dict["invsoftplus_gen_delay"] = (
        np.log(np.expm1(beta * (gen_delay + eps))) / beta
    )

    # Create is_new_defect indicator array (1 if new, 0 if not)
    defect_ids = data_dict["id"]
    is_new_defect = np.array(
        [1 if int(defect_id) in new_defects_ids else 0 for defect_id in defect_ids],
        dtype=np.int32,
    )
    data_dict["is_new_defect"] = is_new_defect

    # Compute invsoftplus transformation of time value
    invsoftplus_time = np.log(np.expm1(beta * (time_value + eps))) / beta

    # Store scalar metadata as single-element arrays
    n_defects = len(df)
    data_dict["thickness"] = np.array([thickness], dtype=np.float32)
    data_dict["time"] = np.array([time_value], dtype=np.float32)
    data_dict["invsoftplus_time"] = np.array([invsoftplus_time], dtype=np.float32)
    data_dict["num_defects"] = np.array([n_defects], dtype=np.int32)
    data_dict["num_new_defects"] = np.array([num_new_defects], dtype=np.int32)

    # Save as NPZ in the simulation subdirectory
    output_file = output_sim_dir / f"sample_{shifted_timestamp_id}.npz"
    np.savez(output_file, **data_dict)


def process_directory(
    subdir: Path,
    input_dir: Path,
    output_dir: Path,
    sim_id_mapping: Dict[Tuple[str, int], int],
    filename_to_timedata_mapping: Dict[Tuple[str, int], int],
) -> None:
    """
    Process all files in a single t_xxx subdirectory.

    Parameters
    ----------
    subdir : Path
        Subdirectory to process (inside defect_data/)
    input_dir : Path
        Input directory containing I_time_t_*.txt files
    output_dir : Path
        Output directory for NPZ files
    sim_id_mapping : Dict[Tuple[str, int], int]
        Mapping from (subdirectory_name, filename_sim_id) to global_sim_id
    filename_to_timedata_mapping : Dict[Tuple[str, int], int]
        Mapping from (subdirectory_name, filename_sim_id) to time_data_index
    """
    # Parse thickness from directory name
    thickness = parse_thickness_from_dirname(subdir.name)
    logging.info(f"Processing {subdir.name} (thickness={thickness})")

    # Find the specific time data file for this subdirectory
    # For subdir "t_4", this looks for "I_time_t_4.txt" in input_dir
    time_file_pattern = f"I_time_{subdir.name}.txt"
    time_file_path = input_dir / time_file_pattern

    if not time_file_path.exists():
        logging.warning(f"Time data file {time_file_pattern} not found, skipping")
        return

    # Read time data for all simulations in this subdirectory
    time_data = read_time_data(time_file_path)
    logging.info(f"  Loaded time data with {len(time_data)} simulations")

    # Find all defect data files in this subdirectory
    defect_files = list(subdir.glob("defect_data_*.tsv"))
    logging.info(f"  Found {len(defect_files)} defect data files")

    # Group files by simulation
    files_by_sim = {}
    for file in defect_files:
        try:
            filename_sim_id, timestamp_id = parse_defect_filename(file.name)
            key = (subdir.name, filename_sim_id)
            if key not in files_by_sim:
                files_by_sim[key] = []
            files_by_sim[key].append((timestamp_id, file))
        except ValueError as e:
            logging.warning(f"  Skipping {file.name}: {e}")

    # Process each simulation
    files_processed = 0
    for (subdir_name, filename_sim_id), file_list in files_by_sim.items():
        # Get global simulation ID
        global_sim_id = sim_id_mapping.get((subdir_name, filename_sim_id))
        if global_sim_id is None:
            logging.warning(
                f"  No global ID mapping for {subdir_name}, sim {filename_sim_id}"
            )
            continue

        # Get the time data index
        time_idx = filename_to_timedata_mapping.get((subdir_name, filename_sim_id))
        if time_idx is None:
            logging.warning(
                f"  No time data index mapping for {subdir_name}, sim {filename_sim_id}"
            )
            continue

        # Sort files by timestamp
        file_list.sort(key=lambda x: x[0])
        timestamps = [ts for ts, _ in file_list]

        # Check that timestamps are contiguous
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        expected_timestamps = set(range(min_ts, max_ts + 1))
        actual_timestamps = set(timestamps)

        if expected_timestamps != actual_timestamps:
            missing = expected_timestamps - actual_timestamps
            raise ValueError(
                f"Non-contiguous timestamps for simulation "
                f"{filename_sim_id} in {subdir_name}!\n"
                f"  Expected: {sorted(expected_timestamps)}\n"
                f"  Found: {sorted(actual_timestamps)}\n"
                f"  Missing: {sorted(missing)}"
            )

        # Create output subdirectory for this simulation
        output_sim_dir = output_dir / f"sim_{global_sim_id}"
        output_sim_dir.mkdir(parents=True, exist_ok=True)

        # Collect metadata for info.json
        time_values = []
        total_defects = 0
        new_defects_values = []
        previous_ids = set()

        # Process each file with shifted timestamp
        for timestamp_id, file in file_list:
            try:
                # Get time value
                if time_idx not in time_data:
                    logging.warning(f"  No time data for time index {time_idx}")
                    continue

                sim_times = time_data[time_idx]
                if timestamp_id >= len(sim_times):
                    logging.warning(f"  Timestamp {timestamp_id} out of range")
                    continue

                time_value = float(sim_times[timestamp_id])
                time_values.append(time_value)

                # Read defect IDs from this file
                df_temp = pd.read_csv(file, sep="\t", comment="#", skiprows=2)
                total_defects += len(df_temp)

                # Get current defect IDs
                current_ids = set(df_temp["ID of the defect/ion"].values)

                # Compute new defects (IDs not in previous timestep)
                if timestamp_id == min_ts:
                    # First timestep: set num_new_defects to 0, no new IDs
                    num_new_defects = 0
                    new_defects_ids = set()
                else:
                    # IDs that are new (not in previous set)
                    new_defects_ids = current_ids - previous_ids
                    num_new_defects = len(new_defects_ids)

                new_defects_values.append(num_new_defects)
                previous_ids = current_ids

                # Shift timestamp to start at 0
                shifted_timestamp_id = timestamp_id - min_ts

                # Process the file
                process_defect_file(
                    file,
                    thickness,
                    time_value,
                    output_sim_dir,
                    shifted_timestamp_id,
                    num_new_defects,
                    new_defects_ids,
                )
                files_processed += 1

            except Exception as e:
                logging.error(f"  Error processing {file.name}: {e}")

        # Create info.json with simulation metadata
        if time_values:
            info = {
                "global_sim_id": global_sim_id,
                "original_sim_id": filename_sim_id,
                "source_ensemble": subdir_name,
                "source_directory": str(subdir),
                "source_file_pattern": f"defect_data_{filename_sim_id}_*.tsv",
                "num_timesteps": len(file_list),
                "thickness": thickness,
                "original_timestamp_range": [min_ts, max_ts],
                "time_range": [min(time_values), max(time_values)],
                "total_defects_all_timesteps": total_defects,
                "max_new_defects": max(new_defects_values) if new_defects_values else 0,
            }

            info_file = output_sim_dir / "info.json"
            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)

    logging.info(f"  Processed {files_processed} files from {subdir.name}")


def reorganize_data(input_dir: Path, output_dir: Path) -> None:
    """
    Reorganize defect data files into NPZ format.

    Parameters
    ----------
    input_dir : Path
        Input directory containing defect_data/ subdirectory and
        I_time_t_*.txt files
    output_dir : Path
        Output directory for NPZ files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get global metadata
    total_simulations, total_files = get_global_metadata(input_dir)

    # Look for defect_data subdirectory
    defect_data_dir = input_dir / "defect_data"
    if not defect_data_dir.exists():
        raise ValueError(f"defect_data subdirectory not found in {input_dir}")

    # Find all t_xxx subdirectories inside defect_data/
    subdirs = sorted(
        [d for d in defect_data_dir.iterdir() if d.is_dir() and d.name.startswith("t_")]
    )
    num_ensembles = len(subdirs)

    logging.info(f"Number of ensembles (t_xxx directories): {num_ensembles}")
    logging.info(f"Total simulations across all ensembles: {total_simulations}")
    logging.info(f"Total files to process: {total_files}")

    # Create mapping from (subdir, filename_sim_id) to global simulation ID
    # Also create mapping from (subdir, filename_sim_id) to time_data_index
    sim_id_mapping = {}
    filename_to_timedata_mapping = {}
    global_id_counter = 0

    # First pass: create the mappings
    for subdir in subdirs:
        # Find the specific time data file for this subdirectory
        time_file_path = input_dir / f"I_time_{subdir.name}.txt"

        if not time_file_path.exists():
            logging.warning(
                f"Time data file I_time_{subdir.name}.txt not found, skipping"
            )
            continue

        time_data = read_time_data(time_file_path)
        num_sims_in_dir = len(time_data)

        # Discover actual sim IDs from filenames in this subdirectory
        defect_files = list(subdir.glob("defect_data_*.tsv"))
        actual_sim_ids = set()
        for file in defect_files:
            try:
                sim_id, _ = parse_defect_filename(file.name)
                actual_sim_ids.add(sim_id)
            except ValueError:
                pass

        actual_sim_ids = sorted(actual_sim_ids)

        if len(actual_sim_ids) != num_sims_in_dir:
            logging.warning(
                f"{subdir.name}: Found {len(actual_sim_ids)} unique sim IDs "
                f"in filenames but {num_sims_in_dir} simulations in time "
                f"data. They should match!"
            )

        start_id = global_id_counter

        # Map each filename sim_id to a time_data index and global ID
        for time_idx, filename_sim_id in enumerate(actual_sim_ids):
            # Map (subdir, filename_sim_id) -> global_id
            sim_id_mapping[(subdir.name, filename_sim_id)] = global_id_counter
            # Map (subdir, filename_sim_id) -> time_data_index
            key = (subdir.name, filename_sim_id)
            filename_to_timedata_mapping[key] = time_idx
            global_id_counter += 1

        logging.info(
            f"{subdir.name}: {num_sims_in_dir} simulations "
            f"(filename IDs: {actual_sim_ids[0]}-{actual_sim_ids[-1]}) -> "
            f"global IDs [{start_id}..{global_id_counter - 1}]"
        )

    # Second pass: process all files
    for subdir in subdirs:
        process_directory(
            subdir,
            input_dir,
            output_dir,
            sim_id_mapping,
            filename_to_timedata_mapping,
        )

    logging.info(f"Reorganization complete! Files saved to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess TCAD defect data into NPZ format."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing t_xxx subdirectories with TSV files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed NPZ files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        exit(1)

    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    reorganize_data(input_dir, output_dir)
