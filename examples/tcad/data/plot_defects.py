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

"""
TCAD Defect Data Visualization Script

This script processes TCAD simulation data from defect_data files and creates
animated 3D scatter plots showing defect evolution over time.

Usage:
    python plot_defects.py --thickness t_4 --simulation_id 0 \
        --color_by "Charge"
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(
    data_dir: str, thickness: str, simulation_id: int
) -> Tuple[Dict[str, Dict[int, np.ndarray]], List[str], int]:
    """
    Load TCAD defect data into nested dictionary structure.

    Args:
        data_dir: Path to the defect_data directory
        thickness: Thickness directory name (e.g., 't_4', 't_5', 't_6')
        simulation_id: Simulation ID to load (e.g., 0, 1, 2, ...)

    Returns:
        data_dict: Dict[variable_name, Dict[defect_id, np.array_of_timesteps]]
        headers: List of header names (excluding ID)
        num_timesteps: Number of timesteps found
    """
    # Find all files for the given simulation_id
    pattern = os.path.join(data_dir, thickness, f"defect_data_{simulation_id}_*.tsv")
    files = glob.glob(pattern)

    if not files:
        raise ValueError(
            f"No files found for simulation_id {simulation_id} in {thickness}"
        )

    # Extract timestep numbers and sort
    timestep_files = []
    for file in files:
        match = re.search(r"defect_data_\d+_(\d+)\.tsv$", file)
        if match:
            timestep = int(match.group(1))
            timestep_files.append((timestep, file))

    timestep_files.sort(key=lambda x: x[0])
    num_timesteps = len(timestep_files)

    print(f"Loading {num_timesteps} timesteps for simulation {simulation_id}")

    # Read first file to get headers
    first_file = timestep_files[0][1]
    with open(first_file, "r") as f:
        lines = f.readlines()

    # Find header line
    header_line = None
    for line in lines:
        if line.startswith("ID of the defect/ion"):
            header_line = line.strip()
            break

    if header_line is None:
        raise ValueError("Could not find header line in file")

    # Parse headers (skip ID column as it's treated specially)
    headers = [h.strip() for h in header_line.split("\t")]
    variable_headers = headers[1:]

    print(f"Found {len(variable_headers)} variables")

    # Initialize data structure
    data_dict = {var: {} for var in variable_headers}
    all_defect_ids = set()

    # First pass: collect all defect IDs across timesteps
    for timestep, file_path in timestep_files:
        try:
            df = pd.read_csv(file_path, sep="\t", skiprows=3, na_values=["None"])
            if not df.empty and "ID of the defect/ion" in df.columns:
                defect_ids = df["ID of the defect/ion"].dropna().astype(int)
                all_defect_ids.update(defect_ids)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue

    print(f"Found {len(all_defect_ids)} unique defects")

    # Initialize arrays for all defects and variables
    for var in variable_headers:
        for defect_id in all_defect_ids:
            data_dict[var][defect_id] = np.full(num_timesteps, np.nan)

    # Second pass: fill in the data
    for i, (timestep, file_path) in enumerate(timestep_files):
        try:
            df = pd.read_csv(file_path, sep="\t", skiprows=3, na_values=["None"])
            if df.empty or "ID of the defect/ion" not in df.columns:
                continue

            for _, row in df.iterrows():
                defect_id = row["ID of the defect/ion"]
                if pd.isna(defect_id):
                    continue

                defect_id = int(defect_id)

                # Fill data for each variable
                for var in variable_headers:
                    if var in row:
                        value = row[var]
                        if pd.notna(value):
                            try:
                                data_dict[var][defect_id][i] = float(value)
                            except (ValueError, TypeError):
                                pass  # Keep as NaN if conversion fails

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    return data_dict, variable_headers, num_timesteps


def find_color_variable(variable_headers: List[str], color_prefix: str) -> str:
    """
    Find the first variable that starts with the given prefix.

    Args:
        variable_headers: List of all available variable names
        color_prefix: Prefix to search for

    Returns:
        Full variable name that starts with the prefix

    Raises:
        ValueError: If no variable starts with the prefix
    """
    for var in variable_headers:
        if var.startswith(color_prefix):
            return var

    # If not found, show available options
    print("Available variables:")
    for i, var in enumerate(variable_headers):
        print(f"  {i}: {var}")

    raise ValueError(
        f"No variable found starting with '{color_prefix}'. "
        f"Available variables listed above."
    )


def plot_data(
    data_dict: Dict[str, Dict[int, np.ndarray]],
    variable_headers: List[str],
    num_timesteps: int,
    thickness: str,
    simulation_id: int,
    color_by: str = "Charge",
    save_animation: bool = False,
    output_file: str | None = None,
) -> None:
    """
    Create animated 3D scatter plot of defect data.

    Args:
        data_dict: Loaded data dictionary
        variable_headers: List of variable names
        num_timesteps: Number of timesteps
        thickness: Thickness identifier (e.g., 't_4')
        simulation_id: Simulation ID
        color_by: Variable prefix to use for coloring points
        save_animation: Whether to save animation as file
        output_file: Output filename for saved animation
    """
    # Find the color variable
    color_variable = find_color_variable(variable_headers, color_by)
    print(f"Using '{color_variable}' for point coloring")

    # Get coordinate and color data
    x_data = data_dict.get("X coordinate (nm)", {})
    y_data = data_dict.get("Y coordinate (nm)", {})
    z_data = data_dict.get("Z coordinate (nm)", {})
    color_data = data_dict.get(color_variable, {})

    # Validate required data
    if not x_data or not y_data or not z_data:
        raise ValueError("Missing required coordinate data (X, Y, Z)")

    defect_ids = list(x_data.keys())

    # Extract thickness value for title
    thickness_value = thickness.replace("t_", "")

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate axis ranges for consistent limits
    all_x = np.concatenate([x_data[did] for did in defect_ids])
    all_y = np.concatenate([y_data[did] for did in defect_ids])
    all_z = np.concatenate([z_data[did] for did in defect_ids])
    all_colors = np.concatenate(
        [color_data[did] for did in defect_ids if did in color_data]
    )

    x_range = [np.nanmin(all_x), np.nanmax(all_x)]
    y_range = [np.nanmin(all_y), np.nanmax(all_y)]
    z_range = [np.nanmin(all_z), np.nanmax(all_z)]

    # Color range for consistent colorbar
    valid_colors = all_colors[~np.isnan(all_colors)]
    if len(valid_colors) > 0:
        color_range = [np.min(valid_colors), np.max(valid_colors)]
    else:
        color_range = [0, 1]

    # Set axis limits with padding
    padding_x = (x_range[1] - x_range[0]) * 0.05
    padding_y = (y_range[1] - y_range[0]) * 0.05
    padding_z = (z_range[1] - z_range[0]) * 0.05
    ax.set_xlim(x_range[0] - padding_x, x_range[1] + padding_x)
    ax.set_ylim(y_range[0] - padding_y, y_range[1] + padding_y)
    ax.set_zlim(z_range[0] - padding_z, z_range[1] + padding_z)

    # Create initial scatter plot for colorbar
    scatter = ax.scatter(
        [],
        [],
        [],
        c=[],
        s=50,
        alpha=0.7,
        cmap="viridis",
        vmin=color_range[0],
        vmax=color_range[1],
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_variable, rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel("X coordinate (nm)")
    ax.set_ylabel("Y coordinate (nm)")
    ax.set_zlabel("Z coordinate (nm)")
    title = (
        f"Defect Evolution - Thickness {thickness_value}nm - Simulation {simulation_id}"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add subtitle for timestep info
    subtitle_text = fig.text(
        0.02,
        0.95,
        "",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    def animate(frame):
        """Animation function for each timestep."""
        # Clear previous plot
        ax.clear()

        # Reset axis properties
        ax.set_xlim(x_range[0] - padding_x, x_range[1] + padding_x)
        ax.set_ylim(y_range[0] - padding_y, y_range[1] + padding_y)
        ax.set_zlim(z_range[0] - padding_z, z_range[1] + padding_z)
        ax.set_xlabel("X coordinate (nm)")
        ax.set_ylabel("Y coordinate (nm)")
        ax.set_zlabel("Z coordinate (nm)")
        ax.set_title(
            f"Defect Evolution - Thickness {thickness_value}nm - "
            f"Simulation {simulation_id}"
        )

        # Collect data for current timestep
        current_x, current_y, current_z, current_colors = [], [], [], []

        for defect_id in defect_ids:
            x_val = x_data[defect_id][frame]
            y_val = y_data[defect_id][frame]
            z_val = z_data[defect_id][frame]

            # Only include points with valid coordinates
            if not (np.isnan(x_val) or np.isnan(y_val) or np.isnan(z_val)):
                current_x.append(x_val)
                current_y.append(y_val)
                current_z.append(z_val)

                # Get color value (use 0 as default for missing data)
                if defect_id in color_data:
                    color_val = color_data[defect_id][frame]
                    if np.isnan(color_val):
                        color_val = 0
                else:
                    color_val = 0
                current_colors.append(color_val)

        # Plot current timestep data
        if current_x:
            ax.scatter(
                current_x,
                current_y,
                current_z,
                c=current_colors,
                s=50,
                alpha=0.7,
                cmap="viridis",
                vmin=color_range[0],
                vmax=color_range[1],
            )

        # Update subtitle
        num_active_defects = len(current_x)
        subtitle_text.set_text(
            f"Timestep {frame}, Number of defects = {num_active_defects}"
        )

        return ax.collections, subtitle_text

    # Create animation
    print(f"Creating animation with {num_timesteps} frames...")
    ani = animation.FuncAnimation(
        fig, animate, frames=num_timesteps, interval=200, blit=False, repeat=True
    )

    # Save animation if requested
    if save_animation:
        if output_file is None:
            output_file = f"defect_animation_{thickness}_sim{simulation_id}.gif"

        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer="pillow", fps=5)
        print(f"Animation saved as {output_file}")

    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the defect visualization script."""
    parser = argparse.ArgumentParser(description="Visualize TCAD defect data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/defect_data",
        help="Path to defect_data directory",
    )
    parser.add_argument(
        "--thickness",
        type=str,
        required=True,
        help="Thickness directory (e.g., 't_4', 't_5', 't_6')",
    )
    parser.add_argument(
        "--simulation_id", type=int, required=True, help="Simulation ID to visualize"
    )
    parser.add_argument(
        "--color_by",
        type=str,
        default="Charge",
        help="Variable prefix to use for point coloring",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save animation as GIF file"
    )
    parser.add_argument(
        "--output", type=str, help="Output filename for saved animation"
    )

    args = parser.parse_args()

    # Validate data directory
    full_data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    if not os.path.exists(full_data_dir):
        print(f"Error: Data directory {full_data_dir} does not exist")
        return

    thickness_dir = os.path.join(full_data_dir, args.thickness)
    if not os.path.exists(thickness_dir):
        print(f"Error: Thickness directory {thickness_dir} does not exist")
        return

    try:
        # Load data
        print(f"Loading data for {args.thickness}, simulation {args.simulation_id}...")
        data_dict, headers, num_timesteps = load_data(
            full_data_dir, args.thickness, args.simulation_id
        )

        # Create visualization
        plot_data(
            data_dict,
            headers,
            num_timesteps,
            args.thickness,
            args.simulation_id,
            args.color_by,
            args.save,
            args.output,
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
