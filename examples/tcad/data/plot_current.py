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


import matplotlib.pyplot as plt
import numpy as np


def plot_current_data(filename):
    """
    Plot current vs time data from a file containing multiple data blocks.

    Args:
        filename (str): Path to the data file containing Time (s) and Current (A) data
    """
    # Initialize lists to store the multiple data blocks
    data_blocks = []
    current_time = []
    current_current = []

    # Read the file and parse the data
    with open(filename, "r") as file:
        lines = file.readlines()

    # Parse the data
    reading_data = False
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if this is an end marker
        if line.startswith("#END"):
            # Save current data block if we have data
            if current_time and current_current:
                data_blocks.append((np.array(current_time), np.array(current_current)))
                current_time = []
                current_current = []
            reading_data = False
        elif "Time (s)" in line and "Current (A)" in line:
            # This is a header line, start reading data
            reading_data = True
        elif reading_data and "\t" in line:
            # This is a data line
            try:
                parts = line.split("\t")
                if len(parts) >= 2:
                    time_val = float(parts[0])
                    current_val = float(parts[1])
                    current_time.append(time_val)
                    current_current.append(current_val)
            except ValueError:
                # Skip lines that can't be parsed as floats
                continue

    # Add the last data block if we have data
    if current_time and current_current:
        data_blocks.append((np.array(current_time), np.array(current_current)))

    # Create the plot with taller aspect ratio
    plt.figure(figsize=(12, 10.5))

    # Generate colors for each data block
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_blocks)))

    # Plot each data block
    for i, (time_data, current_data) in enumerate(data_blocks):
        plt.plot(
            time_data,
            current_data,
            color=colors[i],
            linewidth=1.5,
            label=f"Block {i + 1}",
            marker="o",
            markersize=2,
            alpha=0.8,
        )

    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")

    # Extract thickness from filename for title
    if "t_4" in filename:
        thickness = "4"
    elif "t_5" in filename:
        thickness = "5"
    elif "t_6" in filename:
        thickness = "6"
    else:
        thickness = "Unknown"

    plt.title(f"Current vs Time - Thickness {thickness}")
    plt.yscale("log")  # Use log scale for current since values are very small
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_filename = f"current_time_t_{thickness}_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_filename}")
    print(f"Found {len(data_blocks)} data blocks in the file")

    # Show the plot
    plt.show()

    return data_blocks


if __name__ == "__main__":
    # Plot current data for thickness 4
    print("Plotting I_time_t_4.txt...")
    plot_current_data("I_time_t_4.txt")

    # Plot current data for thickness 5
    print("Plotting I_time_t_5.txt...")
    plot_current_data("I_time_t_5.txt")

    # Plot current data for thickness 6
    print("Plotting I_time_t_6.txt...")
    plot_current_data("I_time_t_6.txt")
