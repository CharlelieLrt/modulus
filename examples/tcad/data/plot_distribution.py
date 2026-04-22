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


def plot_weibull_distribution(filename, title_prefix="Weibull"):
    """
    Plot Weibull distributions from a data file containing 3 distributions.

    Args:
        filename (str): Path to the data file
        title_prefix (str): Prefix for the plot title (e.g., "Weibull TDDB" or "Weibull VDDB")
    """
    # Initialize lists to store the three distributions
    distributions = []
    current_x = []
    current_y = []

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

        # Check if this is a header line or end marker
        if line.startswith("#END"):
            # Save current distribution if we have data
            if current_x and current_y:
                distributions.append((np.array(current_x), np.array(current_y)))
                current_x = []
                current_y = []
            reading_data = False
        elif "Weibit" in line:
            # This is a header line, start reading data
            reading_data = True
        elif reading_data and "\t" in line:
            # This is a data line
            try:
                parts = line.split("\t")
                if len(parts) >= 2:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    current_x.append(x_val)
                    current_y.append(y_val)
            except ValueError:
                # Skip lines that can't be parsed as floats
                continue

    # Add the last distribution if we have data
    if current_x and current_y:
        distributions.append((np.array(current_x), np.array(current_y)))

    # Create the plot
    plt.figure(figsize=(12, 3.5))

    # Define colors and thickness labels
    colors = ["blue", "red", "green"]
    thicknesses = [4, 5, 6]

    # Plot each distribution
    for i, (x_data, y_data) in enumerate(distributions):
        if i < len(colors):
            plt.plot(
                x_data,
                y_data,
                color=colors[i],
                linewidth=2,
                label=f"Thickness = {thicknesses[i]}",
                marker="o",
                markersize=4,
            )

    # Customize the plot
    if "TDDB" in filename.upper():
        plt.xlabel("Time (s)")
        plt.title(f"{title_prefix} TDDB")
    elif "VDDB" in filename.upper():
        plt.xlabel("Voltage (V)")
        plt.title(f"{title_prefix} VDDB")
    else:
        plt.xlabel("X Value")
        plt.title(title_prefix)

    plt.ylabel("Weibit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    if "TDDB" in filename.upper():
        output_filename = "weibull_tddb_plot.png"
    elif "VDDB" in filename.upper():
        output_filename = "weibull_vddb_plot.png"
    else:
        output_filename = "weibull_plot.png"

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_filename}")

    # Show the plot
    plt.show()

    return distributions


if __name__ == "__main__":
    # Plot TDDB data
    print("Plotting TDDB distribution...")
    plot_weibull_distribution("weibull_TDDB.txt", "Weibull")

    # Plot VDDB data
    print("Plotting VDDB distribution...")
    plot_weibull_distribution("weibull_VDDB.txt", "Weibull")
