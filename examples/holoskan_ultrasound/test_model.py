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

"""Minimalistic script to test and measure timing of DiffusionFWINet forward pass."""

import time

import numpy as np
import torch

from nn import DiffusionFWINet


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    x_resolution = [32, 32]
    x_channels = 1
    y_resolution = [896, 180]
    y_channels = 5

    # Create model
    print("Creating model...")
    model = DiffusionFWINet(
        x_resolution=x_resolution,
        x_channels=x_channels,
        y_resolution=y_resolution,
        y_channels=y_channels,
    ).to(device)
    model.eval()

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create input tensors
    batch_size = 1
    x = torch.randn(batch_size, x_channels, *x_resolution, device=device)
    y = torch.randn(batch_size, y_channels, *y_resolution, device=device)
    sigma = torch.ones(batch_size, device=device)

    print(f"Input shapes: x={x.shape}, y={y.shape}, sigma={sigma.shape}")

    # Warmup runs
    print("Warming up...")
    num_warmup = 3
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x, y, sigma)

    # Timing runs
    num_runs = 10
    print(f"Running {num_runs} timed forward passes...")

    if device.type == "cuda":
        # GPU timing with CUDA events for accurate measurement
        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                _ = model(x, y, sigma)
            end_event.record()

            torch.cuda.synchronize()
            # elapsed_time returns milliseconds, convert to seconds
            times.append(start_event.elapsed_time(end_event) / 1000.0)
    else:
        # CPU timing with perf_counter
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x, y, sigma)
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    print(f"\nForward pass timing over {num_runs} runs:")
    print(f"  Mean: {times.mean():.6f} seconds")
    print(f"  Std:  {times.std():.6f} seconds")


if __name__ == "__main__":
    main()
