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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dataloader for the R2C HuggingFace ultrasound dataset.

Usage:
    from dataloader import load_r2c_dataset, collate_fn

    dataset = load_r2c_dataset("datasets/hf_r2c")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    for batch in loader:
        iq = torch.complex(batch["iq_real"], batch["iq_imag"])  # [B, n_tx, n_rx, n_samples]
        c_map = batch["sound_speed_map"]                        # [B, H, W]
        elpos = batch["elpos"]                                  # [B, 3, n_el]
        # scalars: batch["t0"], batch["fs"], batch["fd"], batch["c0"] are [B] tensors
"""

from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Stack HF dataset samples into batched tensors.

    Returns dict with keys:
        iq_real, iq_imag: [B, n_tx, n_rx, n_samples]
        sound_speed_map:  [B, H, W]
        elpos:            [B, 3, n_el]
        t0, fs, fd, c0:   [B] scalar tensors
    """
    return {
        "iq_real": torch.stack([b["iq_real"] for b in batch]),
        "iq_imag": torch.stack([b["iq_imag"] for b in batch]),
        "sound_speed_map": torch.stack([b["sound_speed_map"] for b in batch]),
        "elpos": torch.stack([b["elpos"] for b in batch]),
        "t0": torch.tensor([b["t0"] for b in batch]),
        "fs": torch.tensor([b["fs"] for b in batch]),
        "fd": torch.tensor([b["fd"] for b in batch]),
        "c0": torch.tensor([b["c0"] for b in batch]),
    }


def load_r2c_dataset(path="datasets/hf_r2c"):
    """Load R2C HuggingFace dataset, auto-detecting shards if present.

    Args:
        path: Path to HF dataset directory (single dataset or directory of shard_* folders)

    Returns:
        HuggingFace dataset with torch format, ready for use with DataLoader
    """
    dataset_path = Path(path)

    if (dataset_path / "dataset_info.json").exists():
        hf_dataset = load_from_disk(str(dataset_path))
    else:
        shards = sorted(
            p
            for p in dataset_path.iterdir()
            if p.is_dir() and p.name.startswith("shard_")
        )
        if shards:
            print(f"Found {len(shards)} shards, concatenating...")
            hf_dataset = concatenate_datasets([load_from_disk(str(s)) for s in shards])
        else:
            hf_dataset = load_from_disk(str(dataset_path))

    return hf_dataset.with_format("torch")


if __name__ == "__main__":
    # Quick sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Load R2C HuggingFace dataset")
    parser.add_argument(
        "--dataset-path", type=str, default="datasets/hf_r2c", help="Path to HF dataset"
    )
    args = parser.parse_args()

    dataset = load_r2c_dataset(args.dataset_path)
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
