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

"""SongUNet adapter satisfying the DiffusionModel protocol.

Not counted in LoC: instrumentation glue the user wouldn't have to write if
SongUNet exposed the modern protocol natively.

The adapter also routes the multi-diffusion positional embedding: when MD is
used with ``positional_embedding="sinusoidal"`` (or any other value), the
framework injects a pre-patched ``"positional_embedding"`` tensor into the
``condition`` TensorDict. The adapter concatenates that tensor along the
channel axis before calling SongUNet, so the backbone receives an input with
``data_C + pos_emb_C`` channels.
"""

from __future__ import annotations

import torch
from physicsnemo.core import Module
from physicsnemo.models.diffusion_unets import SongUNet
from tensordict import TensorDict


class SongUNetAdapter(Module):
    """Wrap a SongUNet so it satisfies the DiffusionModel protocol.

    DiffusionModel: ``(x, t, condition=None, **kwargs) -> Tensor``.
    SongUNet:       ``(x, noise_labels, class_labels=None, augment_labels=None)``.
    """

    def __init__(self, song_unet: SongUNet) -> None:
        super().__init__()
        self.model = song_unet

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition=None,
        **kwargs,
    ) -> torch.Tensor:
        # MD positional embedding (and any other tensor concat-style conditioning)
        # arrives via the ``condition`` TensorDict under the key
        # ``"positional_embedding"`` (set by MultiDiffusionModel2D). Concatenate
        # along the channel dim so SongUNet sees ``in_channels = C_data + C_pe``.
        if (
            isinstance(condition, TensorDict)
            and "positional_embedding" in condition.keys()
        ):
            pe = condition["positional_embedding"]
            if pe.dtype != x.dtype:
                pe = pe.to(dtype=x.dtype)
            x = torch.cat([x, pe], dim=1)
        return self.model(x, t, None, None)
