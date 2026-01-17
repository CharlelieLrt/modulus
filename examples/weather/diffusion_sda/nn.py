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

from typing import Dict, List, Literal, Union, Any

import torch
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd
from physicsnemo.nn import PositionalEmbedding


class HRRRSurfaceDiffusionNet(Module):
    """
    HRRR Surface diffusion network.
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        time_embed_channels: int,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        attn_resolutions: List[int] = [28],
        gridtype: Literal["sinusoidal", "learnable", "linear", "test"] = "sinusoidal",
        N_grid_channels: int = 4,
        use_apex_gn: bool = False,
    ):
        super().__init__(meta=ModelMetaData())

        self.time_embed_channels = time_embed_channels

        # Create the time embedding layer
        self.time_embedding = PositionalEmbedding(
            num_channels=time_embed_channels,
            max_positions=365,
            endpoint=True,
            learnable=True,
        )

        self.unet = SongUNetPosEmbd(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            gridtype=gridtype,
            N_grid_channels=N_grid_channels,
            use_apex_gn=use_apex_gn,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Dict[str, Tensor],
        **model_kwargs: Any,
    ) -> Tensor:
        B, C, H, W = x.shape

        # Get space and time conditionings
        cs = condition["cond_spatial"]  # (B, C, H, W)
        ct = condition["cond_time"]  # (B, 1)

        # Embed time conditioning
        ct_embed = self.time_embedding(ct.squeeze(1))  # (B, time_embed_channels)
        ct_embed = ct_embed[:, :, None, None].expand(
            B, -1, H, W
        )  # (B, time_embed_channels, H, W)

        x_concat = torch.cat([x, cs, ct_embed], dim=1)

        return self.unet(x_concat, t, None, **model_kwargs)
