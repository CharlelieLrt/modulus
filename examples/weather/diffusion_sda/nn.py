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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd

from physicsnemo.diffusion.multi_diffusion import BasePatching2D


from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd

from physicsnemo.diffusion.multi_diffusion import BasePatching2D


class HRRRSurfaceDiffusionNet(Module):
    """
    Adapter to make a model callable with the correct
    signature ``forward(x, t, condition, **model_kwargs) -> torch.Tensor``.
    """

    # HRRR grid, 16 variables with 4 conditions
    IMG_RESOLUTION = [1059, 1799]
    IMG_CHANNELS = 16
    CONDITION_CHANNELS = 4

    def __init__(self, use_apex: bool = False):
        super().__init__(meta=ModelMetaData())

        # Multi-diffusion paramters, defines how large a single diffusion patch is
        patch_shape = (448, 448)
        patch_num = 4

        self.sigma_data = 1

        # Create model
        channel_mult = [1, 2, 2, 2, 2]
        num_grid_channels = 8
        self.model = SongUNetPosEmbd(
            img_resolution=self.IMG_RESOLUTION,
            in_channels=self.IMG_CHANNELS + self.CONDITION_CHANNELS + num_grid_channels,
            out_channels=self.IMG_CHANNELS,
            N_grid_channels=num_grid_channels,
            gridtype="learnable",
            model_channels=128,
            channel_mult=channel_mult,
            attn_resolutions=[self.IMG_RESOLUTION[0] >> len(channel_mult)],
            use_apex_gn=use_apex,
        )

    def forward(
        self, x: Tensor, sigma: Tensor, condition: Dict[str, Tensor], **model_kwargs
    ):
        _, c = next(iter(condition.items()))

        x = torch.cat([x, c], dim=1)

        out = self.model(x, sigma, None, **model_kwargs)
        # Final affine transformation
        return out
