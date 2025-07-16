# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
# ruff: noqa: E402

import torch


def setup_model_lt_aware_ce_regression():

    from physicsnemo.models.diffusion import UNet

    torch.manual_seed(0)
    H, W = 48, 32
    C_lr, C_hr = 3, 4
    N_grid_channels, lead_time_channels = 4, 7

    model = UNet(
        img_resolution=(H, W),
        img_in_channels=C_lr + N_grid_channels + lead_time_channels,
        img_out_channels=C_hr,
        model_type="SongUNetPosLtEmbd",
        model_channels=16,
        channel_mult=[1, 2, 2],
        channel_mult_emb=2,
        num_blocks=2,
        attn_resolutions=[8],
        N_grid_channels=N_grid_channels,
        embedding_type="zero",
        lead_time_channels=lead_time_channels,
        lead_time_steps=3,
        prob_channels=[3],
    )
    return model


def setup_model_lt_aware_patched_diffusion():

    from physicsnemo.models.diffusion import EDMPrecondSuperResolution

    torch.manual_seed(0)
    H, W = 48, 32
    C_lr, C_hr = 3, 4
    N_grid_channels, lead_time_channels = 6, 7

    model = EDMPrecondSuperResolution(
        img_resolution=(H, W),
        img_in_channels=2 * C_lr + N_grid_channels + lead_time_channels + C_hr,
        img_out_channels=C_hr,
        model_type="SongUNetPosLtEmbd",
        scale_cond_input=False,
        model_channels=16,
        channel_mult=[1, 2, 2],
        channel_mult_emb=2,
        num_blocks=2,
        attn_resolutions=[8],
        N_grid_channels=N_grid_channels,
        gridtype="learnable",
        lead_time_channels=lead_time_channels,
        lead_time_steps=3,
        prob_channels=[3],
    )
    return model
