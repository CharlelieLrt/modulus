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

from pathlib import Path

import pytest
import torch
from utils_models import (
    setup_model_lt_aware_ce_regression,
    setup_model_lt_aware_patched_diffusion,
)
from validate_utils import validate_accuracy


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_corrdiff_lt_aware_ce_regression(device):
    """
    Non-regression test for lead-time aware regression model (with CE loss) in
    a CorrDiff training loop. Compares results from v1.0.1 checkpoint with
    current model.
    """

    from physicsnemo.metrics.diffusion import RegressionLossCE
    from physicsnemo.models import Module

    reg_model_chkpt = Module.from_checkpoint(
        str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path("lt_aware_ce_regression_UNet_v1.0.1.mdlus")
        )
    ).to(device)
    reg_model = setup_model_lt_aware_ce_regression()
    loss_fn = RegressionLossCE(prob_channels=[3])

    # Generate data
    torch.manual_seed(0)
    B, H, W = 4, 48, 32
    C_lr, C_hr = 3, 4
    lead_time_steps = 3
    x_lr = torch.randn(B, C_lr, H, W).to(device)
    x_hr = torch.randn(B, C_hr, H, W).to(device)
    lead_time_label = torch.randint(0, lead_time_steps, (B,)).to(device)

    # Compute loss with v1.0.1 checkpoint
    loss_chkpt = loss_fn(reg_model_chkpt, x_hr, x_lr, lead_time_label=lead_time_label)
    validate_accuracy(loss_chkpt, file_name="lt_aware_ce_regression_loss_v1.1.1.pt")
    loss_chkpt = loss_chkpt.sum() / B
    loss_chkpt.backward()

    # Compute loss with current model
    loss = loss_fn(reg_model, x_hr, x_lr, lead_time_label=lead_time_label)
    validate_accuracy(loss, file_name="lt_aware_ce_regression_loss_v1.1.1.pt")
    loss = loss.sum() / B
    loss.backward()

    return


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_corrdiff_lt_aware_patched_diffusion(device):
    """
    Non-regression test for lead-time aware patched diffusion model with
    ResidualLoss in a CorrDiff training loop. Compares results from v1.0.1
    checkpoint with current model.
    """

    from physicsnemo.metrics.diffusion import ResidualLoss
    from physicsnemo.models import Module
    from physicsnemo.utils.patching import RandomPatching2D

    patching = RandomPatching2D(img_shape=(48, 32), patch_shape=(12, 12), patch_num=5)
    patch_nums_iter = [2, 2, 1]

    reg_model_chkpt = Module.from_checkpoint(
        str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path("lt_aware_ce_regression_UNet_v1.0.1.mdlus")
        )
    ).to(device)
    reg_model = setup_model_lt_aware_ce_regression()

    diff_model_chkpt = Module.from_checkpoint(
        str(
            Path(__file__).parents[1].resolve()
            / Path("data")
            / Path("lt_aware_patched_diffusion_EDMPrecondSR_v1.0.1.mdlus")
        )
    ).to(device)
    diff_model = setup_model_lt_aware_patched_diffusion()

    loss_fn_chkpt = ResidualLoss(
        regression_net=reg_model_chkpt, hr_mean_conditioning=True
    )
    loss_fn = ResidualLoss(regression_net=reg_model, hr_mean_conditioning=True)

    # Generate data
    torch.manual_seed(0)
    B, H, W = 4, 48, 32
    C_lr, C_hr = 3, 4
    lead_time_steps = 3
    x_lr = torch.randn(B, C_lr, H, W).to(device)
    x_hr = torch.randn(B, C_hr, H, W).to(device)
    lead_time_label = torch.randint(0, lead_time_steps, (B,)).to(device)

    # Compute loss with v1.0.1 checkpoint
    loss_fn_chkpt.y_mean = None
    for patch_num_per_iter in patch_nums_iter:
        patching.set_patch_num(patch_num_per_iter)
        loss_chkpt = loss_fn_chkpt(
            diff_model_chkpt,
            x_hr,
            x_lr,
            patching=patching,
            lead_time_label=lead_time_label,
            use_patch_grad_acc=True,
        )
        validate_accuracy(
            loss_chkpt,
            file_name=f"lt_aware_patched_diffusion_loss_iter_{patch_num_per_iter}_v1.1.1.pt",
        )
        loss_chkpt = loss_chkpt.sum() / B
        loss_chkpt.backward()

    # Compute loss with current model
    loss_fn.y_mean = None
    for patch_num_per_iter in patch_nums_iter:
        patching.set_patch_num(patch_num_per_iter)
        loss = loss_fn(
            diff_model,
            x_hr,
            x_lr,
            patching=patching,
            lead_time_label=lead_time_label,
            use_patch_grad_acc=True,
        )
        validate_accuracy(
            loss,
            file_name=f"lt_aware_patched_diffusion_loss_iter_{patch_num_per_iter}_v1.1.1.pt",
        )
        loss = loss.sum() / B
        loss.backward()

    return
