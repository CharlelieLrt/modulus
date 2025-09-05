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

from typing import Callable, Dict, Any, TypeAlias

import torch
from torch import Tensor
from torch.func import grad, vmap


class ModelBasedGuidance:
    r""" """

    # TODO: for each one of the scaling parameters, need explanations
    # + reference + make sure default values are sensible
    def __init__(
        self,
        guide_model: Callable[[torch.Tensor], torch.Tensor],
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 1,
    ):
        self.guide_model = torch.func.vmap(guide_model)
        self.std = std
        self.gamma = gamma
        self.mu = mu
        self.scale = scale
        self.power = power
        self.norm_ord = norm_ord

    def _log_likelihood(
        self,
        x_0_hat: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Compute L1 error between model prediction and observation
        # NOTE: for now only Tweedie's formula to estimate clean state x_0
        y_x0: torch.Tensor = self.guide_model(x_0_hat)  # (*_y,)
        if y_x0.shape != y.shape:
            raise ValueError(
                f"Expected 'guide_model' output and y to have same shape, "
                f"but got {y_x0.shape} and {y.shape}"
            )
        err1 = torch.abs((y - y_x0)) ** self.norm_ord  # (*_y,)

        # Compute log-likelihood p(y|x_0_hat)
        var = self.std**2 + self.gamma * (t / self.mu) ** 2  # (,)
        log_p = -0.5 * (err1 / var).sum()  # (,)
        return log_p

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        ndim = x.ndim

        # Parameters validation
        if t.shape != (B,):
            raise ValueError(f"Expected t to have shape {(B,)}, but got {t.shape}")
        if y.shape[0] != B:
            raise ValueError(f"Expected y to have batch size {B}, but got {y.shape[0]}")
        if x_0_hat.shape != x.shape:
            raise ValueError(
                f"Expected x_0_hat and x to have same shape, "
                f"but got {x_0_hat.shape} and {x.shape}"
            )

        # NOTE: tensor is detached without requires_grad to save memory
        # (not required with torch.func anyways)
        x_0_hat = x_0_hat.clone().detach().requires_grad_(False)  # (*_x,)

        # Compute likelihood score
        score = torch.func.vmap(
            torch.func.grad(
                self._log_likelihood,
                argnums=0,
            )
        )(x_0_hat, y, t)  # (B, *_x,)

        # Scale the likelihood score
        scale = torch.where(t < 1, self.scale * t.pow(self.power), self.scale).view(
            B, *([1] * (ndim - 1))
        )  # (B, 1, ..., 1)
        score_mag = torch.abs(score).mean(
            dim=tuple(range(1, ndim)), keepdim=True
        )  # (B, 1, ..., 1)
        score_scaled = (
            score * scale * t.view(B, *([1] * (ndim - 1))) / (1 + score_mag)
        )  # (B, *_x)

        return score_scaled

