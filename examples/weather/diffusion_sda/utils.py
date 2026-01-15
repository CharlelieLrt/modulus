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

from physicsnemo.diffusion.multi_diffusion import RandomPatching2D


# TODO: use version from diffusion package once refactor is complete
class EDMPreconditioner(Module):
    """
    Diffusion preconditioner wrapper based on EDM formulation.
    """

    def __init__(self, model: Module, sigma_data: float):
        super().__init__(meta=ModelMetaData())
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Dict[str, Tensor],
        **model_kwargs: Any,
    ):
        if not torch.compiler.is_compiling():
            B = x.shape[0]
            if t.shape != (B,):
                raise ValueError(
                    f"Expected t to be of shape (B,), but got {t.shape} instead."
                    f"Expected tensor of shape (B, D) but got tensor of shape {x.shape}"
                )
            if not all(v.shape[0] == B for v in condition.values()):
                raise ValueError(
                    f"Expected all condition values to be of shape (B, *), "
                    f"but got {condition.values()} instead."
                )

        sigma = t.view(x.shape[0], *((1,) * (x.ndim - 1)))

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = (sigma.log() / 4).flatten()

        return c_skip * x + c_out * self.model(
            c_in * x, c_noise, condition, **model_kwargs
        )



# TODO: use version from diffusion package once refactor is complete
class EDMLoss:
    """
    EDM loss function for conditional and unconditional multi-diffusion.

    Parameters
    ----------
    model : Callable[[Tensor, Tensor, Dict[str, Tensor], *Any], Tensor]
        The model to compute the loss for.
    P_mean : float, optional
        Mean value for the noise level.
    P_std : float, optional
        Standard deviation for the noise level.
    sigma_data : float, optional
        Standard deviation for the data.
    """

    def __init__(
        self,
        model: Callable[[Tensor, Tensor, Dict[str, Tensor], Any], Tensor],
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        patching: Optional[RandomPatching2D] = None,
    ):
        self.model = model
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.patching = patching

    def get_noise_params(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Method to compute the noise parameters for the EDM loss.

        Parameters
        ----------
        x : torch.Tensor
            Latent state of shape :math:`(B, *)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B,)`.
            - Weight ``weight`` of shape :math:`(B,` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([x.shape[0]], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(x) * sigma.view(x.shape[0], *((1,) * (x.ndim - 1)))
        return n, sigma, weight

    def __call__(
        self,
        x: Tensor,  # (B, C, H, W)
        condition: Dict[str, Tensor],
        **model_kwargs: Any,
    ) -> Tensor:
        """
        Calculate and return the loss corresponding to the EDM formulation.
        """
        if self.patching:
            x = self.patching.apply(x)
            for cond_name, y_cond in condition.items():
                condition[cond_name] = self.patching.apply(input=y_cond)
        
        global_index = self.patching.global_index(x.shape[0], x.device)


        # Compute noise parameters
        n, sigma, weight = self.get_noise_params(x)

        x_0 = self.model(
            x + n,
            sigma,
            condition,
            global_index=global_index,
            **model_kwargs,
        )
        return weight.view(x.shape[0], *((1,) * (x.ndim - 1))) * ((x_0 - x) ** 2)
