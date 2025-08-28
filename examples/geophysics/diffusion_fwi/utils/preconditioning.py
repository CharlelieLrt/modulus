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

from typing import Callable, Dict, Any, Union

import torch


def edm_precond(
    model: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    x: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    sigma_data: float = 0.5,
    *model_args: Any,
    **model_kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for EDM preconditioning.

    See :class:`physicsnemo.models.diffusion.BaseEDMPrecond` for details.

    Parameters
    ----------
    model : Union[torch.nn.Module, Callable[..., torch.Tensor]]
        Diffusion model to be wrapped with EDM preconditioning.
    x : torch.Tensor
        Latent state vector of shape (B, *).
    t : torch.Tensor
        Diffusion time, used to compute the noise level sigma. Should be of
        shape (B,).
    cond : Dict[str, torch.Tensor]
        Dictionary of conditioning information for the model.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    *model_args : Tuple, optional
        Additional positional arguments to pass to the wrapped model.
    **model_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the wrapped model.

    Returns
    -------
    torch.Tensor
        The output tensor, with the same shape (B, *) as the input tensor x.
    """
    sigma = t

    # Compute conditioning parameters
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma_data**2 + sigma**2).sqrt()
    c_noise = sigma.log() / 4

    # Apply conditioning to input
    x_precond = c_in * x

    # Call model with conditioned input
    F_x = model(
        x_precond,
        c_noise.flatten(),
        cond,
        *model_args,
        **model_kwargs,
    )

    # Apply output conditioning
    D_x = c_skip * x + c_out * F_x.to(torch.float32)

    return D_x
