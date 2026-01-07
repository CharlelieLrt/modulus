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

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


class BasePreconditioner(Module, ABC):
    r"""
    Abstract base class for diffusion model preconditioners.

    This class provides a standardized interface for implementing
    preconditioners used in diffusion models.

    The preconditioner wraps a neural network model :math:`F` and applies
    a preconditioning formula to transform the network output to produce
    the denoised output :math:`D(\mathbf{x}, t)` according to:

    .. math::

        D(\mathbf{x}, t) = c_{\text{skip}}(t) \mathbf{x} +
        c_{\text{out}}(t) F(c_{\text{in}}(t) \mathbf{x}, c_{\text{noise}}(t))

    where:

    - :math:`c_{\text{in}}(t)`: Input scaling coefficient
    - :math:`c_{\text{noise}}(t)`: Noise conditioning value
    - :math:`c_{\text{out}}(t)`: Output scaling coefficient
    - :math:`c_{\text{skip}}(t)`: Skip connection scaling coefficient

    The wrapped model :math:`F` must have the following signature:

    .. code-block::

        model(
            x: torch.Tensor,
            t: torch.Tensor,
            condition: Dict[str, torch.Tensor],
            **model_kwargs: Any,
        ) -> torch.Tensor

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model :math:`F` to wrap with the
        signature described above.
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None.
        Subclasses can pass their own metadata.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time step tensor of shape :math:`(B,)`.
    condition : Dict[str, torch.Tensor]
        Dictionary of conditioning tensors. Each tensor must have shape
        :math:`(B, *)` where the batch size :math:`B` matches that of ``x``.
        These are passed to the wrapped ``model`` without modification.
    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Denoised latent state with the same shape :math:`(B, *)` as the input
        tensor ``x``.

    .. note::

        - Subclasses must implement the :meth:`compute_coefficients` method to
          define the specific preconditioning scheme.

        - If a subclass implements the :meth:`sigma` method, the diffusion
          time step :math:`t` is first transformed to a noise level
          :math:`\sigma(t)` before being passed to
          :meth:`compute_coefficients`. This allows implementing
          preconditioners for different noise schedules while keeping the
          same preconditioning interface.

    Examples
    --------
    The following example shows how to implement a classical EDM
    preconditioner. For EDM, there is no need to override the
    :meth:`sigma` method since :math:`\sigma(t) = t` (noise level and
    diffusion time step are the same).

    We first define a simple model to wrap (for illustration purposes):

    >>> import torch
    >>> from physicsnemo.nn import Module
    >>> class SimpleModel(Module):
    ...     '''A simple model for illustration purposes.'''
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.channels = channels
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...
    ...     def forward(self, x, t, condition, **kwargs):
    ...         return self.net(x)

    Now we define the EDM preconditioner:

    >>> from physicsnemo.diffusion.preconditioners import BasePreconditioner
    >>> class EDMPreconditioner(BasePreconditioner):
    ...     def __init__(self, model, sigma_data: float = 0.5):
    ...         super().__init__(model)
    ...         self.sigma_data = sigma_data
    ...
    ...     def compute_coefficients(self, t: torch.Tensor):
    ...         sigma_data = self.sigma_data
    ...         c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    ...         c_out = t * sigma_data / (t**2 + sigma_data**2).sqrt()
    ...         c_in = 1 / (sigma_data**2 + t**2).sqrt()
    ...         c_noise = t.log() / 4
    ...         return c_in, c_noise, c_out, c_skip
    ...
    >>> model = SimpleModel(channels=3)
    >>> precond = EDMPreconditioner(model, sigma_data=0.5)
    >>> x = torch.randn(2, 3, 16, 16)
    >>> t = torch.rand(2)
    >>> out = precond(x, t, {})
    >>> out.shape
    torch.Size([2, 3, 16, 16])

    The following example shows how to override the :meth:`sigma` method to
    implement a Variance Exploding (VE) preconditioner where
    :math:`\sigma(t) = \sqrt{t}`.

    >>> class VEPreconditioner(BasePreconditioner):
    ...     def __init__(self, model):
    ...         super().__init__(model)
    ...
    ...     def sigma(self, t: torch.Tensor) -> torch.Tensor:
    ...         # Override sigma to implement VE noise schedule
    ...         return t.sqrt()
    ...
    ...     def compute_coefficients(self, sigma: torch.Tensor):
    ...         # Here the argument passed to compute_coefficients is already sigma(t) = sqrt(t)
    ...         # due to override of the sigma method
    ...         c_skip = torch.ones_like(sigma)
    ...         c_out = sigma
    ...         c_in = torch.ones_like(sigma)
    ...         c_noise = (0.5 * sigma).log()
    ...         return c_in, c_noise, c_out, c_skip
    ...
    >>> precond_ve = VEPreconditioner(model)
    >>> out_ve = precond_ve(x, t, condition={})
    >>> out_ve.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        model: Module,
        meta: ModelMetaData | None = None,
    ) -> None:
        super().__init__(meta=meta)
        self.model = model

    @abstractmethod
    def compute_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute the preconditioning coefficients for a given time step.

        This abstract method must be implemented by subclasses to define
        the specific preconditioning scheme.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time step (or noise level if :meth:`sigma` is
            overridden) tensor of shape :math:`(B, 1, ..., 1)` where
            :math:`B` is the batch size and the trailing singleton
            dimensions match the spatial dimensions of the latent state
            ``x`` for broadcasting. If the subclass defines the
            :meth:`sigma` method, then ``t`` contains :math:`\sigma(t)`.

        Returns
        -------
        c_in : torch.Tensor
            Input scaling coefficient of shape :math:`(B, 1, ..., 1)`.
        c_noise : torch.Tensor
            Noise conditioning value of shape :math:`(B, 1, ..., 1)`.
        c_out : torch.Tensor
            Output scaling coefficient of shape :math:`(B, 1, ..., 1)`.
        c_skip : torch.Tensor
            Skip connection scaling coefficient of shape
            :math:`(B, 1, ..., 1)`.
        """
        ...

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Map diffusion time step :math:`t` to noise level :math:`\sigma(t)`.

        By default, this is the identity function :math:`\sigma(t) = t`.
        Subclasses can override this to implement preconditioners for different
        noise schedules.

        When overridden, the output of this method is passed to
        :meth:`compute_coefficients` instead of the raw time step ``t``.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time step tensor of shape :math:`(B,)` where
            :math:`B` is the batch size.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(B,)`.
        """
        return t

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        **model_kwargs: Any,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            B = x.shape[0]
            if t.shape != (B,):
                raise ValueError(
                    f"Expected t to have shape ({B},) matching batch size of "
                    f"x, but got {t.shape}."
                )
            for k, v in condition.items():
                if v.shape[0] != B:
                    raise ValueError(
                        f"Condition tensor '{k}' has batch size {v.shape[0]} "
                        f"but expected {B} to match x."
                    )

        # Map time step to noise level via sigma method
        sigma_t = self.sigma(t).reshape(-1, *([1] * (x.ndim - 1)))

        # Compute preconditioning coefficients
        c_in, c_noise, c_out, c_skip = self.compute_coefficients(sigma_t)

        # Forward through the underlying model
        F_x = self.model(
            c_in * x,
            c_noise.flatten(),
            condition,
            **model_kwargs,
        )

        D_x = c_skip * x + c_out * F_x

        return D_x
