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

"""Protocols and type hints for diffusion model interfaces."""

from typing import Any, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from tensordict import TensorDict


@runtime_checkable
class DiffusionModel(Protocol):
    r"""
    Protocol defining the common interface for diffusion models.

    A diffusion model is any neural network or function that transforms a noisy
    state ``x`` at diffusion time (or noise level) ``t`` into a prediction.
    This protocol defines the standard interface that all diffusion models must
    satisfy.

    Any model or function that implements this interface can be used with
    preconditioners, losses, samplers, and other diffusion utilities.

    The interface is **prediction-agnostic**: whether your model predicts
    clean data (:math:`\mathbf{x}_0`), noise (:math:`\epsilon`), score
    (:math:`\nabla \log p`), or velocity (:math:`\mathbf{v}`), the signature
    remains the same.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion import DiffusionModel
    >>>
    >>> class Denoiser:
    ...     def __call__(self, x, t, condition, **kwargs):
    ...         return F.relu(x)
    ...
    >>> isinstance(Denoiser(64), DiffusionModel)
    True
    """

    def __call__(
        self,
        x: Float[torch.Tensor, "B *dims"],  # noqa: F821
        t: Float[torch.Tensor, "B"],  # noqa: F821
        condition: TensorDict,
        **model_kwargs: Any,
    ) -> Float[torch.Tensor, "B *dims"]:  # noqa: F821
        r"""
        Forward pass of the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor
            Diffusion time or noise level tensor of shape :math:`(B,)`.
        condition : TensorDict
            TensorDict containing conditioning tensors. The TensorDict should
            have batch size :math:`B` matching that of ``x``. If the model is
            unconditional, the condition should be the empty ``TensorDict()``.
        **model_kwargs : Any
            Additional keyword arguments specific to the model implementation.

        Returns
        -------
        torch.Tensor
            Model output with the same shape as ``x``.
        """
        ...


@runtime_checkable
class DiffusionDenoiser(Protocol):
    r"""
    Protocol defining a denoiser interface for diffusion model inference.

    A denoiser is a callable that takes a noisy state ``x`` and a noise level
    (or diffusion time) ``t``, and returns a denoised prediction. This is the
    minimal interface required for sampling from a diffusion model.

    This protocol is typically used during inference with the
    :func:`~physicsnemo.diffusion.samplers.sample` function. For training,
    which often requires additional inputs like conditioning, use the more
    general :class:`DiffusionModel` protocol instead.

    .. note::

        A :class:`DiffusionDenoiser` can be obtained from a
        :class:`DiffusionModel` by partially applying the ``condition`` and
        any other keyword arguments. For example:

        .. code-block:: python

            from functools import partial
            denoiser = partial(model, condition=my_condition)

    See Also
    --------
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function
        that uses this denoiser interface.
    :class:`DiffusionModel` : The full diffusion model interface with
        conditioning support.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion import DiffusionDenoiser
    >>>
    >>> class SimpleDenoiser:
    ...     def __call__(self, x, t):
    ...         # A trivial denoiser that returns the input unchanged
    ...         return x
    ...
    >>> denoiser = SimpleDenoiser()
    >>> isinstance(denoiser, DiffusionDenoiser)
    True
    """

    def __call__(
        self,
        x: Float[torch.Tensor, "B *dims"],  # noqa: F821
        t: Float[torch.Tensor, "B"],  # noqa: F821
    ) -> Float[torch.Tensor, "B *dims"]:  # noqa: F821
        r"""
        Denoise the input at the given noise level.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor
            Diffusion time or noise level tensor of shape :math:`(B,)`.

        Returns
        -------
        torch.Tensor
            Denoised prediction with the same shape as ``x``.
        """
        ...
