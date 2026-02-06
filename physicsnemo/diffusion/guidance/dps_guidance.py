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

"""DPS (Diffusion Posterior Sampling) guidance for diffusion models."""

from typing import Callable, Protocol, Sequence, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import DiffusionDenoiser


@runtime_checkable
class DPSGuidance(Protocol):
    r"""
    Protocol defining the interface for Diffusion Posterior Sampling (DPS)
    guidance.

    A DPS guidance is a callable that computes a guidance term to steer the
    diffusion sampling process toward satisfying some observation constraint.
    A DPSGuidance is expected to be a score-predictor, as it returns a quantity
    analogous to a score.

    The typical form is:

    .. math::
        \gamma(t) \nabla_{\mathbf{x}}
        \ell(A(\hat{\mathbf{x}}_0) - \mathbf{y})

    where :math:`\gamma(t)` is a time-dependent guidance strength,
    :math:`A` is a (potentially nonlinear) observation operator,
    :math:`\mathbf{y}` is the observed data, and :math:`\ell` is a scalar loss
    function. However, variants are possible as long as the guidance produces
    a quantity similar to a score (e.g., a likelihood score).

    This is the minimal interface for guidance, and any object that implements
    this interface can be used with diffusion utilities such as
    :class:`DPSDenoiser` or
    :meth:`~physicsnemo.diffusion.noise_schedulers.get_denoiser`.

    See Also
    --------
    :class:`DPSDenoiser` : Combines a denoiser with one or more guidances.

    Examples
    --------
    **Example 1:** Minimal guidance for inpainting. Given a binary mask and
    observed pixels, guide the diffusion to match observations:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSGuidance
    >>>
    >>> class InpaintingGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask  # Binary mask: 1 = observed, 0 = missing
    ...         self.y_obs = y_obs  # Observed pixel values
    ...         self.gamma = gamma
    ...
    ...     def __call__(self, x, t, x_0):
    ...         # Compute residual at observed locations
    ...         residual = self.mask * (x_0 - self.y_obs)
    ...         # Gradient of L2 loss w.r.t. x_0 is just the residual
    ...         # (simplified: assumes identity observation operator)
    ...         return -self.gamma * residual
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = InpaintingGuidance(mask, y_obs)
    >>> isinstance(guidance, DPSGuidance)
    True

    **Example 2:** Building a guided denoiser from scratch. A common pattern
    is to combine an x0-predictor with a guidance to create a score predictor
    that can be used for sampling. This shows the complete workflow:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSGuidance
    >>>
    >>> # Define a guidance that pushes toward observed values
    >>> class MyGuidance:
    ...     def __init__(self, y_obs, gamma=0.1):
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * (x_0 - self.y_obs)
    ...
    >>> # Toy x0-predictor (in practice, a trained neural network)
    >>> x0_predictor = lambda x, t: x * 0.9
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = MyGuidance(y_obs, gamma=0.5)
    >>>
    >>> # Build a guided denoiser that combines x0-predictor + guidance
    >>> def guided_denoiser(x, t):
    ...     # Step 1: Get x0 estimate
    ...     x_0 = x0_predictor(x, t)
    ...     # Step 2: Compute guidance term
    ...     guidance_term = guidance(x, t, x_0)
    ...     # Step 3: Convert x0 to score (for EDM: score = (x_0 - x) / t^2)
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     score = (x_0 - x) / (t_bc ** 2)
    ...     # Step 4: Sum and return
    ...     return score + guidance_term
    ...
    >>> # guided_denoiser is now a DiffusionDenoiser (score predictor),
    >>> # and can be used with any sampling utility that expects this interface
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> output = guided_denoiser(x, t)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    Note: :class:`DPSDenoiser` provides a convenient way to apply one or more
    guidances to a denoiser without manually implementing the above pattern.
    """

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
            Typically used to compute gradients when the guidance requires
            backpropagation through the diffusion process, in which case it
            needs to have ``requires_grad=True``.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state, of shape :math:`(B, *)`.
            Typically produced by an x0-predictor or clean data predictor.

        Returns
        -------
        Tensor
            Guidance term of the same shape as ``x``. This is analogous to a
            likelihood score and is typically added to the unconditional score
            to guide the sampling process.
        """
        ...


class DPSDenoiser(DiffusionDenoiser):
    r"""
    Denoiser that combines an x0-predictor with DPS-style guidance.

    This class transforms a :class:`~physicsnemo.diffusion.DiffusionDenoiser`
    (specifically, an **x0-predictor**) into another
    :class:`~physicsnemo.diffusion.DiffusionDenoiser` (a **score predictor**)
    by applying one or more DPS guidances. The resulting denoiser can be used
    directly with ODE/SDE solvers and sampling utilities.

    The output is the sum of the unconditional score (derived from the
    x0-prediction) and all guidance terms:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{x})
        + \sum_i g_i(\mathbf{x}, t, \hat{\mathbf{x}}_0)

    where :math:`g_i` are the guidance terms implementing the
    :class:`DPSGuidance` interface.

    Each guidance must implement the :class:`DPSGuidance` protocol, which is a
    callable with the following signature:

    .. code-block:: python

        def guidance(x: Tensor, t: Tensor, x_0: Tensor) -> Tensor:
            # x: noisy latent state at time t, shape (B, *)
            # t: diffusion time, shape (B,)
            # x_0: estimated clean state, shape (B, *)
            # returns: guidance term, shape (B, *)
            ...

    Parameters
    ----------
    denoiser_in : DiffusionDenoiser
        Input denoiser that takes ``(x, t)`` and returns an estimate of the
        clean data :math:`\hat{\mathbf{x}}_0`. This is typically an x0-predictor
        obtained from a trained diffusion model.
    x0_to_score_fn : Callable[[Tensor, Tensor, Tensor], Tensor]
        Callback to convert x0-prediction to score. Signature:
        ``x0_to_score_fn(x_0, x, t) -> score``. Typically obtained from a noise
        scheduler, e.g.,
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_score`.
    guidances : DPSGuidance | Sequence[DPSGuidance]
        One or more guidance objects implementing the :class:`DPSGuidance`
        interface.

    See Also
    --------
    :class:`DPSGuidance` : Protocol for guidance implementations.
    :func:`~physicsnemo.diffusion.samplers.sample` : Sampling function that
        uses denoisers.

    Examples
    --------
    **Example 1:** Basic usage with a single guidance for inpainting:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSDenoiser, DPSGuidance
    >>>
    >>> # Toy x0-predictor (in practice, this is a trained neural network)
    >>> x0_predictor = lambda x, t: x * 0.9
    >>>
    >>> # Simple x0_to_score function (for EDM: score = (x_0 - x) / t^2)
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
    ...     return (x_0 - x) / (t_bc ** 2)
    ...
    >>> # Simple inpainting guidance
    >>> class InpaintGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * self.mask * (x_0 - self.y_obs)
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance = InpaintGuidance(mask, y_obs)
    >>>
    >>> # Create DPS denoiser
    >>> dps_denoiser = DPSDenoiser(
    ...     denoiser_in=x0_predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>>
    >>> # Use in sampling
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> output = dps_denoiser(x, t)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** Multiple guidances for multi-constraint problems:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DPSDenoiser
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> # Use scheduler to get x0_to_score_fn
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_predictor = lambda x, t: x * 0.9
    >>>
    >>> # Guidance 1: match observed values at specific locations
    >>> class ObservationGuidance:
    ...     def __init__(self, mask, y_obs, gamma=1.0):
    ...         self.mask = mask
    ...         self.y_obs = y_obs
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * self.mask * (x_0 - self.y_obs)
    ...
    >>> # Guidance 2: regularization toward zero mean
    >>> class ZeroMeanGuidance:
    ...     def __init__(self, gamma=0.1):
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0):
    ...         return -self.gamma * x_0.mean() * torch.ones_like(x_0)
    ...
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>> guidance1 = ObservationGuidance(mask, y_obs)
    >>> guidance2 = ZeroMeanGuidance()
    >>>
    >>> # Combine multiple guidances
    >>> dps_denoiser = DPSDenoiser(
    ...     denoiser_in=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=[guidance1, guidance2],
    ... )
    >>>
    >>> x = torch.randn(2, 3, 8, 8)
    >>> t = torch.tensor([1.0, 1.0])
    >>> output = dps_denoiser(x, t)
    >>> output.shape
    torch.Size([2, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser_in: DiffusionDenoiser,
        x0_to_score_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B *dims"], Float[Tensor, " B"]],
            Float[Tensor, " B *dims"],
        ],
        guidances: DPSGuidance | Sequence[DPSGuidance],
    ) -> None:
        self.denoiser_in = denoiser_in
        self.x0_to_score_fn = x0_to_score_fn
        # Normalize guidances to a list
        if isinstance(guidances, Sequence) and not isinstance(guidances, str):
            self.guidances = list(guidances)
        else:
            self.guidances = [guidances]

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the guided score for sampling.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Guided score of shape :math:`(B, *)`, computed as the sum of the
            unconditional score and all guidance terms.
        """
        x = x.detach().clone().requires_grad_(True)
        x_0 = self.denoiser_in(x, t)

        guidance_sum = torch.zeros_like(x)
        for guidance in self.guidances:
            guidance_sum += guidance(x, t, x_0)

        score = self.x0_to_score_fn(x_0, x, t)
        return score + guidance_sum
