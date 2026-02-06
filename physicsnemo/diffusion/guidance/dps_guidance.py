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
            Guided score of same shape :math:`(B, *)` as ``x``. Computed as the
            sum of the unconditional score and all guidance terms.
        """
        x = x.detach().clone().requires_grad_(True)
        x_0 = self.denoiser_in(x, t)

        guidance_sum = torch.zeros_like(x)
        for guidance in self.guidances:
            guidance_sum += guidance(x, t, x_0)

        score = self.x0_to_score_fn(x_0, x, t)
        return score + guidance_sum


class ModelConsistencyDPSGuidance:
    r"""
    DPS guidance for generic observation models with Gaussian noise.

    Computes the likelihood score for an observation model of the form:

    .. math::
        \mathbf{y} = A(\mathbf{x}_0) + \boldsymbol{\epsilon}, \quad
        \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_y^2 \mathbf{I})

    where :math:`A` is a (potentially nonlinear) observation operator,
    :math:`\mathbf{y}` is the observed data, and :math:`\sigma_y` is the
    measurement noise standard deviation.

    The guidance term is the likelihood score:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{y} | \hat{\mathbf{x}}_0)
        = -\frac{1}{\sigma_y^2} \nabla_{\mathbf{x}}
        \| A(\hat{\mathbf{x}}_0) - \mathbf{y} \|_p^p

    where :math:`\| \cdot \|_p` is the :math:`L^p` norm and :math:`p` is the
    ``norm_order``. This is computed via automatic differentiation.

    An optional **SDA (Score-Based Data Assimilation) scaling** can be applied,
    which scales the guidance by :math:`\sigma(t)^2` to properly weight the
    likelihood relative to the prior at different noise levels:

    .. math::
        \text{guidance} = \sigma(t)^2 \cdot \nabla_{\mathbf{x}}
        \log p(\mathbf{y} | \hat{\mathbf{x}}_0)

    The observation operator ``A`` must be a differentiable callable with the
    following signature:

    .. code-block:: python

        def A(x_0: Float[Tensor, "B *dims"]) -> Float[Tensor, "B *obs_dims"]:
            # x_0: estimated clean state, shape (B, *)
            # returns: predicted observations, shape (B, *obs_dims)
            ...

    Parameters
    ----------
    A : Callable[[Tensor], Tensor]
        Observation operator mapping clean state to observations.
        Must be differentiable (supports ``torch.autograd``).
    y : Tensor
        Observed data of shape :math:`(B, *obs\_dims)` matching the output
        of ``A``.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm_order : int, default=2
        Order of the norm used to compute the residual. Use ``2`` for
        standard Gaussian likelihood (L2 norm), ``1`` for L1 norm, etc.
    sda_scaling : bool, default=False
        If ``True``, applies SDA scaling by multiplying the guidance by
        :math:`\sigma(t)^2`. Requires ``sigma_fn`` to be provided.
    sigma_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``sda_scaling=True``. Typically obtained from a noise
        scheduler, e.g.,
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.

    See Also
    --------
    :class:`DataConsistencyDPSGuidance` : Simplified guidance for masked
        observations.
    :class:`DPSDenoiser` : Combines a denoiser with one or more guidances.

    Examples
    --------
    **Example 1:** Guidance for a downsampling observation operator:

    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion.guidance import ModelConsistencyDPSGuidance
    >>>
    >>> # Observation operator: 2x downsampling
    >>> def downsample_2x(x):
    ...     return F.avg_pool2d(x, kernel_size=2, stride=2)
    ...
    >>> # Low-resolution observations
    >>> y_obs = torch.randn(1, 3, 4, 4)  # 4x4 from 8x8 original
    >>>
    >>> guidance = ModelConsistencyDPSGuidance(
    ...     A=downsample_2x,
    ...     y=y_obs,
    ...     std_y=0.1,
    ... )
    >>>
    >>> # Use in DPS sampling
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9  # Toy x0 estimate
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** With SDA scaling for improved assimilation:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import ModelConsistencyDPSGuidance
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # Simple linear observation operator (select first channel)
    >>> A = lambda x: x[:, :1]
    >>> y_obs = torch.randn(1, 1, 8, 8)
    >>>
    >>> guidance = ModelConsistencyDPSGuidance(
    ...     A=A,
    ...     y=y_obs,
    ...     std_y=0.05,
    ...     sda_scaling=True,
    ...     sigma_fn=scheduler.sigma,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        A: Callable[[Float[Tensor, " B *dims"]], Float[Tensor, " B *obs_dims"]],
        y: Float[Tensor, " B *obs_dims"],
        std_y: float,
        norm_order: int = 2,
        sda_scaling: bool = False,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
    ) -> None:
        if sda_scaling and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when sda_scaling=True")
        self.A = A
        self.y = y
        self.std_y = std_y
        self.norm_order = norm_order
        self.sda_scaling = sda_scaling
        self.sigma_fn = sigma_fn

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
            Must have ``requires_grad=True`` for gradient computation.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state, of shape :math:`(B, *)`.

        Returns
        -------
        Tensor
            Likelihood score guidance term of same shape as ``x``.
        """
        # Ensure x_0 has gradients for autograd
        x_0_grad = x_0.detach().requires_grad_(True)

        # Compute predicted observations and residual
        y_pred = self.A(x_0_grad)
        residual = y_pred - self.y

        # Compute norm^p of residual (summed over all dims except batch)
        residual_flat = residual.reshape(residual.shape[0], -1)
        norm_p = residual_flat.abs().pow(self.norm_order).sum(dim=1)

        # Compute gradient of norm w.r.t. x_0
        grad_x0 = torch.autograd.grad(
            outputs=norm_p.sum(),
            inputs=x_0_grad,
            create_graph=False,
        )[0]

        # Likelihood score: -1/std_y^2 * grad
        guidance = -grad_x0 / (self.std_y**2)

        # Apply SDA scaling if enabled
        if self.sda_scaling and self.sigma_fn is not None:
            t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
            sigma_t_sq = self.sigma_fn(t_bc) ** 2
            guidance = sigma_t_sq * guidance

        return guidance


class DataConsistencyDPSGuidance:
    r"""
    DPS guidance for masked observations with Gaussian noise.

    A simplified version of :class:`ModelConsistencyDPSGuidance` where the
    observation operator is a mask applied element-wise. This is typical for
    data assimilation tasks like inpainting or outpainting, where observations
    are available at specific locations.

    The observation model is:

    .. math::
        \mathbf{y} = \mathbf{M} \odot \mathbf{x}_0 + \boldsymbol{\epsilon},
        \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_y^2 \mathbf{I})

    where :math:`\mathbf{M}` is a binary mask (1 = observed, 0 = missing),
    :math:`\odot` denotes element-wise multiplication, and :math:`\sigma_y`
    is the measurement noise standard deviation.

    The guidance term is the likelihood score:

    .. math::
        \nabla_{\mathbf{x}} \log p(\mathbf{y} | \hat{\mathbf{x}}_0)
        = -\frac{1}{\sigma_y^2} \nabla_{\mathbf{x}}
        \| \mathbf{M} \odot (\hat{\mathbf{x}}_0 - \mathbf{y}) \|_p^p

    An optional **SDA (Score-Based Data Assimilation) scaling** can be applied,
    which scales the guidance by :math:`\sigma(t)^2`.

    Parameters
    ----------
    mask : Tensor
        Binary mask of shape :math:`(B, *)` or broadcastable shape.
        Values should be 1 for observed locations and 0 for missing.
    y : Tensor
        Observed data of shape :math:`(B, *)` matching the state shape.
        Values at unobserved locations (where ``mask=0``) are ignored.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm_order : int, default=2
        Order of the norm used to compute the residual. Use ``2`` for
        standard Gaussian likelihood (L2 norm), ``1`` for L1 norm, etc.
    sda_scaling : bool, default=False
        If ``True``, applies SDA scaling by multiplying the guidance by
        :math:`\sigma(t)^2`. Requires ``sigma_fn`` to be provided.
    sigma_fn : Callable[[Tensor], Tensor] | None, default=None
        Function mapping diffusion time to noise level :math:`\sigma(t)`.
        Required when ``sda_scaling=True``.

    See Also
    --------
    :class:`ModelConsistencyDPSGuidance` : Guidance for general observation
        operators.
    :class:`DPSDenoiser` : Combines a denoiser with one or more guidances.

    Examples
    --------
    **Example 1:** Inpainting with known pixels at specific locations:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DataConsistencyDPSGuidance
    >>>
    >>> # Mask: observe 50% of pixels randomly
    >>> mask = (torch.rand(1, 3, 8, 8) > 0.5).float()
    >>> y_obs = torch.randn(1, 3, 8, 8)  # Observed values
    >>>
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.1,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9  # Toy x0 estimate
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    **Example 2:** With SDA scaling and L1 norm for robustness:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import DataConsistencyDPSGuidance
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>>
    >>> # Observe boundary pixels only (outpainting scenario)
    >>> mask = torch.zeros(1, 3, 8, 8)
    >>> mask[:, :, 0, :] = 1  # Top row
    >>> mask[:, :, -1, :] = 1  # Bottom row
    >>> mask[:, :, :, 0] = 1  # Left column
    >>> mask[:, :, :, -1] = 1  # Right column
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>>
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.05,
    ...     norm_order=1,  # L1 norm for robustness to outliers
    ...     sda_scaling=True,
    ...     sigma_fn=scheduler.sigma,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8, requires_grad=True)
    >>> t = torch.tensor([1.0])
    >>> x_0 = x * 0.9
    >>> output = guidance(x, t, x_0)
    >>> output.shape
    torch.Size([1, 3, 8, 8])

    **Example 3:** Using with DPSDenoiser for complete sampling:

    >>> import torch
    >>> from physicsnemo.diffusion.guidance import (
    ...     DataConsistencyDPSGuidance,
    ...     DPSDenoiser,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_predictor = lambda x, t: x * 0.9  # Toy x0-predictor
    >>>
    >>> mask = torch.ones(1, 3, 8, 8)
    >>> y_obs = torch.randn(1, 3, 8, 8)
    >>>
    >>> guidance = DataConsistencyDPSGuidance(
    ...     mask=mask,
    ...     y=y_obs,
    ...     std_y=0.1,
    ... )
    >>>
    >>> dps_denoiser = DPSDenoiser(
    ...     denoiser_in=x0_predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance,
    ... )
    >>>
    >>> x = torch.randn(1, 3, 8, 8)
    >>> t = torch.tensor([1.0])
    >>> output = dps_denoiser(x, t)
    >>> output.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        mask: Float[Tensor, " *mask_shape"],
        y: Float[Tensor, " B *dims"],
        std_y: float,
        norm_order: int = 2,
        sda_scaling: bool = False,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
    ) -> None:
        if sda_scaling and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when sda_scaling=True")
        self.mask = mask
        self.y = y
        self.std_y = std_y
        self.norm_order = norm_order
        self.sda_scaling = sda_scaling
        self.sigma_fn = sigma_fn

    def __call__(
        self,
        x: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        x_0: Float[Tensor, " B *dims"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Compute the likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy latent state at diffusion time ``t``, of shape :math:`(B, *)`.
            Must have ``requires_grad=True`` for gradient computation.
        t : Tensor
            Batched diffusion time of shape :math:`(B,)`.
        x_0 : Tensor
            Estimate of the clean latent state, of shape :math:`(B, *)`.

        Returns
        -------
        Tensor
            Likelihood score guidance term of same shape as ``x``.
        """
        # Ensure x_0 has gradients for autograd
        x_0_grad = x_0.detach().requires_grad_(True)

        # Compute masked residual
        residual = self.mask * (x_0_grad - self.y)

        # Compute norm^p of residual (summed over all dims except batch)
        residual_flat = residual.reshape(residual.shape[0], -1)
        norm_p = residual_flat.abs().pow(self.norm_order).sum(dim=1)

        # Compute gradient of norm w.r.t. x_0
        grad_x0 = torch.autograd.grad(
            outputs=norm_p.sum(),
            inputs=x_0_grad,
            create_graph=False,
        )[0]

        # Likelihood score: -1/std_y^2 * grad
        guidance = -grad_x0 / (self.std_y**2)

        # Apply SDA scaling if enabled
        if self.sda_scaling and self.sigma_fn is not None:
            t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))
            sigma_t_sq = self.sigma_fn(t_bc) ** 2
            guidance = sigma_t_sq * guidance

        return guidance
