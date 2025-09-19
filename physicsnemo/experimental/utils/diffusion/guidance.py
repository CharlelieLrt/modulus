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

from typing import Callable, Literal

import torch


class ModelBasedGuidance:
    r"""
    Guidance function for `Diffusion Posterior Sampling (DPS)
    <https://arxiv.org/abs/2209.14687>`_ based on a generic user-defined model
    (possibly non-linear).
    An instance of ``ModelBasedGuidance`` is a callable object that returns
    the likelihood score :math:`\nabla_{\mathbf{x}_t} \log
    p(\mathbf{y}|\mathbf{x}_t)`, where :math:`\mathbf{y}` is some conditioniong
    variable (observation) that can be predicted by a forward model
    :math:`\mathcal{M}` (called ``guide_model``). This guidance enforces
    consistency between the predicted observation
    :math:`\mathcal{M}(\hat{\mathbf{x}}_0 (\mathbf{x}_t))` and the
    observed data :math:`\mathbf{y}`, where :math:`\hat{\mathbf{x}}_0` is an
    estimate of the clean latent state :math:`\mathbf{x}_0`, usually obtained
    by Tweedie's formula.

    The likelihood score follows a `modified version
    <https://arxiv.org/abs/2506.22780>`_ of the implementation introduced in
    `Score-based data assimilation
    <https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html>`_.
    It is computed as:

    :math:`\dfrac{S}{1 + M} s_l( \mathbf{y} | \mathbf{x}_t)`, where :math:`S`
    and :math:`M` are scaling parameters and :math:`s_l( \mathbf{y} | \mathbf{x}_t)`
    is the likelihood score. Those are computed fined as:

    - :math:`M = |s_l( \mathbf{y} | \mathbf{x}_t)|` is the magnitude of the
      likelihood score.

    - :math:`S = \text{scale} t^{\text{power}}` if :math:`t < 1` else
      :math:`\text{scale}` is the scaled guidance strength.

    - :math:`s_l( \mathbf{y} | \mathbf{x}_t) = \nabla_{\mathbf{x}_t} \dfrac{1}{2} \log
      p(\mathbf{y}|\mathbf{x}_t) = \nabla_{\mathbf{x}_t} \dfrac{- || \mathbf{y}
      - \mathcal{M}(\hat{\mathbf{x}}_0) ||^{\text{ord}}}{2 (\Sigma_y + \Gamma (\sigma_t /
      \mu)^2)} \log p(\mathbf{y}|\hat{\mathbf{x}}_0 (\mathbf{x}_t))` is the
      likelihood score.

    A ``ModelBasedGuidance`` instance is expected to be the most useful when
    passed to a sampler such as the ``EDMStochasticSampler`` class.

    Parameters
    ----------
    guide_model: Callable[[torch.Tensor], torch.Tensor]
        The forward model :math:`\mathcal{M}` that predicts the observation :math:`\mathbf{y}` from
        the clean latent state :math:`\hat{\mathbf{x}}_0`. Should be a callable
        object that takes as input a single tensor ``x_0_hat`` and returns a
        single tensor ``y_x0`` that is the predicted observation.
    std: float, optional, default=0.075
        The standard deviation :math:`\Sigma_y` of the observation noise.
    gamma: float, optional, default=0.05
        The parameter :math:`\Gamma` of the sclaing, that depends on the
        covariance :math:`\Sigma_x` of the prior.
    mu: float, optional, default=1
        The parameter :math:`\mu` that scales the noise level :math:`\sigma_t`.
    scale: float, optional, default=1
        The :math:`\text{scale}` parameter used to compute the guidance
        strength :math:`S`.
    power: float, optional, default=1
        The :math:`\text{power}` parameter used to compute the guidance
        strength :math:`S`.
    norm_ord: float, optional, default=2
        The order of the norm used to compute the error in the likelihood
        score.
    magnitude_scaling: bool, optional, default=True
        Whether to divide the likelihood score by :math:`1 + M`, where
        :math:`M` is its magnitude.
    model_exec_mode: Literal["batched"], optional, default="batched"
        The execution mode of the ``guide_model``. For ``model_exec_mode="batched"``,
        the ``guide_model`` is expected to be a batched function that takes as input
        a tensor ``x_0_hat`` of shape :math:`(B, *_x)` and returns a tensor ``y_x0``
        of shape :math:`(B, *_y)`. The ``guide_model`` is also expected to be
        differentiable with ``torch.autograd.grad``.

    Forward
    -------
    x: torch.Tensor
        The latent state vector of the diffusion model. Should be of shape
        :math:`(B, *_x)`.
    x_0_hat: torch.Tensor
        The estimate of the clean latent state :math:`\hat{\mathbf{x}}_0`.
        Should be of shape :math:`(B, *_x)`.
    sigma: torch.Tensor
        The noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    y: torch.Tensor
        The observed data :math:`\mathbf{y}`. Should be of shape :math:`(B,
        *_y)`. When used with a sampler such as in instance of the
        ``EDMStochasticSampler`` class, this should be passed as a
        ``guidance_args`` argument.

    Outputs
    -------
    torch.Tensor
        The scaled likelihood score of shape :math:`(B, *_x)`.

    """

    def __init__(
        self,
        guide_model: Callable[[torch.Tensor], torch.Tensor],
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 2,
        magnitude_scaling: bool = True,
        # NOTE: for now only support model that processes batched inputs. Need
        # more execution modes (e.g. vmap-able, single sample, non-pytorch impl,
        # etc.)
        model_exec_mode: Literal["batched"] = "batched",
    ):
        self.guide_model = guide_model
        self.std = std
        self.gamma = gamma
        self.mu = mu
        self.scale = scale
        self.power = power
        self.norm_ord = norm_ord
        self.magnitude_scaling = magnitude_scaling
        _valid_model_exec_mode = ["batched"]
        if model_exec_mode in _valid_model_exec_mode:
            self.model_exec_mode = model_exec_mode
        else:
            raise ValueError(
                f"'model_exec_mode' should be one of {', '.join(_valid_model_exec_mode)}, "
                f"but got {model_exec_mode}"
            )

    def _log_likelihood(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper function to compute the log-likelihood.
        """
        # Compute L1 error between model prediction and observation
        # NOTE: for now only Tweedie's formula to estimate clean state x_0
        if self.model_exec_mode == "batched":
            B = y.shape[0]
            y_x0: torch.Tensor = self.guide_model(x_0_hat)  # (B, *_y)
        if y_x0.shape != y.shape:
            raise ValueError(
                f"Expected 'guide_model' output and y to have same shape, "
                f"but got {y_x0.shape} and {y.shape}"
            )
        err1 = torch.abs((y - y_x0)) ** self.norm_ord  # (B, *_y)

        # Compute log-likelihood p(y|x_0_hat)
        var = self.std**2 + self.gamma * (sigma / self.mu) ** 2  # (B,)
        log_p = -0.5 * (err1 / var.view(B, *([1] * (y.ndim - 1)))).sum(
            dim=tuple(range(1, y.ndim))
        )  # (B,)
        return log_p

    def _get_score(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper function to compute the likelihood score.
        """
        # NOTE: the sum + grad trick only woks with independent samples (i.e.
        # no cross-sample coupling)
        if self.model_exec_mode == "batched":
            log_p = self._log_likelihood(x, x_0_hat, sigma, y)
            dlog_p_dx = torch.autograd.grad(
                outputs=log_p.sum(),  # (,)
                inputs=x,  # (B, *_x)
            )[0]  # (B, *_x)
        return dlog_p_dx

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        ndim = x.ndim

        # Parameters validation
        if sigma.shape != (B,):
            raise ValueError(
                f"Expected sigma to have shape {(B,)}, but got {sigma.shape}"
            )
        if y.shape[0] != B:
            raise ValueError(f"Expected y to have batch size {B}, but got {y.shape[0]}")
        if x_0_hat.shape != x.shape:
            raise ValueError(
                f"Expected x_0_hat and x to have same shape, "
                f"but got {x_0_hat.shape} and {x.shape}"
            )

        # Compute likelihood score
        score = self._get_score(x, x_0_hat, sigma, y)  # (B, *_x)

        # Scale the likelihood score
        scale = torch.where(
            sigma < 1, self.scale * sigma.pow(self.power), self.scale
        ).view(B, *([1] * (ndim - 1)))  # (B, 1, ..., 1)
        if self.magnitude_scaling:
            score_mag = torch.abs(score).mean(
                dim=tuple(range(1, ndim)), keepdim=True
            )  # (B, 1, ..., 1)
        else:
            score_mag = 0
        score_scaled = (
            score * scale * sigma.view(B, *([1] * (ndim - 1))) / (1 + score_mag)
        )  # (B, *_x)

        return score_scaled


class DataConsistencyGuidance(ModelBasedGuidance):
    r"""
    ``DataConsistencyGuidance`` is a specific type of ``ModelBasedGuidance``
    where the model :math:`\mathcal{M}` used in the guidance is a (linear)
    measurement operator, defined by a relation of the form :math:`\mathcal{M}
    (\hat{\mathbf{x}}_0) = \text{Mask} (\mathbf{x}_0)`, where :math:`\text{Mask}`
    is a mask operator that selects a subset of clean latent state
    :math:`\hat{\mathbf{x}}_0`.

    This guidance is useful for applications such as image inpainting, channel
    infilling, etc.

    It returns the scaled likelihood score :math:`\nabla_{\mathbf{x}_t} \log
    p(\mathbf{y}|\mathbf{x}_t)`, in a similar way as ``ModelBasedGuidance``.
    Most of the parameters are the same as ``ModelBasedGuidance``.

    Parameters
    ----------
    std: float, optional, default=0.075
        The standard deviation :math:`\Sigma_y` of the observation noise.
    gamma: float, optional, default=0.05
        The parameter :math:`\Gamma` of the sclaing, that depends on the
        covariance :math:`\Sigma_x` of the prior.
    mu: float, optional, default=1
        The parameter :math:`\mu` that scales the noise level :math:`\sigma_t`.
    scale: float, optional, default=1
        The :math:`\text{scale}` parameter used to compute the guidance
        strength :math:`S`.
    power: float, optional, default=1
        The :math:`\text{power}` parameter used to compute the guidance
        strength :math:`S`.
    norm_ord: float, optional, default=2
        The order of the norm used to compute the error in the likelihood
        score.
    magnitude_scaling: bool, optional, default=True
        Whether to divide the likelihood score by :math:`1 + M`, where
        :math:`M` is its magnitude.

    Forward
    -------
    x: torch.Tensor
        The latent state vector of the diffusion model. Should be of shape
        :math:`(B, *_x)`.
    x_0_hat: torch.Tensor
        The estimate of the clean latent state :math:`\hat{\mathbf{x}}_0`.
        Should be of shape :math:`(B, *_x)`.
    sigma: torch.Tensor
        The noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    y: torch.Tensor
        The observed data :math:`\mathbf{y}`. Should have the same shape as
        ``x``, that is :math:`(B, *_x)`. When used with a sampler such as in
        instance of the ``EDMStochasticSampler`` class, this should be passed
        as a ``guidance_args`` argument.
    mask: torch.Tensor
        The measurement operator :math:`\text{Mask}`. Should be a tensor of
        boolean values of shape :math:`(B, *_x)`. Values that are ``True``
        correspond to the observed data, and values that are ``False`` correspond
        to the unobserved data (ignored in the guidance). When used with a
        sampler, this should be passed as a ``guidance_args`` argument.

    Outputs
    -------
    torch.Tensor
        The scaled likelihood score of shape :math:`(B, *_x)`.
    """

    def __init__(
        self,
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 2,
        magnitude_scaling: bool = True,
    ):
        super().__init__(
            guide_model=lambda x: x,
            std=std,
            gamma=gamma,
            mu=mu,
            scale=scale,
            power=power,
            norm_ord=norm_ord,
            magnitude_scaling=magnitude_scaling,
            model_exec_mode="batched",
        )

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape != x.shape:
            raise ValueError(
                f"Expected mask and x to have same shape, "
                f"but got {mask.shape} and {x.shape}"
            )
        if y.shape != x.shape:
            raise ValueError(
                f"Expected y and x to have same shape, but got {y.shape} and {x.shape}"
            )
        y_masked = torch.where(mask, y, 0.0)  # (B, *_x)
        return super().__call__(self, x, x_0_hat, sigma, y_masked)
