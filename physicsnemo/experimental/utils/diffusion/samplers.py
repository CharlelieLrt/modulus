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


from typing import Any, Dict, List, Tuple, TypeAlias
from collections.abc import Callable, Sequence

import torch
import nvtx

from physicsnemo.utils.diffusion import StackedRandomGenerator
from physicsnemo.experimental.utils.diffusion import (
    ModelBasedGuidance, DataConsistencyGuidance,
)

# Some type annotations
_Guidance: TypeAlias = (
    ModelBasedGuidance | DataConsistencyGuidance | Callable[..., torch.Tensor]
)
_SamplerFn: TypeAlias = Callable[
    [torch.Tensor, Dict[str, torch.Tensor], Any, Any], torch.Tensor
]
_DiffusionModel = Callable[
    [torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Any, Any], torch.Tensor
]


def generate(
    sampler_fn: _SamplerFn,
    x_channels: int,
    x_resolution: Tuple[int, ...],
    rank_batches: List[List[int]] | List[torch.Tensor],
    cond: Dict[str, torch.Tensor],
    device: torch.device,
    sampler_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    r"""
    Function to generate samples from a diffusion model. It starts by
    initializing a batch of noisy latent states :math:`\mathbf{x}_T` and then generates
    a batch of clean samples :math:`\mathbf{x}_0` by applying the ``sampler_fn`` function.
    It supports in addition generation minibatch by minibatch by splitting the
    seeds in ``rank_batches``.

    The ``sampler_fn`` function is expected to have the following signature:
    ``sampler_fn(x, cond, **sampler_kwargs)``, where ``x`` is the latent state and
    ``cond`` is the conditioning variables, as specified below. It should return
    a single tensor corresponding to a batch of generated samples. Typically,
    the ``sampler_fn`` function is an instance of ``EDMStochasticSampler``.

    Parameters
    ----------
    sampler_fn : Callable
        Function used to generate samples from the diffusion model.
    x_channels : int
        Number of channels :math:`C_{\mathbf{x}}` for the latent state
        :math:`\mathbf{x}`.
    x_resolution : Tuple[int, ...]
        Spatial resolution :math:`\mathbf{x}`. For example, for a 2D image it
        should be of the form :math:`(H, W)`, where :math:`H` and :math:`W` are
        the height and width of the image, respectively.
    rank_batches : List[List[int]] | List[torch.Tensor]
        List of mini-batches of seeds to process. Each mini-batch is a list of
        seeds. The mini-batches are generated sequentially, and the final generated
        samples are concatenated across the batch dimension. This is typically used
        to generate large ensembles that do not fit in device memory.
    cond : Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditional generation. Can be set to ``{}`` for unconditional
        generation.
    device : torch.device
        Device to perform computations.
    sampler_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the ``sampler_fn`` function.

    Returns
    -------
    torch.Tensor
        Generated samples. Has shape ``(B, x_channels, *x_resolution)``, where
        ``B`` is the total number of seeds in ``rank_batches``.
    """

    # Loop over batches
    x_generated = []
    for batch_seeds in rank_batches:
        with nvtx.annotate(f"generate {len(x_generated)}", color="rapids"):
            B = len(batch_seeds)
            if B == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            x_T = rnd.randn(
                (B, x_channels) + x_resolution,
                device=device,
            ).to(memory_format=torch.channels_last)

            # Call the sampler function
            x_0: torch.Tensor = sampler_fn(x_T, cond, **sampler_kwargs)

            x_generated.append(x_0)
    return torch.cat(x_generated)


class EDMStochasticSampler:
    r"""
    Stochastic sampler proposed in the `EDM paper
    <https://arxiv.org/abs/2206.00364>`_, with optional guidance.
    The sampler starts from a batch of noisy latent states :math:`\mathbf{x}_T`
    and generates a batch of clean samples :math:`\mathbf{x}_0` by iteratively denoising
    the noisy latent states.

    The diffusion model is expected to be called with:
    ``x_0_hat = model(x, sigma, cond, *model_args, **model_kwargs)``, where ``x`` is the
    latent state, ``sigma`` is the noise level, ``cond`` is the conditioning
    variables, and ``*model_args`` and ``**model_kwargs`` are additional
    arguments to pass to the model (see below for details on the expected
    arguments). The model should be an :math:`\mathbf{x}_0`-prediction model.
    It should return a tensor :math:`\hat{\mathbf{x}}_0` of
    same shape as ``x``, that is an estimate of the clean latent state
    :math:`\mathbf{x}_0`.

    Guidance sampling (e.g. posterior sampling, classifier guidance, etc.) can be
    enabled by passing one or multiple ``guidance`` functions
    to the sampler. The outputs of the guidance functions are summed and added
    to the score function as a correction or drift term.
    Each guidance function must be an instance of the available guidance types (e.g.
    ``ModelBasedGuidance`` for posterior sampling based on consistency with a nonlinear
    model, ``DataConsistencyGuidance`` for guidance based data measured at
    specific locations, etc.)
    For example, in the case of posterior sampling, the guidance function
    should be an instance of ``ModelBasedGuidance`` that returns the
    likelihood score :math:`\nabla_{\mathbf{x}} \log p(\mathbf{y}|\mathbf{x}_t)`,
    which is a tensor of same shape as ``x``, and where :math:`\mathbf{y}` is some
    conditioniong variable.

    Parameters
    ----------
    model: _DiffusionModel
        The denoising diffusion model to use in the sampling process. Should be
        an :math:`\mathbf{x}_0`-prediction model.
    num_steps: int, optional, default=18
        Number of time steps for the sampler.
    sigma_min: float, optional, default=0.002
        Minimum noise level. If the model has a ``sigma_min`` attribute, the
        larger value between the two will be used.
    sigma_max: float, optional, default=800
        Maximum noise level. If the model has a ``sigma_max`` attribute, the
        smaller value between the two will be used.
    rho: float, optional, default=7
        Exponent used in the time step discretization.
    S_churn: float, optional, default=0
        Churn parameter controlling the level of noise added in each step.
    S_min: float, optional, default=0
        Minimum time step for applying churn.
    S_max: float, optional, default=float("inf")
        Maximum time step for applying churn.
    S_noise: float, optional, default=1
        Noise scaling factor applied during the churn step.

    Forward
    -------
    x: torch.Tensor
        The noisy latent state used as the initial input for the sampler.
        Typically pure noise :math:`\mathbf{x}_T`.
        Should have shape :math:`(B, *)`, where :math:`B` is the batch size and
        :math:`*` is any number of dimensions.
    cond: Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditioning.
    model_args: Tuple, optional, default=()
        Additional positional arguments to pass to the model.
    model_kwargs: Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the model.
    guidance: _Guidance | Sequence[_Guidance] | None, optional, default=None
        Guidance function that is added as a correction to the score function (computed by
        ``model``). Typically used for posterior sampling, classifier guidance,
        etc. Also support multiple guidance functions by passing a list or tuple.
    guidance_args: Tuple | Sequence[Tuple], optional, default=()
        Additional positional arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.
    guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]], optional, default={}
        Additional keyword arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.
    guidance_second_order: bool | Sequence[bool], optional, default=False
        Whether to compute the guidance function in the second order
        correction. If ``True``, the guidance function is computed twice, while
        if ``False``, it is computed only once, which can typically save some
        computation for guidance functions that are expensive to compute.
        If multiple guidance functions are passed, this should be
        a list or tuple of the same length as the number of guidance functions.

    Outputs
    -------
    torch.Tensor
        The final clean latent state :math:`\mathbf{x}_0` produced by the
        sampler. Same shape :math:`(B, *)` as ``x``.
    """

    def __init__(
        self,
        model: _DiffusionModel,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def __call__(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        model_args: Tuple = (),
        model_kwargs: Dict[str, Any] = {},
        guidance: _Guidance | Sequence[_Guidance] | None = None,
        guidance_args: Tuple | Sequence[Tuple] = (),
        guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]] = {},
        guidance_second_order: bool | Sequence[bool] = False,
    ) -> torch.Tensor:
        # Set container structures for guidance functions
        if guidance is None:
            guidances = []
            guidances_args = []
            guidances_kwargs = []
            guidances_second_order = []
        elif not isinstance(guidance, (list, tuple)):
            guidances = [guidance]
            guidances_args = [guidance_args]
            guidances_kwargs = [guidance_kwargs]
            guidances_second_order = [guidance_second_order]
        elif (
            isinstance(guidance, (list, tuple))
            and isinstance(guidance_args, (list, tuple))
            and isinstance(guidance_kwargs, (list, tuple))
            and isinstance(guidance_second_order, (list, tuple))
        ):
            guidances = guidance
            guidances_args = guidance_args
            guidances_kwargs = guidance_kwargs
            guidances_second_order = guidance_second_order
        else:
            raise ValueError(
                "When multiple guidance functions are passed, 'guidance', "
                "'guidance_args', 'guidance_kwargs', and 'guidance_second_order' "
                "must be lists or tuples of the same length"
            )
        if not (
            len(guidances)
            == len(guidances_args)
            == len(guidances_kwargs)
            == len(guidances_second_order)
        ):
            raise ValueError(
                f"Number of guidance functions, arguments, keyword "
                f"arguments, and second order correction must match, "
                f"but got {len(guidances)}, {len(guidances_args)}, "
                f"{len(guidances_kwargs)}, and {len(guidances_second_order)}"
            )

        # Determine if we need to differentiate through the model
        req_grad: bool = False
        req_grad_sec_ord: bool = False
        for gd, gd_sec_ord in zip(guidances, guidances_second_order):
            if isinstance(gd, (ModelBasedGuidance, DataConsistencyGuidance)):
                req_grad: bool = True
                if gd_sec_ord:
                    req_grad_sec_ord: bool = True

        B = x.shape[0]

        # Adjust noise levels based on what's supported by the network.
        # Proposed EDM sampler (Algorithm 2) with minor changes to enable
        # posterior sampling
        if hasattr(self.model, "sigma_min"):
            sigma_min = max(self.sigma_min, self.model.sigma_min)
        else:
            sigma_min = self.sigma_min
        if hasattr(self.model, "sigma_max"):
            sigma_max = min(self.sigma_max, self.model.sigma_max)
        else:
            sigma_max = self.sigma_max
        if hasattr(self.model, "round_sigma") and callable(self.model.round_sigma):
            round_sigma = self.model.round_sigma
        else:
            round_sigma = torch.as_tensor

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, device=x.device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = x * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # NOTE: break the computational graph here to save memory when
            # computing the guidance terms --> Cannot backpropagate through the
            # sampler, even when guidance is disabled
            x_cur = x_next.detach()

            # Increase noise temporarily.
            gamma = (
                self.S_churn / self.num_steps
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = round_sigma(t_cur + gamma * t_cur)
            x_hat: torch.Tensor = (
                x_cur
                + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            ).to(x.device)

            # Move conditioning to the device
            for key, value in cond.items():
                cond[key] = value.to(x.device)

            # Activate gradient computation if needed for guidance
            with torch.set_grad_enabled(req_grad):
                if req_grad:
                    x_hat_in = x_hat.clone().detach().requires_grad_(True)
                else:
                    x_hat_in = x_hat

                x_0_hat = self.model(
                    x_hat_in,
                    t_hat.expand(
                        B,
                    ),
                    cond,
                    *model_args,
                    **model_kwargs,
                )

                # Guidance terms (e.g. posterior sampling, etc...)
                # Guidance terms required for 2nd order correction are computed
                # twice, while other guidance terms are only computed once to
                # save cost
                gd_sum = 0
                gd_sum_sec_ord = 0
                if guidances:
                    for gd, gd_args, gd_kwargs, gd_sec_ord in zip(
                        guidances,
                        guidances_args,
                        guidances_kwargs,
                        guidances_second_order,
                    ):
                        if isinstance(
                            guidance, (ModelBasedGuidance, DataConsistencyGuidance)
                        ):
                            gd_val = gd(
                                x_hat_in,
                                x_0_hat,
                                t_hat.expand(
                                    B,
                                ),
                                *gd_args,
                                **gd_kwargs,
                            )
                        else:
                            raise ValueError(f"Unsupported guidance type: {type(gd)}")
                        if gd_sec_ord:
                            gd_sum += gd_val
                        else:
                            # Count twice since we only compute once
                            gd_sum_sec_ord += 2 * gd_val

            d_cur = (x_hat - x_0_hat) / t_hat - gd_sum
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd order correction
            if i < self.num_steps - 1:
                x_next = x_next.to(x.device)

                # Activate gradient computation if needed for guidance
                with torch.set_grad_enabled(req_grad_sec_ord):
                    if req_grad_sec_ord:
                        x_next_in = x_next.clone().detach().requires_grad_(True)
                    else:
                        x_next_in = x_next

                    x_0_hat_next = self.model(
                        x_next_in,
                        t_next.expand(
                            B,
                        ),
                        cond,
                        *model_args,
                        **model_kwargs,
                    )

                    # Only recompute guidance terms specifically required in
                    # the 2nd correction
                    if guidances:
                        for gd, gd_args, gd_kwargs, gd_sec_ord in zip(
                            guidances,
                            guidances_args,
                            guidances_kwargs,
                            guidances_second_order,
                        ):
                            if gd_sec_ord:
                                if isinstance(
                                    guidance,
                                    (ModelBasedGuidance, DataConsistencyGuidance),
                                ):
                                    gd_sum_sec_ord += gd(
                                        x_next_in,
                                        x_0_hat_next,
                                        t_next.expand(
                                            B,
                                        ),
                                        *gd_args,
                                        **gd_kwargs,
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported guidance type: {type(gd)}"
                                    )

                d_prime = (x_next - x_0_hat_next) / t_next - gd_sum_sec_ord
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next
