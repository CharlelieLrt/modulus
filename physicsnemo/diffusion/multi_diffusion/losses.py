# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Multi-diffusion denoising score matching losses for patch-based training."""

from typing import Callable, Literal

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.multi_diffusion.models import MultiDiffusionModel2D
from physicsnemo.diffusion.noise_schedulers import NoiseScheduler


class MultiDiffusionMSEDSMLoss:
    r"""Patch-based MSE denoising score matching loss for multi-diffusion
    training.

    This is the multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`. It operates on
    a :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D`
    wrapper and computes the denoising score matching objective independently
    on each patch. A separate diffusion time is sampled per patch, giving
    :math:`P \times B` independent noise levels per training step.

    The model's patching strategy (random or grid with ``fuse=False``) must
    be configured before using this loss. See
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D` for
    details on patching and condition pre-processing.

    For details on prediction types and ``score_to_x0_fn``, see
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with a patching strategy configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol.
    prediction_type : {"x0", "score"}, default="x0"
        Type of prediction the model outputs.
    score_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction applied to the output.

    Examples
    --------
    **Example 1:** Unconditional model with EDM schedule:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionMSEDSMLoss,
    ... )
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md_model = MultiDiffusionModel2D(
    ...     model=UnconditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ... )
    >>> md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionMSEDSMLoss(md_model, EDMNoiseScheduler())
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> loss = loss_fn(x0)
    >>> loss.shape
    torch.Size([])

    **Example 2:** Conditional model with score prediction and no reduction:

    >>> from tensordict import TensorDict
    >>>
    >>> class ConditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(6, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(torch.cat([x, condition["image"]], dim=1))
    >>>
    >>> cond_md_model = MultiDiffusionModel2D(
    ...     model=ConditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ...     condition_patch={"image": True},
    ... )
    >>> cond_md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> scheduler = EDMNoiseScheduler()
    >>> loss_fn = MultiDiffusionMSEDSMLoss(
    ...     cond_md_model, scheduler,
    ...     prediction_type="score",
    ...     score_to_x0_fn=scheduler.score_to_x0,
    ...     reduction="none",
    ... )
    >>> cond = TensorDict({"image": torch.randn(2, 3, 16, 16)}, batch_size=[2])
    >>> loss = loss_fn(x0, condition=cond)
    >>> loss.shape
    torch.Size([8, 3, 8, 8])

    See Also
    --------
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` :
        Non-patched version of this loss.
    :class:`MultiDiffusionWeightedMSEDSMLoss` :
        Weighted variant that supports per-element masking.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
        noise_scheduler: NoiseScheduler,
        prediction_type: Literal["x0", "score"] = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if prediction_type == "x0":
            self._to_x0 = lambda prediction, x_t, t: prediction

        elif prediction_type == "score":
            if score_to_x0_fn is None:
                raise ValueError(
                    "score_to_x0_fn must be provided when prediction_type='score'."
                )
            self._to_x0 = score_to_x0_fn

        else:
            raise ValueError(
                f"prediction_type must be 'x0' or 'score', got '{prediction_type}'."
            )

        _reductions = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }
        if reduction not in _reductions:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'."
            )
        self._reduce = _reductions[reduction]

    def __call__(
        self,
        x0: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
    ) -> Float[Tensor, "PB C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the multi-diffusion denoising score matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution (batch size
            :math:`B`).

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss of shape
            :math:`(P \times B, C, H_p, W_p)`. Otherwise a scalar tensor.
        """
        # Patch x0 first, then sample per-patch noise
        x0_patched = self.model.patch(x0)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]

        t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        # Forward with pre-patched x and t
        prediction = self.model(x_t, t, condition=condition, patched_x_and_t=True)

        x0_pred = self._to_x0(prediction, x_t, t)

        w = self.noise_scheduler.loss_weight(t)
        loss = w.reshape(-1, *([1] * (x0_pred.ndim - 1))) * (x0_pred - x0_patched) ** 2

        return self._reduce(loss)


class MultiDiffusionWeightedMSEDSMLoss:
    r"""Weighted patch-based MSE denoising score matching loss.

    Identical to :class:`MultiDiffusionMSEDSMLoss` but accepts an
    additional ``weight`` tensor that multiplies the per-element squared
    error. This is the multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`.

    The ``weight`` tensor is provided at global resolution and is patched
    via :meth:`~MultiDiffusionModel2D.patch` alongside
    :math:`\mathbf{x}_0`.

    .. math::
        \mathcal{L} = \mathbb{E}_{t, \boldsymbol{\epsilon}}
        \left[ w(t) \left\| \mathbf{m} \odot
        \left(\hat{\mathbf{x}}_0(\mathbf{x}_t, t)
        - \mathbf{x}_0\right) \right\|^2 \right]

    For additional details, see :class:`MultiDiffusionMSEDSMLoss` and
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with a patching strategy configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol.
    prediction_type : {"x0", "score"}, default="x0"
        Type of prediction the model outputs.
    score_to_x0_fn : callable, optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction applied to the output.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionWeightedMSEDSMLoss,
    ... )
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md_model = MultiDiffusionModel2D(
    ...     model=UnconditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ... )
    >>> md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionWeightedMSEDSMLoss(
    ...     md_model, EDMNoiseScheduler()
    ... )
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> mask = torch.ones(2, 3, 16, 16)
    >>> mask[:, :, :, :8] = 0.0
    >>> loss = loss_fn(x0, weight=mask)
    >>> loss.shape
    torch.Size([])

    See Also
    --------
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss` :
        Non-patched weighted loss.
    :class:`MultiDiffusionMSEDSMLoss` :
        Unweighted variant.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
        noise_scheduler: NoiseScheduler,
        prediction_type: Literal["x0", "score"] = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if prediction_type == "x0":
            self._to_x0 = lambda prediction, x_t, t: prediction

        elif prediction_type == "score":
            if score_to_x0_fn is None:
                raise ValueError(
                    "score_to_x0_fn must be provided when prediction_type='score'."
                )
            self._to_x0 = score_to_x0_fn

        else:
            raise ValueError(
                f"prediction_type must be 'x0' or 'score', got '{prediction_type}'."
            )

        _reductions = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }
        if reduction not in _reductions:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'."
            )
        self._reduce = _reductions[reduction]

    def __call__(
        self,
        x0: Float[Tensor, "B C H W"],
        weight: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
    ) -> Float[Tensor, "PB C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the weighted multi-diffusion DSM loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        weight : Tensor
            Per-element weight of shape :math:`(B, C, H, W)`. Patched
            alongside :math:`\mathbf{x}_0` via
            :meth:`~MultiDiffusionModel2D.patch`.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution.

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss of shape
            :math:`(P \times B, C, H_p, W_p)`. Otherwise a scalar tensor.
        """
        # Patch x0 and weight first, then sample per-patch noise
        x0_patched = self.model.patch(x0)  # (P*B, C, Hp, Wp)
        weight_patched = self.model.patch(weight)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]

        t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        # Forward with pre-patched x and t
        prediction = self.model(x_t, t, condition=condition, patched_x_and_t=True)

        x0_pred = self._to_x0(prediction, x_t, t)

        w = self.noise_scheduler.loss_weight(t)
        loss = (
            w.reshape(-1, *([1] * (x0_pred.ndim - 1)))
            * weight_patched
            * (x0_pred - x0_patched) ** 2
        )

        return self._reduce(loss)
