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

"""DPM-Solver++(2M) multistep sampler for diffusion ODEs."""

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class DPMPlusPlus2M(Solver):
    r"""
    DPM-Solver++(2M): second-order multistep sampler for diffusion ODEs.

    Unlike :class:`HeunSolver`, which attains second order with two denoiser
    evaluations per step, this solver attains second order with a single
    evaluation per step by reusing the previous step's data prediction. It
    applies to linear-Gaussian schedules
    :math:`\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
    \boldsymbol{\epsilon}` and steps in the half log-SNR
    :math:`\lambda = \log(\alpha / \sigma)`. With :math:`h` the current and
    :math:`h_-` the previous step size in :math:`\lambda`, the update is:

    .. math::
        \mathbf{x}_{n-1} = \frac{\sigma_{n-1}}{\sigma_n} \, \mathbf{x}_n
        + \alpha_{n-1} \left(1 - e^{-h}\right)
        \left[ \hat{\mathbf{x}}_0^{(n)}
        + \frac{h}{2 h_-}
        \left( \hat{\mathbf{x}}_0^{(n)} - \hat{\mathbf{x}}_0^{(n+1)} \right)
        \right]

    where :math:`\hat{\mathbf{x}}_0^{(n)}` is the data prediction at step
    :math:`n`, recovered internally from the ``denoiser`` output and the
    schedule callbacks. This is the exponential two-step (Adams-Bashforth)
    integrator of the data prediction in :math:`\lambda`, with the first
    exponential moment approximated by :math:`(h/2) W_0` as in DPM-Solver++.
    The first step, and the final step to zero noise, fall back to the
    first-order update without extrapolation; the final step returns
    :math:`\alpha_{n-1} \hat{\mathbf{x}}_0^{(n)}`, matching common practice.

    The four schedule callbacks describe the linear-Gaussian schedule. They
    must come together or not at all; omitting them selects the EDM defaults
    (:math:`\alpha_t = 1`, :math:`\sigma_t = t`). Their signatures are:

    .. code-block:: python

        def alpha_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # alpha(t), shape: (B,)

        def sigma_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # sigma(t), shape: (B,)

        def alpha_dot_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # d(alpha)/dt, shape: (B,)

        def sigma_dot_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # d(sigma)/dt, shape: (B,)

    .. note::

        This solver is **stateful**: it caches the previous data prediction
        across calls to :meth:`step`, so a single instance tracks a single
        trajectory. Call :meth:`reset` before reusing an instance on a new
        trajectory. String-key selection in
        :func:`~physicsnemo.diffusion.samplers.sample` constructs a fresh
        instance for each call, which is always safe.

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it returns
        the right-hand side of the probability-flow ODE; the solver recovers
        the data prediction internally from it and from the schedule
        callbacks. Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    alpha_fn : Callable[[Tensor], Tensor] | None, optional
        The signal coefficient :math:`\alpha_t`, with the signature shown
        above. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha`.
        By default ``None`` (EDM schedule, :math:`\alpha_t = 1`).
    sigma_fn : Callable[[Tensor], Tensor] | None, optional
        The noise level :math:`\sigma_t`, with the signature shown above.
        Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
        By default ``None`` (EDM schedule, :math:`\sigma_t = t`).
    alpha_dot_fn : Callable[[Tensor], Tensor] | None, optional
        The derivative :math:`\dot{\alpha}_t`, with the signature shown
        above. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.alpha_dot`.
        By default ``None`` (EDM schedule, :math:`\dot{\alpha}_t = 0`).
    sigma_dot_fn : Callable[[Tensor], Tensor] | None, optional
        The derivative :math:`\dot{\sigma}_t`, with the signature shown
        above. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma_dot`.
        By default ``None`` (EDM schedule, :math:`\dot{\sigma}_t = 1`).

    Note
    ----
    Reference: `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models <https://arxiv.org/abs/2211.01095>`_

    Examples
    --------
    Basic usage on an EDM schedule (the default schedule callbacks):

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import DPMPlusPlus2M
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> solver = DPMPlusPlus2M(scheduler.get_denoiser(x0_predictor=x0_pred))
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_1 = solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_0 = solver.step(x_1, torch.tensor([2.5]), torch.tensor([0.0]))
    >>> x_0.shape
    torch.Size([1, 3, 8, 8])
    >>> solver.reset()  # Before reusing the instance on a new trajectory

    On a VP schedule, pass the four schedule callbacks from the scheduler:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> solver = DPMPlusPlus2M(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     alpha_fn=scheduler.alpha,
    ...     sigma_fn=scheduler.sigma,
    ...     alpha_dot_fn=scheduler.alpha_dot,
    ...     sigma_dot_fn=scheduler.sigma_dot,
    ... )
    >>> x_1 = solver.step(x_t, torch.tensor([0.6]), torch.tensor([0.3]))
    >>> x_0 = solver.step(x_1, torch.tensor([0.3]), torch.tensor([0.0]))
    >>> x_0.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        alpha_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        sigma_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        alpha_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]]
        | None = None,
        sigma_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]]
        | None = None,
    ) -> None:
        self.denoiser = denoiser
        if (
            alpha_fn is None
            and sigma_fn is None
            and alpha_dot_fn is None
            and sigma_dot_fn is None
        ):
            # Default to the EDM schedule (alpha = 1, sigma = t)
            self.alpha_fn = lambda t: torch.ones_like(t)
            self.sigma_fn = lambda t: t
            self.alpha_dot_fn = lambda t: torch.zeros_like(t)
            self.sigma_dot_fn = lambda t: torch.ones_like(t)
        elif (
            alpha_fn is not None
            and sigma_fn is not None
            and alpha_dot_fn is not None
            and sigma_dot_fn is not None
        ):
            self.alpha_fn = alpha_fn
            self.sigma_fn = sigma_fn
            self.alpha_dot_fn = alpha_dot_fn
            self.sigma_dot_fn = sigma_dot_fn
        else:
            raise ValueError(
                "alpha_fn, sigma_fn, alpha_dot_fn, and sigma_dot_fn must all "
                "be provided or all None."
            )
        self._x0_prev: Tensor | None = None
        self._h_prev: Tensor | None = None

    def reset(self) -> None:
        """
        Clear the cached history from the previous trajectory.

        Call this method before reusing the same solver instance to sample a
        new trajectory. The first :meth:`step` after a reset is a first-order
        update without extrapolation.
        """
        self._x0_prev = None
        self._h_prev = None

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one DPM-Solver++(2M) integration step.

        Successive calls must belong to a single trajectory with consecutive
        time intervals (``t_cur`` equal to the previous call's ``t_next``);
        see :meth:`reset`.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_{n}` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time :math:`t_n` of shape :math:`(B,)`.
        t_next : Tensor
            Target diffusion time :math:`t_{n-1}` of shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{n-1}` at time
            ``t_next``, same shape as ``x``.
        """
        # Ensure contiguous strides so successive denoiser calls (across
        # sampling steps) present the same stride layout to torch.compile,
        # avoiding spurious recompilations / silently divergent traces.
        t_cur = t_cur.contiguous()
        t_next = t_next.contiguous()

        # Shape for broadcasting time-only quantities: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)

        a_cur = self.alpha_fn(t_cur).reshape(expected_shape)
        s_cur = self.sigma_fn(t_cur).reshape(expected_shape)
        a_next = self.alpha_fn(t_next).reshape(expected_shape)
        s_next = self.sigma_fn(t_next).reshape(expected_shape)
        a_dot = self.alpha_dot_fn(t_cur).reshape(expected_shape)
        s_dot = self.sigma_dot_fn(t_cur).reshape(expected_shape)

        # At sigma == 0 the state is noise-free, and the right-hand side may
        # be singular there. Use a surrogate time to keep the call finite;
        # the final mask drops the result, as such a step is the identity.
        is_degenerate = s_cur == 0
        t_cur_safe = torch.where(
            is_degenerate.reshape(t_cur.shape), torch.ones_like(t_cur), t_cur
        )
        a_cur = torch.where(is_degenerate, torch.ones_like(a_cur), a_cur)
        s_cur = torch.where(is_degenerate, torch.ones_like(s_cur), s_cur)
        a_dot = torch.where(is_degenerate, torch.zeros_like(a_dot), a_dot)
        s_dot = torch.where(is_degenerate, torch.ones_like(s_dot), s_dot)

        # Single evaluation; recover the data prediction from the right-hand
        # side. Written to avoid an explicit division by sigma near the
        # endpoint. For the EDM schedule this is x - t * RHS.
        rhs = self.denoiser(x, t_cur_safe)
        x0_hat = (s_dot * x - s_cur * rhs) / (a_cur * s_dot - s_cur * a_dot)

        # Step in lambda = log(alpha / sigma), half the log-SNR. lambda
        # diverges at the zero-noise endpoint, so the general branch uses a
        # strictly positive surrogate and the final mask selects
        # alpha_next * x0_hat.
        is_final = s_next == 0
        s_next_safe = torch.where(is_final, 0.5 * s_cur, s_next)

        compute_dtype = torch.promote_types(t_cur.dtype, torch.float32)
        emh_hp = (s_next_safe.to(compute_dtype) * a_cur.to(compute_dtype)) / (
            a_next.to(compute_dtype) * s_cur.to(compute_dtype)
        )
        h = -torch.log(emh_hp)
        # expm1 rather than 1 - exp(-h): the latter cancels catastrophically
        # for the small h of a fine ladder, losing most of the value in
        # low-precision dtypes.
        one_minus_emh = (-torch.expm1(-h)).to(x.dtype)
        # Computed in the same promoted dtype. On non-terminal EDM steps it
        # equals exp(-h); matching precision avoids rounding differences.
        sigma_ratio = (s_next.to(compute_dtype) / s_cur.to(compute_dtype)).to(x.dtype)

        if self._x0_prev is None or self._h_prev is None:
            x0_bar = x0_hat  # first-order update without extrapolation
        else:
            # Extrapolate the data prediction in lambda with the step ratio
            # h / (2 h_prev). A repeated timestep gives h_prev == 0; use a
            # finite dummy denominator because torch.where evaluates both
            # branches, and fall back to first order there.
            has_history = self._h_prev != 0
            h_prev_safe = torch.where(
                has_history, self._h_prev, torch.ones_like(self._h_prev)
            )
            coeff = torch.where(
                has_history, h / (2.0 * h_prev_safe), torch.zeros_like(h)
            ).to(x.dtype)
            x0_bar = (1.0 + coeff) * x0_hat - coeff * self._x0_prev

        general = sigma_ratio * x + a_next * one_minus_emh * x0_bar
        # At sigma_next == 0, use the lower-order final update
        # alpha_next * x0_hat instead of the extrapolated x0_bar.
        x_next = torch.where(
            is_degenerate, x, torch.where(is_final, a_next * x0_hat, general)
        )

        # Cache unconditionally to avoid a data-dependent branch (device
        # sync, breaks fullgraph); `reset` clears it. Terminal and degenerate
        # steps use a surrogate step size, which is not a real one, so store
        # zero; a repeated timestep naturally yields h == 0 already.
        self._x0_prev = x0_hat
        self._h_prev = torch.where(is_final | is_degenerate, torch.zeros_like(h), h)

        return x_next
