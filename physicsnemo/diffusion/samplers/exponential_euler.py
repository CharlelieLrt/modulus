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

"""First-order exponential Euler solver for semi-linear diffusion ODEs."""

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class ExponentialEulerSolver(Solver):
    r"""
    First-order exponential Euler solver for semi-linear ODEs.

    A general-purpose exponential time differencing (ETD) integrator,
    specialized for ODEs whose right-hand side is the sum of a term that is
    linear in the state and a nonlinear term:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t)
        = A(t) \, \mathbf{x} + N(\mathbf{x}, t)

    ``denoiser`` returns the full right-hand side :math:`D` and ``linear_fn``
    the coefficient :math:`A(t)` of its linear part; the solver isolates the
    nonlinear part :math:`N` by difference. Both callbacks are generic: they
    typically come from a noise scheduler, as in the examples below, but any
    callables with the expected signatures work.

    The method integrates the linear term exactly over the step, with the
    coefficient held at its current value, at the same cost as the explicit
    Euler method: a single evaluation of ``denoiser`` per step. Without a
    linear coefficient, it reduces to the explicit Euler method. Applied to
    the probability-flow ODE of a linear-Gaussian noise schedule, this is
    the first-order DPM-Solver; on an EDM-like schedule with a score or
    noise parameterization it reproduces DDIM, the standard sampler of
    distilled few-step models, as in the second example below.

    The ``linear_fn`` callable has the signature:

    .. code-block:: python

        def linear_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # linear coefficient A(t), shape: (B,)

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it returns
        the right-hand side of the ODE. Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    linear_fn : Callable[[Tensor], Tensor] | None, optional
        The coefficient :math:`A(t)` of the linear part of the ``denoiser``,
        with the signature shown above. Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`,
        with the same predictor parameterization as the ``denoiser``. By
        default ``None``, which corresponds to a zero linear coefficient
        (explicit Euler method).

    Note
    ----
    References:

    - `Exponential integrators <https://doi.org/10.1017/S0962492910000048>`_
      (Hochbruck & Ostermann, 2010)
    - `DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
      Sampling <https://arxiv.org/abs/2206.00927>`_

    Examples
    --------
    Basic usage on the probability-flow ODE of an EDM schedule:

    >>> import torch
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import ExponentialEulerSolver
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> solver = ExponentialEulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(x0_predictor=x0_pred),
    ... )
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_tm1 = solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Reproduce DDIM on an EDM schedule, the standard sampler of distilled
    few-step models. With a score parameterization the linear coefficient
    vanishes and each step re-noises the data prediction to the arrival
    noise level:

    >>> score_pred = lambda x, t: -x * 0.1  # Toy score-predictor
    >>> ddim_solver = ExponentialEulerSolver(
    ...     scheduler.get_denoiser(score_predictor=score_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(score_predictor=score_pred),
    ... )
    >>> x_tm1 = ddim_solver.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        linear_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
        if linear_fn is None:
            self.linear_fn = lambda t: torch.zeros_like(t)
        else:
            self.linear_fn = linear_fn

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one exponential Euler integration step.

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

        h_bc = (t_next - t_cur).reshape(expected_shape)

        # Isolate the nonlinear part of the RHS: N = D - A x
        d_cur = self.denoiser(x, t_cur)
        a_bc = self.linear_fn(t_cur).reshape(expected_shape)
        n_cur = d_cur - a_bc * x

        # h * phi1(h A) = expm1(h A) / A; the A -> 0 limit equals h
        z = h_bc * a_bc
        a_safe = torch.where(a_bc == 0, torch.ones_like(a_bc), a_bc)
        h_phi1 = torch.where(a_bc == 0, h_bc, torch.expm1(z) / a_safe)

        x_next = torch.exp(z) * x + h_phi1 * n_cur

        return x_next
