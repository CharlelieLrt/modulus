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

"""Second-order exponential Adams-Bashforth solver for semi-linear ODEs."""

from typing import Callable, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class ExponentialAB2Solver(Solver):
    r"""
    Second-order exponential Adams-Bashforth (AB2) solver for semi-linear
    ODEs.

    A general-purpose exponential time differencing (ETD) multistep
    integrator, specialized for ODEs whose right-hand side is the sum of a
    term that is linear in the state and a nonlinear term. The optional
    ``x_scale_fn`` and ``time_fn`` arguments apply the update under a change
    of variables on the state and on the integration variable, so that the
    integrated ODE is:

    .. math::
        \frac{d\tilde{\mathbf{x}}}{d\tau} = G(\mathbf{x}, t),
        \qquad
        \tilde{\mathbf{x}} = \frac{\mathbf{x}}{s(t)},
        \qquad
        \tau = \tau(t)

    where :math:`G` is the ``denoiser`` :math:`D` converted internally to
    the transformed coordinates; :math:`D` always returns the right-hand
    side :math:`d\mathbf{x}/dt` of the ODE in the original variables
    :math:`(\mathbf{x}, t)`. Without a change of variables (the default),
    :math:`G(\mathbf{x}, t) = D(\mathbf{x}, t)` and the solver integrates
    the ODE in the original variables. The change-of-variables callables are
    generic and allow arbitrary transformations: configurations that
    reproduce well-known samplers typically derive them from a noise
    schedule, but any ad-hoc choice works.

    The semi-linear decomposition also lives in the original variables:

    .. math::
        D(\mathbf{x}, t) = A(t) \, \mathbf{x} + N(\mathbf{x}, t)

    ``linear_fn`` provides the coefficient :math:`A(t)`, and the solver
    derives the corresponding decomposition of :math:`G` internally. The
    method integrates the linear term exactly and extrapolates the nonlinear
    term through its two most recent evaluations, reaching second order with
    a single evaluation of ``denoiser`` per step, where :class:`HeunSolver`
    requires two. The first step, and steps where the extrapolation is not
    usable (such as the final step to zero noise under a log-SNR clock),
    fall back to a first-order exponential Euler update. The last example
    below reproduces the multistep diffusion sampler DPM-Solver++(2M) this
    way.

    The ``linear_fn`` and change-of-variables callables have the signatures:

    .. code-block:: python

        def linear_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # linear coefficient A(t), shape: (B,)

        def x_scale_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # scaling s(t), shape: (B,)

        def x_scale_dot_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # derivative ds/dt, shape: (B,)

        def time_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # integration variable tau(t), shape: (B,)

        def time_dot_fn(
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # derivative dtau/dt, shape: (B,)

    .. note::

        This solver is **stateful**: it caches the previous evaluation
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
        the right-hand side of the ODE, always in the original variables.
        Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
    linear_fn : Callable[[Tensor], Tensor] | None, optional
        The coefficient :math:`A(t)` of the linear part of the ``denoiser``,
        with the signature shown above. Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.get_linear_denoiser`,
        with the same predictor parameterization as the ``denoiser``. By
        default ``None``, which corresponds to a zero linear coefficient;
        without a change of variables the update then reduces to the
        classical (variable-step) AB2 method.
    slope_variant : {"heun", "midpoint"}, optional
        Weight of the extrapolated slope term. ``"heun"`` integrates the
        extrapolated nonlinear term exactly; ``"midpoint"`` uses the midpoint
        weight of DPM-Solver++(2M) instead. Both are second order, and they
        coincide when the linear coefficient vanishes. By default ``"heun"``.
    x_scale_fn : Callable[[Tensor], Tensor] | None, optional
        Time-dependent scaling :math:`s(t)` applied to the latent state,
        :math:`\tilde{\mathbf{x}} = \mathbf{x} / s(t)`, with the signature
        shown above; requires ``x_scale_dot_fn``. By default ``None``, which
        applies no rescaling.
    x_scale_dot_fn : Callable[[Tensor], Tensor] | None, optional
        Time derivative :math:`\dot{s}(t)` of ``x_scale_fn``, with the
        signature shown above. Required with ``x_scale_fn``. By default
        ``None``.
    time_fn : Callable[[Tensor], Tensor] | None, optional
        Reparameterization :math:`\tau(t)` of the integration variable, with
        the signature shown above; requires ``time_dot_fn``. The slope
        extrapolation also happens in :math:`\tau`. By default ``None``,
        which integrates in the diffusion time itself.
    time_dot_fn : Callable[[Tensor], Tensor] | None, optional
        Time derivative :math:`\dot{\tau}(t)` of ``time_fn``, with the
        signature shown above. Required with ``time_fn``. By default
        ``None``.

    Note
    ----
    References:

    - `DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
      Probabilistic Models <https://arxiv.org/abs/2211.01095>`_

    Examples
    --------
    Without a linear coefficient and without a change of variables, the
    update is a classical AB2 step (first step: explicit Euler):

    >>> import torch
    >>> from physicsnemo.diffusion.samplers import ExponentialAB2Solver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = ExponentialAB2Solver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> x_1 = solver.step(x_t, torch.tensor([1.0]), torch.tensor([0.5]))
    >>> x_0 = solver.step(x_1, torch.tensor([0.5]), torch.tensor([0.0]))
    >>> x_0.shape
    torch.Size([1, 3, 8, 8])
    >>> solver.reset()  # Before reusing the instance on a new trajectory

    Reproduce DPM-Solver++(2M). The change of variables rescales the state
    by :math:`\alpha(t)` and integrates in the half log-SNR (``lam``) clock
    :math:`\tau = \log(\alpha(t) / \sigma(t))`; the ``"midpoint"`` slope
    weight completes the recipe:

    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> scheduler = EDMNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> lam = lambda t: torch.log(scheduler.alpha(t) / scheduler.sigma(t))
    >>> lam_dot = lambda t: (
    ...     scheduler.alpha_dot(t) / scheduler.alpha(t)
    ...     - scheduler.sigma_dot(t) / scheduler.sigma(t)
    ... )
    >>> dpmpp_2m = ExponentialAB2Solver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     linear_fn=scheduler.get_linear_denoiser(x0_predictor=x0_pred),
    ...     slope_variant="midpoint",
    ...     x_scale_fn=scheduler.alpha,
    ...     x_scale_dot_fn=scheduler.alpha_dot,
    ...     time_fn=lam,
    ...     time_dot_fn=lam_dot,
    ... )
    >>> x_1 = dpmpp_2m.step(x_t, torch.tensor([5.0]), torch.tensor([2.5]))
    >>> x_0 = dpmpp_2m.step(x_1, torch.tensor([2.5]), torch.tensor([0.0]))
    >>> x_0.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        linear_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        slope_variant: Literal["heun", "midpoint"] = "heun",
        x_scale_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        x_scale_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]]
        | None = None,
        time_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        time_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
        if linear_fn is None:
            self.linear_fn = lambda t: torch.zeros_like(t)
        else:
            self.linear_fn = linear_fn
        self.slope_variant = slope_variant
        if x_scale_fn is None and x_scale_dot_fn is None:
            self.x_scale_fn = lambda t: torch.ones_like(t)
            self.x_scale_dot_fn = lambda t: torch.zeros_like(t)
        elif x_scale_fn is not None and x_scale_dot_fn is not None:
            self.x_scale_fn = x_scale_fn
            self.x_scale_dot_fn = x_scale_dot_fn
        else:
            raise ValueError(
                "x_scale_fn and x_scale_dot_fn must both be provided or both None."
            )
        if time_fn is None and time_dot_fn is None:
            self.time_fn = lambda t: t
            self.time_dot_fn = lambda t: torch.ones_like(t)
        elif time_fn is not None and time_dot_fn is not None:
            self.time_fn = time_fn
            self.time_dot_fn = time_dot_fn
        else:
            raise ValueError(
                "time_fn and time_dot_fn must both be provided or both None."
            )
        self._n_prev: Tensor | None = None
        self._tau_prev: Tensor | None = None

    def reset(self) -> None:
        """
        Clear the cached history from the previous trajectory.

        Call this method before reusing the same solver instance to sample a
        new trajectory. The first :meth:`step` after a reset is a first-order
        exponential Euler step.
        """
        self._n_prev = None
        self._tau_prev = None

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one exponential AB2 integration step.

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

        tau_cur_bc = self.time_fn(t_cur).reshape(expected_shape)
        h_bc = self.time_fn(t_next).reshape(expected_shape) - tau_cur_bc
        s_cur_bc = self.x_scale_fn(t_cur).reshape(expected_shape)
        y = x / s_cur_bc

        # Convert the RHS and the linear coefficient to the transformed
        # variables, then split off the nonlinear part:
        # G = (D - (s_dot / s) x) / (tau_dot s)
        # A_tilde = (A - s_dot / s) / tau_dot, N = G - A_tilde y
        d_cur = self.denoiser(x, t_cur)
        s_dot_cur_bc = self.x_scale_dot_fn(t_cur).reshape(expected_shape)
        tau_dot_cur_bc = self.time_dot_fn(t_cur).reshape(expected_shape)
        g_cur = (d_cur - (s_dot_cur_bc / s_cur_bc) * x) / (tau_dot_cur_bc * s_cur_bc)
        a_bc = (
            self.linear_fn(t_cur).reshape(expected_shape) - s_dot_cur_bc / s_cur_bc
        ) / tau_dot_cur_bc
        n_cur = g_cur - a_bc * y

        # h * phi1(h A) = expm1(h A) / A; the A -> 0 limit equals h, and the
        # quotient stays finite for infinite steps in the transformed
        # variable when A < 0
        z = h_bc * a_bc
        a_safe = torch.where(a_bc == 0, torch.ones_like(a_bc), a_bc)
        h_phi1 = torch.where(a_bc == 0, h_bc, torch.expm1(z) / a_safe)

        if self._n_prev is None or self._tau_prev is None:
            # No history yet: first-order exponential Euler step
            y_next = torch.exp(z) * y + h_phi1 * n_cur
        else:
            # Extrapolation ratio in the transformed variable, masked to
            # fall back to first order at non-finite ratios (final step to
            # zero noise under a log-SNR clock, or repeated nodes)
            den_bc = tau_cur_bc - self._tau_prev
            ok = torch.isfinite(h_bc) & torch.isfinite(den_bc) & (den_bc != 0)
            r_safe = torch.where(ok, h_bc, torch.zeros_like(h_bc)) / torch.where(
                ok, den_bc, torch.ones_like(den_bc)
            )
            # Slope weight: "heun" is the exact moment h * phi2(h A) with
            # phi2(z) = (e^z - 1 - z)/z^2; "midpoint" swaps it for
            # h * phi1(h A) / 2 as in DPM-Solver++(2M)
            if self.slope_variant == "midpoint":
                h_phi_slope = h_phi1 / 2
            else:
                z_safe = torch.where(z == 0, torch.ones_like(z), z)
                h_phi_slope = torch.where(
                    z == 0, h_bc / 2, (torch.expm1(z) - z) / (a_safe * z_safe)
                )
            slope_bc = torch.where(ok, h_phi_slope * r_safe, torch.zeros_like(h_bc))
            y_next = (
                torch.exp(z) * y + h_phi1 * n_cur + slope_bc * (n_cur - self._n_prev)
            )

        self._n_prev = n_cur
        self._tau_prev = tau_cur_bc

        x_next = self.x_scale_fn(t_next).reshape(expected_shape) * y_next

        return x_next
