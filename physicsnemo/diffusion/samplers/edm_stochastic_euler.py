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

"""First-order stochastic Euler sampler from the EDM paper."""

import math
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class EDMStochasticEulerSolver(Solver):
    r"""
    First-order stochastic Euler sampler from the EDM paper.

    Implements stochastic sampling with configurable noise injection
    controlled by the "churn" parameters.

    .. important::

        This is **not** a true SDE solver. It performs ad-hoc noise injection
        ("churn") at each step to improve sample diversity, but the underlying
        integration is still an ODE step. Therefore, the denoiser should return
        the right-hand side of the **ODE**, not the SDE.

    By default, noise injection is performed directly in time-step space.
    For linear-Gaussian noise schedules where diffusion time and noise level
    are not equal (e.g., VP schedule), provide ``sigma_fn`` and
    ``sigma_inv_fn`` to apply churn in noise-level space rather than
    time-step space. Optionally provide ``diffusion_fn`` to control the
    time-dependent magnitude of the injected noise.

    Two further options turn this class into the other members of the
    stochastic sampler family. The optional ``x_scale_fn`` and ``time_fn``
    arguments apply the Euler stage under a change of variables on the state
    and on the integration variable, so that the integrated ODE is:

    .. math::
        \frac{d\tilde{\mathbf{x}}}{d\tau} = G(\mathbf{x}, t),
        \qquad
        \tilde{\mathbf{x}} = \frac{\mathbf{x}}{s(t)},
        \qquad
        \tau = \tau(t)

    where :math:`G` is the ``denoiser`` :math:`D` converted internally to
    the transformed coordinates; :math:`D` always returns the right-hand
    side :math:`d\mathbf{x}/dt` of the ODE in the original variables.
    Without a change of variables (the default),
    :math:`G(\mathbf{x}, t) = D(\mathbf{x}, t)`. The
    change-of-variables callables are generic and allow arbitrary
    transformations: configurations that reproduce well-known samplers
    typically derive them from a noise schedule, but any ad-hoc choice
    works.

    The ``renoise`` dial :math:`r \in [0, 1]` adds a second noise injection:
    the Euler stage undershoots to a reduced noise level, and fresh noise
    restores the exact arrival level. Unlike the churn, which perturbs the
    state before the step, this injection trades a fraction of the carried
    noise for fresh noise at the arrival point:

    - ``renoise=0`` keeps all the carried noise and recovers the churn-style
      samplers of the EDM paper.
    - Intermediate values mix carried and fresh noise and give the ancestral
      samplers.
    - ``renoise=1`` renews the noise entirely and gives the re-noising
      sampler of distilled and consistency models (a stochastic DDIM), as in
      the last example below.

    The optional callables have the signatures:

    .. code-block:: python

        def sigma_fn(
            t: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # noise level, same shape as t

        def sigma_inv_fn(
            sigma: Tensor,  # shape: (B,) or broadcastable
        ) -> Tensor: ...  # diffusion time, same shape as sigma

        def diffusion_fn(
            x: Tensor,  # shape: (B, *dims)
            t: Tensor,  # shape: (B,)
        ) -> Tensor: ...  # g^2(x, t), broadcastable to shape of x

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

    Parameters
    ----------
    denoiser : Denoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.Denoiser` interface. Should
        return the right-hand side of the **ODE** in the original variables
        (not the SDE, since this solver handles the stochastic noise
        injection internally). Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
        with ``denoising_type="ode"``.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (deterministic), in which case this
        solver is equivalent to the deterministic :class:`EulerSolver`.
    S_min : float, optional
        Minimum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default 0.
    S_max : float, optional
        Maximum diffusion time (or noise level if ``sigma_fn`` and
        ``sigma_inv_fn`` are provided) for applying churn. By default
        ``float("inf")``.
    S_noise : float, optional
        Noise scaling factor. Large values add more noise to the latent state.
        By default 1.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.
    sigma_fn : Callable[[Tensor], Tensor] | None, optional
        Maps time to noise level :math:`\sigma(t)`. Useful for linear-Gaussian
        schedules where :math:`\sigma(t) \neq t`. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma`.
        If provided, ``sigma_inv_fn`` must also be provided.
        By default ``None`` (identity mapping).
    sigma_inv_fn : Callable[[Tensor], Tensor] | None, optional
        Maps noise level back to time. Typically
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.sigma_inv`.
        If provided, ``sigma_fn`` must also be provided.
        By default ``None`` (identity mapping).
    diffusion_fn : Callable[[Tensor, Tensor], Tensor] | None, optional
        Controls the time-dependent magnitude of the injected
        noise, in addition of the ``S_noise`` scaling factor. Typically the
        squared diffusion coefficient :math:`g^2(\mathbf{x}, t)` from the
        reverse SDE, obtained from
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.diffusion`.
        By default ``None`` (:math:`g^2 = 2t`), which corresponds to an
        EDM-like noise schedule.
    renoise : float, optional
        Fraction :math:`r \in [0, 1]` of the arrival noise level renewed with
        fresh noise at each step, as described above. ``0`` keeps all the
        carried noise, ``1`` renews it entirely, and intermediate values mix
        the two. By default 0.
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
        the signature shown above; requires ``time_dot_fn``. By default
        ``None``, which integrates in the diffusion time itself.
    time_dot_fn : Callable[[Tensor], Tensor] | None, optional
        Time derivative :math:`\dot{\tau}(t)` of ``time_fn``, with the
        signature shown above. Required with ``time_fn``. By default
        ``None``.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    Basic usage with default parameters (noise injection in time-step space):

    >>> import torch
    >>> from physicsnemo.diffusion.samplers import (
    ...     EDMStochasticEulerSolver,
    ... )
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EDMStochasticEulerSolver(denoiser, S_churn=40, num_steps=18)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Using noise scheduler methods for linear-Gaussian schedules where
    :math:`\sigma(t) \neq t` (e.g., VP schedule). The callbacks map between
    time and noise level, allowing the churn to be applied in noise-level
    space before converting back to time-step space:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> num_steps = 10
    >>> solver = EDMStochasticEulerSolver(
    ...     denoiser,
    ...     S_churn=40,
    ...     num_steps=num_steps,
    ...     sigma_fn=scheduler.sigma,
    ...     sigma_inv_fn=scheduler.sigma_inv,
    ...     diffusion_fn=scheduler.diffusion,
    ... )
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Reproduce the stochastic re-noising sampler of distilled and consistency
    models (a stochastic DDIM) on a VP schedule. The DDIM change of variables
    rescales the state by :math:`\alpha(t)` and integrates in the
    noise-to-signal ratio (``ntsr``) clock
    :math:`\tau = \sigma(t) / \alpha(t)`; full noise renewal and no churn
    complete the recipe:

    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> ntsr = lambda t: scheduler.sigma(t) / scheduler.alpha(t)
    >>> ntsr_dot = lambda t: ntsr(t) * (
    ...     scheduler.sigma_dot(t) / scheduler.sigma(t)
    ...     - scheduler.alpha_dot(t) / scheduler.alpha(t)
    ... )
    >>> solver = EDMStochasticEulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     renoise=1.0,
    ...     x_scale_fn=scheduler.alpha,
    ...     x_scale_dot_fn=scheduler.alpha_dot,
    ...     time_fn=ntsr,
    ...     time_dot_fn=ntsr_dot,
    ... )
    >>> x_tm1 = solver.step(x_t, torch.tensor([0.6]), torch.tensor([0.3]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        num_steps: int = 18,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        sigma_inv_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        diffusion_fn: Callable[
            [Float[Tensor, " B *dims"], Float[Tensor, " B"]], Float[Tensor, " B *_"]
        ]
        | None = None,
        renoise: float = 0,
        x_scale_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        x_scale_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]]
        | None = None,
        time_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        time_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.num_steps = num_steps
        if not 0 <= renoise <= 1:
            raise ValueError(f"renoise must be in [0, 1], got {renoise}")
        self.renoise = renoise
        # Noise level kept by the Euler stage, so that the renewed noise
        # restores the exact arrival level
        self._kept_fraction = math.sqrt(1 - renoise**2)
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
        if sigma_fn is None and sigma_inv_fn is None:
            self.sigma_fn = lambda t: t
            self.sigma_inv_fn = lambda sigma: sigma
            self._use_noise_level_space = False
        elif sigma_fn is not None and sigma_inv_fn is not None:
            self.sigma_fn = sigma_fn
            self.sigma_inv_fn = sigma_inv_fn
            self._use_noise_level_space = True
        else:
            raise ValueError(
                "sigma_fn and sigma_inv_fn must both be provided or both None."
            )
        if diffusion_fn is None:
            self.diffusion_fn = lambda x, t: 2 * t.reshape(-1, *([1] * (x.ndim - 1)))
        else:
            self.diffusion_fn = diffusion_fn

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one stochastic Euler sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_n` of shape
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

        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_cur_bc = t_cur.reshape(expected_shape)

        gamma_base = min(self.S_churn / self.num_steps, math.sqrt(2) - 1)

        # Compute perturbed time t_hat with increased noise
        # NOTE: sigma_fn and sigma_inv_fn are identity if not provided (stays
        # in time-step space). diffusion_fn defaults to g^2 = 2t (EDM-like
        # noise schedule).
        sigma_cur_bc = self.sigma_fn(t_cur_bc)
        # Mask: apply churn only where S_min <= sigma <= S_max
        churn_mask = (sigma_cur_bc >= self.S_min) & (sigma_cur_bc <= self.S_max)
        gamma_bc = torch.where(churn_mask, gamma_base, 0.0)
        sigma_hat_bc = sigma_cur_bc + gamma_bc * sigma_cur_bc
        t_hat_bc = self.sigma_inv_fn(sigma_hat_bc)
        # Noise scale: sqrt(sigma_hat^2 - sigma_cur^2) * S_noise * g(x,t) / sqrt(2*t)
        g_sq_bc = self.diffusion_fn(x, t_cur)
        safe_t_cur_bc = torch.where(t_cur_bc == 0, torch.ones_like(t_cur_bc), t_cur_bc)
        noise_scale_bc = (
            (t_hat_bc**2 - t_cur_bc**2).clamp(min=0).sqrt()
            * self.S_noise
            * (g_sq_bc / (2 * safe_t_cur_bc)).sqrt()
        )
        noise_scale_bc = torch.where(
            t_cur_bc == 0, torch.zeros_like(noise_scale_bc), noise_scale_bc
        )

        # Perturb latent with noise
        x_hat = x + noise_scale_bc * torch.randn_like(x)

        # RHS evaluation at t_hat, converted to the transformed variables:
        # dy/dtau = (D - (s_dot / s) x) / (tau_dot s), with y = x / s
        t_hat = t_hat_bc.reshape(x.shape[0])
        d_cur = self.denoiser(x_hat, t_hat)
        s_hat_bc = self.x_scale_fn(t_hat).reshape(expected_shape)
        s_dot_hat_bc = self.x_scale_dot_fn(t_hat).reshape(expected_shape)
        tau_dot_hat_bc = self.time_dot_fn(t_hat).reshape(expected_shape)
        g_cur = (d_cur - (s_dot_hat_bc / s_hat_bc) * x_hat) / (
            tau_dot_hat_bc * s_hat_bc
        )

        # Euler step from t_hat in the transformed variables, aimed at the
        # reduced arrival level kept by the renoise dial
        tau_hat_bc = self.time_fn(t_hat).reshape(expected_shape)
        tau_next_bc = self.time_fn(t_next).reshape(expected_shape)
        tau_dn_bc = self._kept_fraction * tau_next_bc
        y_next = x_hat / s_hat_bc + (tau_dn_bc - tau_hat_bc) * g_cur
        # The zero-renoise branch skips the fresh draw so that renoise=0
        # consumes the same random sequence as the churn-only sampler, which
        # keeps seeded trajectories reproducible across the two
        if self.renoise != 0:
            y_next = y_next + self.renoise * tau_next_bc * torch.randn_like(x)
        x_next = self.x_scale_fn(t_next).reshape(expected_shape) * y_next

        return x_next
