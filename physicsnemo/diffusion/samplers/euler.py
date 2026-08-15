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

"""First-order Euler solver for diffusion ODEs."""

from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.base import Denoiser

from .base import Solver


class EulerSolver(Solver):
    r"""
    First-order Euler solver for diffusion ODEs.

    This is a fast solver with one denoiser evaluation per step, but typically
    produces lower quality samples compared to higher-order methods.

    The optional ``x_scale_fn`` and ``time_fn`` arguments apply the update
    under a change of variables on the state and on the integration variable,
    so that the integrated ODE is:

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
    the ODE in the original variables. The change of variables can reproduce
    many widely used diffusion samplers, including DDIM and the few-step
    samplers of distilled models. The
    change-of-variables callables are generic and allow arbitrary
    transformations: configurations that reproduce well-known samplers
    typically derive them from a noise schedule, but any ad-hoc choice
    works.

    The change-of-variables callables have the signatures:

    .. code-block:: python

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
        :class:`~physicsnemo.diffusion.Denoiser` interface. Here it returns
        the right-hand side of the ODE, always in the original variables.
        Typically obtained via
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`,
        but any callable with the correct signature works.
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
    References:

    - DDIM: `Denoising Diffusion Implicit Models
      <https://arxiv.org/abs/2010.02502>`_

    Examples
    --------
    Basic usage on a diffusion ODE right-hand side:

    >>> import torch
    >>> from physicsnemo.diffusion.samplers import EulerSolver
    >>>
    >>> denoiser = lambda x, t: x / (1 + t.view(-1, 1, 1, 1)**2)  # Toy denoiser
    >>> solver = EulerSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])

    Reproduce the DDIM sampler on a VP schedule, which is also the standard
    way to sample from distilled few-step models. The change of variables
    rescales the state by :math:`\alpha(t)` and integrates in the
    noise-to-signal ratio (``ntsr``) clock
    :math:`\tau = \sigma(t) / \alpha(t)`:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> scheduler = VPNoiseScheduler()
    >>> x0_pred = lambda x, t: x * 0.1  # Toy x0-predictor
    >>> ntsr = lambda t: scheduler.sigma(t) / scheduler.alpha(t)
    >>> ntsr_dot = lambda t: ntsr(t) * (
    ...     scheduler.sigma_dot(t) / scheduler.sigma(t)
    ...     - scheduler.alpha_dot(t) / scheduler.alpha(t)
    ... )
    >>> ddim_solver = EulerSolver(
    ...     scheduler.get_denoiser(x0_predictor=x0_pred),
    ...     x_scale_fn=scheduler.alpha,
    ...     x_scale_dot_fn=scheduler.alpha_dot,
    ...     time_fn=ntsr,
    ...     time_dot_fn=ntsr_dot,
    ... )
    >>> x_tm1 = ddim_solver.step(x_t, torch.tensor([0.6]), torch.tensor([0.3]))
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: Denoiser,
        x_scale_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        x_scale_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]]
        | None = None,
        time_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
        time_dot_fn: Callable[[Float[Tensor, " B"]], Float[Tensor, " B"]] | None = None,
    ) -> None:
        self.denoiser = denoiser
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

    def step(
        self,
        x: Float[Tensor, " B *dims"],
        t_cur: Float[Tensor, " B"],
        t_next: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""
        Perform one Euler integration step.

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

        # Convert the RHS to the transformed variables:
        # dy/dtau = (D - (s_dot / s) x) / (tau_dot s), with y = x / s
        d_cur = self.denoiser(x, t_cur)
        s_cur_bc = self.x_scale_fn(t_cur).reshape(expected_shape)
        s_dot_cur_bc = self.x_scale_dot_fn(t_cur).reshape(expected_shape)
        tau_dot_cur_bc = self.time_dot_fn(t_cur).reshape(expected_shape)
        g_cur = (d_cur - (s_dot_cur_bc / s_cur_bc) * x) / (tau_dot_cur_bc * s_cur_bc)

        # Euler step in the transformed variables
        h_bc = (self.time_fn(t_next) - self.time_fn(t_cur)).reshape(expected_shape)
        y_next = x / s_cur_bc + h_bc * g_cur
        x_next = self.x_scale_fn(t_next).reshape(expected_shape) * y_next

        return x_next
