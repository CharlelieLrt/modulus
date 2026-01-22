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

"""ODE/SDE solvers for diffusion model sampling."""

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from physicsnemo.diffusion.base import DiffusionDenoiser


class Solver(ABC):
    r"""
    Abstract base class for diffusion ODE/SDE solvers.

    A solver implements a numerical method to integrate the diffusion process
    from a noisy state to a less noisy (or clean) state. Each call to
    :meth:`step` advances the state from time ``t_cur`` to ``t_next``.

    To create a custom solver, subclass :class:`Solver` and implement the
    :meth:`step` method. The solver can then be used with the
    :func:`~physicsnemo.diffusion.samplers.sample` function.

    Parameters
    ----------
    The denoiser must implement the
    :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface with the
    following signature:

    .. code-block:: python

        def denoiser(x: Tensor, t: Tensor) -> Tensor: ...

    denoiser : DiffusionDenoiser
        A callable that takes ``(x, t)`` and returns the denoised prediction.
        See :class:`~physicsnemo.diffusion.DiffusionDenoiser` for the expected
        interface.

    See Also
    --------
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function that
        uses solvers to generate samples.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import Solver
    >>>
    >>> class SimpleEuler(Solver):
    ...     def step(self, x, t_cur, t_next):
    ...         denoised = self.denoiser(x, t_cur)
    ...         d = (x - denoised) / t_cur
    ...         return x + (t_next - t_cur) * d
    ...
    >>> denoiser = lambda x, t: x * 0.9
    >>> solver = SimpleEuler(denoiser)
    >>> x = torch.randn(2, 3, 32, 32)
    >>> t_cur = torch.tensor([1.0, 1.0])
    >>> t_next = torch.tensor([0.5, 0.5])
    >>> x_next = solver.step(x, t_cur, t_next)
    >>> x_next.shape
    torch.Size([2, 3, 32, 32])
    """

    def __init__(self, denoiser: DiffusionDenoiser) -> None:
        self.denoiser = denoiser

    @abstractmethod
    def step(
        self,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
    ) -> Tensor:
        r"""
        Perform one integration step from ``t_cur`` to ``t_next``.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_t` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time (or noise level) :math:`t` of shape
            :math:`(B,)`.
        t_next : Tensor
            Target diffusion time (or noise level) :math:`t - 1` of shape
            :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{t-1}` at time ``t_next``,
            same shape as ``x``.
        """
        ...


class EulerSolver(Solver):
    r"""
    First-order Euler solver for diffusion ODEs.

    This is the fastest solver but typically produces lower quality samples
    compared to higher-order methods like :class:`HeunSolver`.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface. See
        :class:`Solver` for details.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import EulerSolver
    >>>
    >>> denoiser = lambda x, t: x * 0.9
    >>> solver = EulerSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def step(
        self,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
    ) -> Tensor:
        r"""
        Perform one Euler integration step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_t` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time (or noise level) :math:`t` of shape
            :math:`(B,)`.
        t_next : Tensor
            Target diffusion time (or noise level) :math:`t - 1` of shape
            :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{t-1}` at time ``t_next``,
            same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        denoised = self.denoiser(x, t_cur)
        d_cur = (x - denoised) / t_cur_bc
        x_next = x + (t_next_bc - t_cur_bc) * d_cur

        return x_next


class HeunSolver(Solver):
    r"""
    Second-order Heun solver for diffusion ODEs.

    Also known as the improved Euler method or explicit trapezoidal rule.
    This method requires two denoiser evaluations per step but produces
    higher quality samples than :class:`EulerSolver`.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface. See
        :class:`Solver` for details.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import HeunSolver
    >>>
    >>> denoiser = lambda x, t: x * 0.9
    >>> solver = HeunSolver(denoiser)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def step(
        self,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
    ) -> Tensor:
        r"""
        Perform one Heun integration step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_t` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time (or noise level) :math:`t` of shape
            :math:`(B,)`.
        t_next : Tensor
            Target diffusion time (or noise level) :math:`t - 1` of shape
            :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{t-1}` at time ``t_next``,
            same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        h = t_next_bc - t_cur_bc

        # First denoiser evaluation
        denoised = self.denoiser(x, t_cur)
        d_cur = (x - denoised) / t_cur_bc

        # Predictor step
        x_prime = x + h * d_cur

        # Check if this is the last step (t_next == 0)
        # If so, skip the correction step to avoid division by zero
        if (t_next == 0).all():
            return x_prime

        # Second denoiser evaluation for correction
        denoised_prime = self.denoiser(x_prime, t_next)
        d_prime = (x_prime - denoised_prime) / t_next_bc

        # Corrector step (trapezoidal rule)
        x_next = x + h * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


class EDMStochasticEulerSolver(Solver):
    r"""
    First-order stochastic Euler sampler from the EDM paper.

    Implements stochastic sampling with configurable noise injection
    controlled by the "churn" parameters. Setting ``S_churn=0`` reduces
    this to a deterministic Euler solver.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface. See
        :class:`Solver` for details.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (deterministic).
    S_min : float, optional
        Minimum noise level for applying churn. By default 0.
    S_max : float, optional
        Maximum noise level for applying churn. By default ``float("inf")``.
    S_noise : float, optional
        Noise scaling factor. By default 1.
    randn_like : Callable, optional
        Function to generate random noise with the same shape as input.
        By default ``torch.randn_like``.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import (
    ...     EDMStochasticEulerSolver,
    ... )
    >>> denoiser = lambda x, t: x * 0.9
    >>> solver = EDMStochasticEulerSolver(denoiser, S_churn=0)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
        num_steps: int = 18,
    ) -> None:
        super().__init__(denoiser)
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.randn_like = randn_like
        self.num_steps = num_steps

    def step(
        self,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
    ) -> Tensor:
        r"""
        Perform one stochastic Euler sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_t` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time (or noise level) :math:`t` of shape
            :math:`(B,)`.
        t_next : Tensor
            Target diffusion time (or noise level) :math:`t - 1` of shape
            :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{t-1}` at time ``t_next``,
            same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        # Compute gamma based on churn parameters
        t_cur_scalar = t_cur[0].item() if t_cur.numel() > 0 else t_cur.item()
        if self.S_min <= t_cur_scalar <= self.S_max:
            gamma = self.S_churn / self.num_steps
        else:
            gamma = 0

        # Increase noise temporarily
        t_hat = t_cur_bc + gamma * t_cur_bc
        noise_scale = (t_hat**2 - t_cur_bc**2).sqrt() * self.S_noise
        x_hat = x + noise_scale * self.randn_like(x)

        # Compute denoised prediction at increased noise level
        t_hat_flat = t_hat.reshape(x.shape[0])
        denoised = self.denoiser(x_hat, t_hat_flat)

        # Euler step from t_hat to t_next
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next_bc - t_hat) * d_cur

        return x_next


class EDMStochasticHeunSolver(Solver):
    r"""
    Second-order stochastic Heun sampler from the EDM paper.

    Implements stochastic sampling with configurable noise injection
    controlled by the "churn" parameters, using a second-order Heun
    correction step. Setting ``S_churn=0`` reduces this to a deterministic
    Heun solver.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable implementing the
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface. See
        :class:`Solver` for details.
    S_churn : float, optional
        Controls the amount of noise added at each step. Higher values add
        more stochasticity. By default 0 (deterministic).
    S_min : float, optional
        Minimum noise level for applying churn. By default 0.
    S_max : float, optional
        Maximum noise level for applying churn. By default ``float("inf")``.
    S_noise : float, optional
        Noise scaling factor. By default 1.
    randn_like : Callable, optional
        Function to generate random noise with the same shape as input.
        By default ``torch.randn_like``.
    num_steps : int, optional
        Total number of sampling steps, used to scale churn. By default 18.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers.solvers import (
    ...     EDMStochasticHeunSolver,
    ... )
    >>> denoiser = lambda x, t: x * 0.9
    >>> solver = EDMStochasticHeunSolver(denoiser, S_churn=0)
    >>> x_t = torch.randn(1, 3, 8, 8)
    >>> t_cur = torch.tensor([1.0])
    >>> t_next = torch.tensor([0.5])
    >>> x_tm1 = solver.step(x_t, t_cur, t_next)
    >>> x_tm1.shape
    torch.Size([1, 3, 8, 8])
    """

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
        num_steps: int = 18,
    ) -> None:
        super().__init__(denoiser)
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.randn_like = randn_like
        self.num_steps = num_steps

    def step(
        self,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
    ) -> Tensor:
        r"""
        Perform one stochastic Heun sampling step.

        Parameters
        ----------
        x : Tensor
            Current noisy latent state :math:`\mathbf{x}_t` of shape
            :math:`(B, *)` where :math:`B` is the batch size.
        t_cur : Tensor
            Current diffusion time (or noise level) :math:`t` of shape
            :math:`(B,)`.
        t_next : Tensor
            Target diffusion time (or noise level) :math:`t - 1` of shape
            :math:`(B,)`.

        Returns
        -------
        Tensor
            Updated latent state :math:`\mathbf{x}_{t-1}` at time ``t_next``,
            same shape as ``x``.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, ..., 1)
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))

        # Compute gamma based on churn parameters
        t_cur_scalar = t_cur[0].item() if t_cur.numel() > 0 else t_cur.item()
        if self.S_min <= t_cur_scalar <= self.S_max:
            gamma = self.S_churn / self.num_steps
        else:
            gamma = 0

        # Increase noise temporarily
        t_hat = t_cur_bc + gamma * t_cur_bc
        noise_scale = (t_hat**2 - t_cur_bc**2).sqrt() * self.S_noise
        x_hat = x + noise_scale * self.randn_like(x)

        # Compute denoised prediction at increased noise level
        t_hat_flat = t_hat.reshape(x.shape[0])
        denoised = self.denoiser(x_hat, t_hat_flat)

        # Euler step from t_hat to t_next (predictor)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next_bc - t_hat) * d_cur

        # Apply 2nd order correction if not at last step
        if not (t_next == 0).all():
            denoised_next = self.denoiser(x_next, t_next)
            d_prime = (x_next - denoised_next) / t_next_bc
            d_avg = 0.5 * d_cur + 0.5 * d_prime
            x_next = x_hat + (t_next_bc - t_hat) * d_avg

        return x_next
