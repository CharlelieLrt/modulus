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

"""Noise schedulers for diffusion models."""

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NoiseScheduler(ABC):
    r"""
    Abstract base class for noise schedulers.

    A noise scheduler defines the relationship between diffusion time
    :math:`t` and noise level :math:`\sigma`, as well as methods for
    generating discrete time-steps for sampling.

    Subclasses must implement:

    - :meth:`sigma`: Map diffusion time to noise level
    - :meth:`sigma_inv`: Map noise level back to diffusion time
    - :meth:`timesteps`: Generate discrete time-steps for sampling

    Parameters
    ----------
    sigma_min : float
        Minimum noise level.
    sigma_max : float
        Maximum noise level.

    See Also
    --------
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function that
        uses noise schedulers to generate time-steps.

    Examples
    --------
    To create a custom noise scheduler, subclass :class:`NoiseScheduler` and
    implement the required methods:

    >>> import torch
    >>> class LinearScheduler(NoiseScheduler):
    ...     def __init__(self, sigma_min=0.002, sigma_max=80.0):
    ...         super().__init__(sigma_min, sigma_max)
    ...
    ...     def sigma(self, t: torch.Tensor) -> torch.Tensor:
    ...         return t  # Linear: sigma(t) = t
    ...
    ...     def sigma_inv(self, sigma: torch.Tensor) -> torch.Tensor:
    ...         return sigma  # Inverse: t = sigma
    ...
    ...     def timesteps(self, num_steps: int, device=None) -> torch.Tensor:
    ...         smax, smin = self.sigma_max, self.sigma_min
    ...         steps = torch.linspace(smax, smin, num_steps, device=device)
    ...         return torch.cat([steps, torch.zeros(1, device=device)])
    ...
    >>> scheduler = LinearScheduler()
    >>> t = scheduler.timesteps(5)
    >>> t.shape
    torch.Size([6])
    """

    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        r"""
        Map diffusion time :math:`t` to noise level :math:`\sigma(t)`.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(N,)`.
        """
        ...

    @abstractmethod
    def sigma_inv(self, sigma: Tensor) -> Tensor:
        r"""
        Map noise level :math:`\sigma` back to diffusion time :math:`t`.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Diffusion time :math:`t` of shape :math:`(N,)`.
        """
        ...

    @abstractmethod
    def timesteps(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        r"""
        Generate discrete time-steps for sampling.

        The returned tensor should contain ``num_steps + 1`` values, with the
        last value being 0 (corresponding to the final clean state).

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)` where :math:`N` is
            ``num_steps``.
        """
        ...


class VPNoiseScheduler(NoiseScheduler):
    r"""
    Variance Preserving (VP) noise scheduler.

    Implements the noise schedule from the VP formulation of score-based
    generative models:

    .. math::

        \sigma(t) = \sqrt{\exp\left(\frac{\beta_d}{2} t^2
        + \beta_{\min} t\right) - 1}

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default computed from ``epsilon_s``.
    sigma_max : float, optional
        Maximum noise level, by default computed from ``beta_d`` and
        ``beta_min``.
    beta_d : float, optional
        Extent of the noise level schedule, by default 19.9.
    beta_min : float, optional
        Initial slope of the noise level schedule, by default 0.1.
    epsilon_s : float, optional
        Small time value for numerical stability, by default 1e-3.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    >>> scheduler = VPNoiseScheduler()
    >>> t = scheduler.timesteps(18, device="cpu")
    >>> t.shape
    torch.Size([19])
    """

    def __init__(
        self,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        epsilon_s: float = 1e-3,
    ) -> None:
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_s = epsilon_s

        # Compute default sigma_min/max from VP parameters
        if sigma_min is None:
            sigma_min = self._sigma_from_t(epsilon_s)
        if sigma_max is None:
            sigma_max = self._sigma_from_t(1.0)

        super().__init__(sigma_min, sigma_max)

    def _sigma_from_t(self, t: float) -> float:
        """Compute sigma from t using VP formula (scalar version)."""
        exponent = 0.5 * self.beta_d * (t**2) + self.beta_min * t
        return (math.exp(exponent) - 1) ** 0.5

    def sigma(self, t: Tensor) -> Tensor:
        r"""
        Compute :math:`\sigma(t)` for the VP formulation.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(N,)`.
        """
        exponent = 0.5 * self.beta_d * (t**2) + self.beta_min * t
        return (exponent.exp() - 1).sqrt()

    def sigma_inv(self, sigma: Tensor) -> Tensor:
        r"""
        Compute :math:`t` from :math:`\sigma` for the VP formulation.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Diffusion time :math:`t` of shape :math:`(N,)`.
        """
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def timesteps(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        r"""
        Generate VP time-steps.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        t_frac = step_indices / (num_steps - 1) * (self.epsilon_s - 1)
        orig_t_steps = 1 + t_frac
        sigma_steps = self.sigma(orig_t_steps)
        t_steps = self.sigma_inv(sigma_steps)
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])


class VENoiseScheduler(NoiseScheduler):
    r"""
    Variance Exploding (VE) noise scheduler.

    Implements the noise schedule from the VE formulation:

    .. math::

        \sigma(t) = \sqrt{t}

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.02.
    sigma_max : float, optional
        Maximum noise level, by default 100.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    >>> scheduler = VENoiseScheduler()
    >>> t = scheduler.timesteps(18, device="cpu")
    >>> t.shape
    torch.Size([19])
    """

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
    ) -> None:
        super().__init__(sigma_min, sigma_max)

    def sigma(self, t: Tensor) -> Tensor:
        r"""
        Compute :math:`\sigma(t) = \sqrt{t}` for the VE formulation.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(N,)`.
        """
        return t.sqrt()

    def sigma_inv(self, sigma: Tensor) -> Tensor:
        r"""
        Compute :math:`t = \sigma^2` for the VE formulation.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Diffusion time :math:`t` of shape :math:`(N,)`.
        """
        return sigma**2

    def timesteps(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        r"""
        Generate VE time-steps with geometric spacing.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        # Geometric spacing in sigma^2 space
        ratio = self.sigma_min**2 / self.sigma_max**2
        exponent = step_indices / (num_steps - 1)
        orig_t_steps = (self.sigma_max**2) * (ratio**exponent)
        sigma_steps = self.sigma(orig_t_steps)
        t_steps = self.sigma_inv(sigma_steps)
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])


class IDDPMNoiseScheduler(NoiseScheduler):
    r"""
    Improved DDPM (iDDPM) noise scheduler.

    Implements the noise schedule from the improved DDPM formulation,
    which uses a cosine-based schedule for better sample quality.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.002.
    sigma_max : float, optional
        Maximum noise level, by default 81.
    C_1 : float, optional
        Timestep adjustment at low noise levels, by default 0.001.
    C_2 : float, optional
        Timestep adjustment at high noise levels, by default 0.008.
    M : int, optional
        Number of discretization steps, by default 1000.

    Note
    ----
    Reference: `Improved Denoising Diffusion Probabilistic Models
    <https://arxiv.org/abs/2102.09672>`_

    Examples
    --------
    >>> scheduler = IDDPMNoiseScheduler()
    >>> t = scheduler.timesteps(18, device="cpu")
    >>> t.shape
    torch.Size([19])
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 81.0,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
    ) -> None:
        super().__init__(sigma_min, sigma_max)
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M

        # Precompute the noise level schedule u_j, j = 0, ..., M
        self._u = self._compute_u_schedule()

    def _compute_u_schedule(self) -> Tensor:
        """Precompute the iDDPM noise level schedule."""
        u = torch.zeros(self.M + 1, dtype=torch.float64)
        for j in range(self.M, 0, -1):
            angle_j = 0.5 * math.pi * j / self.M / (self.C_2 + 1)
            angle_jm1 = 0.5 * math.pi * (j - 1) / self.M / (self.C_2 + 1)
            alpha_bar_j = math.sin(angle_j) ** 2
            alpha_bar_jm1 = math.sin(angle_jm1) ** 2
            alpha_ratio = alpha_bar_jm1 / alpha_bar_j
            val = (u[j] ** 2 + 1) / max(alpha_ratio, self.C_1) - 1
            u[j - 1] = val.sqrt()
        return u

    def sigma(self, t: Tensor) -> Tensor:
        r"""
        For iDDPM, sigma and t are the same (identity mapping).

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t) = t` of shape :math:`(N,)`.
        """
        return t

    def sigma_inv(self, sigma: Tensor) -> Tensor:
        r"""
        For iDDPM, t and sigma are the same (identity mapping).

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Diffusion time :math:`t = \sigma` of shape :math:`(N,)`.
        """
        return sigma

    def timesteps(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        r"""
        Generate iDDPM time-steps from precomputed schedule.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        u = self._u.to(device=device, dtype=dtype)
        # Filter to sigma range
        in_range = torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)
        u_filtered = u[in_range]

        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        scale = (len(u_filtered) - 1) / (num_steps - 1)
        indices = (scale * step_indices).round().to(torch.int64)
        sigma_steps = u_filtered[indices]

        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([sigma_steps, zero])


class EDMNoiseScheduler(NoiseScheduler):
    r"""
    EDM noise scheduler.

    Implements the improved noise schedule from the EDM paper with
    polynomial spacing controlled by the ``rho`` parameter.

    For EDM, the noise schedule is identity: :math:`\sigma(t) = t`.

    The time-steps are computed as:

    .. math::

        t_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}
        \left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)
        \right)^{\rho}

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level, by default 0.002.
    sigma_max : float, optional
        Maximum noise level, by default 80.
    rho : float, optional
        Exponent controlling the spacing of time-steps. Values in [5, 10]
        typically produce good results, by default 7.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    >>> scheduler = EDMNoiseScheduler()
    >>> t = scheduler.timesteps(18, device="cpu")
    >>> t.shape
    torch.Size([19])
    >>> t[0]  # First step is sigma_max
    tensor(80., dtype=torch.float64)
    >>> t[-1]  # Last step is 0
    tensor(0., dtype=torch.float64)
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ) -> None:
        super().__init__(sigma_min, sigma_max)
        self.rho = rho

    def sigma(self, t: Tensor) -> Tensor:
        r"""
        For EDM, :math:`\sigma(t) = t` (identity mapping).

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t) = t` of shape :math:`(N,)`.
        """
        return t

    def sigma_inv(self, sigma: Tensor) -> Tensor:
        r"""
        For EDM, :math:`t = \sigma` (identity mapping).

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(N,)`.

        Returns
        -------
        torch.Tensor
            Diffusion time :math:`t = \sigma` of shape :math:`(N,)`.
        """
        return sigma

    def timesteps(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        r"""
        Generate EDM time-steps with polynomial spacing.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.
        device : torch.device, optional
            Device to place the tensor on.
        dtype : torch.dtype, optional
            Data type of the tensor, by default ``torch.float64``.

        Returns
        -------
        torch.Tensor
            Time-steps tensor of shape :math:`(N + 1,)`.
        """
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        smax_inv_rho = self.sigma_max ** (1 / self.rho)
        smin_inv_rho = self.sigma_min ** (1 / self.rho)
        frac = step_indices / (num_steps - 1)
        interp = smax_inv_rho + frac * (smin_inv_rho - smax_inv_rho)
        t_steps = interp**self.rho
        zero = torch.zeros(1, dtype=dtype, device=device)
        return torch.cat([t_steps, zero])
