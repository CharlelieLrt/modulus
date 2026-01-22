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

"""Diffusion model sampling interface."""

from typing import Any, Dict, List, Optional

from torch import Tensor

from physicsnemo.diffusion.base import DiffusionDenoiser
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    IDDPMNoiseScheduler,
    NoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)

from .solvers import (
    EDMStochasticEulerSolver,
    EDMStochasticHeunSolver,
    EulerSolver,
    HeunSolver,
    Solver,
)

SOLVERS: Dict[str, type] = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "edm_stochastic_euler": EDMStochasticEulerSolver,
    "edm_stochastic_heun": EDMStochasticHeunSolver,
}

NOISE_SCHEDULES: Dict[str, type] = {
    "vp": VPNoiseScheduler,
    "ve": VENoiseScheduler,
    "iddpm": IDDPMNoiseScheduler,
    "edm": EDMNoiseScheduler,
}


def sample(
    denoiser: DiffusionDenoiser,
    xT: Tensor,
    time_steps: int | Tensor,
    solver: str | Solver,
    noise_schedule: str | NoiseScheduler | None = None,
    solver_options: Dict[str, Any] | None = None,
    noise_schedule_options: Dict[str, Any] | None = None,
    time_eval: list[int] | None = None,
) -> Tensor | List[Tensor]:
    r"""
    Generate batched samples from a diffusion model.

    This function integrates the diffusion ODE/SDE from a noisy initial state
    :math:`\mathbf{x}_T` to produce clean samples. It supports various
    numerical solvers and noise schedules.

    The diffusion time-steps can be specified in two ways:

    1. **Explicit time-steps**: Pass a 1D tensor to ``time_steps`` containing
       the exact diffusion time values. In this case, ``noise_schedule`` is
       not used and can be omitted.

    2. **Generated time-steps**: Pass an integer to ``time_steps`` specifying
       the number of steps, and provide a ``noise_schedule`` to generate the
       time-step values automatically. The ``noise_schedule`` can be either a
       string key with options specified by ``noise_schedule_options``  or a
       pre-configured
       :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
       instance.

    Similarly, the ``solver`` can be specified as a string key (with optional
    ``solver_options``) or as a pre-configured
    :class:`~physicsnemo.diffusion.samplers.solvers.Solver` instance.

    The operator that progressively removes noise is specified by the
    ``denoiser`` argument (typically an :math:`x_0`-predictor or clean data
    predictor). It must implement the following signature, specified by the
    :class:`~physicsnemo.diffusion.DiffusionDenoiser` interface:

    .. code-block:: python

        def denoiser(x: Tensor, t: Tensor) -> Tensor: ...

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable that takes ``(x, t)`` and returns the an estimate of the
        clean data :math:`\mathbf{x}_0`.See
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` for the expected
        interface.
    xT : Tensor
        Initial noisy latent state of shape :math:`(B, *)` where :math:`B`
        is the batch size. All batch elements share the same diffusion time
        values. The ``dtype`` and ``device`` of ``xT`` determine the ``dtype``
        and ``device`` of the generated samples and any internally created
        tensors.
    time_steps : int | Tensor
        Either an integer specifying the number of sampling steps (requires
        ``noise_schedule`` to be provided), or a 1D tensor of shape
        :math:`(N + 1,)` containing the explicit diffusion time values
        :math:`t_N, t_{N-1}, ..., t_0` in decreasing order. To produce a fully
        denoised latent state :math:`\mathbf{x}_0`, the last element must be
        :math:`t_0 = 0`.
    solver : str | Solver
        The numerical solver to use. Can be a string key or an instance of
        :class:`~physicsnemo.diffusion.samplers.solvers.Solver`. Any object
        implementing the :class:`Solver` interface can be used for custom
        solvers. Available string keys:

        * ``"euler"``: First-order Euler method. Fast but lower quality.
          See :class:`~physicsnemo.diffusion.samplers.solvers.EulerSolver`.

        * ``"heun"``: Second-order Heun method. Higher quality but requires
          two denoiser evaluations per step.
          See :class:`~physicsnemo.diffusion.samplers.solvers.HeunSolver`.

        * ``"edm_stochastic_euler"``: First-order stochastic sampler from
          the EDM paper with configurable noise injection. See
          :class:`~physicsnemo.diffusion.samplers.solvers.EDMStochasticEulerSolver`.

        * ``"edm_stochastic_heun"``: Second-order stochastic sampler from
          the EDM paper with configurable noise injection. See
          :class:`~physicsnemo.diffusion.samplers.solvers.EDMStochasticHeunSolver`.

    noise_schedule : str | NoiseScheduler | None
        The noise schedule for generating time-steps. Required when
        ``time_steps`` is an integer. Can be a string key or an instance of
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`. Any
        subclass of :class:`NoiseScheduler` can be used for custom schedules.
        Available string keys:

        * ``"edm"``: EDM schedule with polynomial spacing. See
          :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`.

        * ``"vp"``: Variance Preserving schedule. See
          :class:`~physicsnemo.diffusion.noise_schedulers.VPNoiseScheduler`.

        * ``"ve"``: Variance Exploding schedule. See
          :class:`~physicsnemo.diffusion.noise_schedulers.VENoiseScheduler`.

        * ``"iddpm"``: Improved DDPM schedule. See
          :class:`~physicsnemo.diffusion.noise_schedulers.IDDPMNoiseScheduler`.

    solver_options : Dict[str, Any] | None
        Additional options passed to the solver constructor. Only used when
        ``solver`` is a string. See individual solver classes for available
        options.
    noise_schedule_options : Dict[str, Any] | None
        Additional options passed to the noise schedule constructor. Only
        used when ``noise_schedule`` is a string. See individual scheduler
        classes for available options.
    time_eval : List[int] | None
        Indices of time-steps at which to return intermediate samples. If
        provided, returns a list of tensors. If ``None``, returns only the
        final denoised latent state :math:`\mathbf{x}_0`.

    Returns
    -------
    Tensor | List[Tensor]
        If ``time_eval`` is ``None``, returns the final denoised latent state
        :math:`\mathbf{x}_0` of shape :math:`(B, *)`. Otherwise, returns a list
        of tensors :math:`\mathbf{x}_t` of shape :math:`(B, *)` containing
        latent states at time-step indices specified in ``time_eval``.

    See Also
    --------
    :mod:`~physicsnemo.diffusion.samplers.solvers` : Available ODE/SDE solvers.
    :mod:`~physicsnemo.diffusion.noise_schedulers` : Available noise schedules.

    Examples
    --------
    Generate samples using the Heun solver with EDM noise schedule and a
    simple toy denoiser:

    >>> import torch
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> denoiser = lambda x, t: x * 0.9
    >>> xT = torch.randn(2, 3, 64, 64) * 80
    >>> x0 = sample(
    ...     denoiser=denoiser,
    ...     xT=xT,
    ...     time_steps=50,
    ...     solver="heun",
    ...     noise_schedule="edm",
    ... )
    >>> x0.shape
    torch.Size([2, 3, 64, 64])
    """
    solver_options = solver_options or {}
    noise_schedule_options = noise_schedule_options or {}

    if isinstance(solver, Solver) and solver_options:
        raise ValueError("solver_options is ignored when solver is a Solver instance.")
    if isinstance(noise_schedule, NoiseScheduler) and noise_schedule_options:
        raise ValueError(
            "noise_schedule_options is ignored when noise_schedule is a "
            "NoiseScheduler instance."
        )
    if isinstance(time_steps, Tensor) and noise_schedule is not None:
        raise ValueError("noise_schedule is ignored when time_steps is a Tensor.")
    if isinstance(time_steps, int) and noise_schedule is None:
        raise ValueError("noise_schedule must be provided when time_steps is an int.")

    # Validate and instantiate solver
    if isinstance(solver, str):
        if solver not in SOLVERS:
            available = ", ".join(f'"{k}"' for k in SOLVERS.keys())
            raise ValueError(
                f"Unknown solver '{solver}'. Available solvers: {available}."
            )
        solver_cls = SOLVERS[solver]
        solver_ = solver_cls(denoiser, **solver_options)
    elif isinstance(solver, Solver):
        solver_ = solver
    else:
        raise TypeError(
            f"solver must be a string or Solver instance, got {type(solver).__name__}."
        )

    # Validate and instantiate noise schedule
    if isinstance(noise_schedule, str):
        if noise_schedule not in NOISE_SCHEDULES:
            available = ", ".join(f'"{k}"' for k in NOISE_SCHEDULES.keys())
            raise ValueError(
                f"Unknown noise_schedule '{noise_schedule}'. "
                f"Available schedules: {available}."
            )
        schedule_cls = NOISE_SCHEDULES[noise_schedule]
        schedule_: Optional[NoiseScheduler] = schedule_cls(**noise_schedule_options)
    elif isinstance(noise_schedule, NoiseScheduler):
        schedule_ = noise_schedule
    elif noise_schedule is None:
        schedule_ = None
    else:
        raise TypeError(
            "noise_schedule must be a string, NoiseScheduler instance, "
            f"or None, got {type(noise_schedule).__name__}."
        )

    # Generate time-steps
    if isinstance(time_steps, int):
        # schedule_ is guaranteed to be not None due to earlier validation
        t_steps = schedule_.timesteps(time_steps, device=xT.device, dtype=xT.dtype)
    elif isinstance(time_steps, Tensor):
        t_steps = time_steps.to(device=xT.device, dtype=xT.dtype)
    else:
        raise TypeError(
            f"time_steps must be int or Tensor, got {type(time_steps).__name__}."
        )

    # Initialize sample collection if time_eval is provided
    samples: List[Tensor] = []

    # Main sampling loop
    x = xT
    num_steps = len(t_steps) - 1  # Last element is 0 (final time)

    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]

        # Expand t to batch dimension: scalar -> (B,)
        batch_size = x.shape[0]
        t_cur_batch = t_cur.expand(batch_size)
        t_next_batch = t_next.expand(batch_size)

        # Perform one solver step
        x = solver_.step(x, t_cur_batch, t_next_batch)

        # Collect sample if requested
        if time_eval is not None and i in time_eval:
            samples.append(x.clone())

    # Return based on time_eval
    if time_eval is not None:
        return samples

    return x
