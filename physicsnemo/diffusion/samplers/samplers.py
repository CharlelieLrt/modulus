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

from typing import Any, Dict, List, Optional, Union

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
    EDMStochasticSolver,
    EulerSolver,
    HeunSolver,
    Solver,
)

SOLVERS: Dict[str, type] = {
    "euler": EulerSolver,
    "heun": HeunSolver,
    "edm_stochastic": EDMStochasticSolver,
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
    time_steps: Union[int, Tensor],
    solver: Union[str, Solver],
    noise_schedule: Optional[Union[str, NoiseScheduler]] = None,
    solver_options: Optional[Dict[str, Any]] = None,
    noise_schedule_options: Optional[Dict[str, Any]] = None,
    time_eval: Optional[List[int]] = None,
) -> Union[Tensor, List[Tensor]]:
    r"""
    Generate samples from a diffusion model.

    This function integrates the diffusion ODE/SDE from a noisy initial state
    ``xT`` to produce clean samples. It supports various numerical solvers
    and noise schedules.

    Parameters
    ----------
    denoiser : DiffusionDenoiser
        A callable that takes ``(x, t)`` and returns the denoised prediction.
        This is typically a preconditioned diffusion model. See
        :class:`~physicsnemo.diffusion.DiffusionDenoiser` for the expected
        interface.
    xT : torch.Tensor
        Initial noisy latent state of shape :math:`(B, *)` where :math:`B`
        is the batch size. This is typically sampled from a Gaussian
        distribution scaled by ``sigma_max``.
    time_steps : int or torch.Tensor
        Either an integer specifying the number of sampling steps (requires
        ``noise_schedule`` to be provided), or a tensor of shape
        :math:`(N + 1,)` containing the explicit time-step values ending
        with 0.
    solver : str or Solver
        The numerical solver to use. Can be a string key from
        ``{"euler", "heun", "edm_stochastic"}`` or an instance of
        :class:`~physicsnemo.diffusion.samplers.solvers.Solver`. See
        :mod:`~physicsnemo.diffusion.samplers.solvers` for details on each
        solver.
    noise_schedule : str or NoiseScheduler, optional
        The noise schedule for generating time-steps. Required when
        ``time_steps`` is an integer. Can be a string key from
        ``{"vp", "ve", "iddpm", "edm"}`` or an instance of
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`.
        See :mod:`~physicsnemo.diffusion.noise_schedulers` for details.
    solver_options : dict, optional
        Additional options passed to the solver constructor. Only used when
        ``solver`` is a string. See individual solver classes for available
        options.
    noise_schedule_options : dict, optional
        Additional options passed to the noise schedule constructor. Only
        used when ``noise_schedule`` is a string. See individual scheduler
        classes for available options.
    time_eval : list of int, optional
        Indices of time-steps at which to return intermediate samples. If
        provided, returns a list of tensors. If ``None``, returns only the
        final sample.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        If ``time_eval`` is ``None``, returns the final denoised sample of
        shape :math:`(B, *)`. Otherwise, returns a list of tensors
        containing samples at the specified time-step indices.

    Raises
    ------
    ValueError
        If ``time_steps`` is an integer but ``noise_schedule`` is not
        provided.
    ValueError
        If ``solver`` is a string not in the available solvers.
    ValueError
        If ``noise_schedule`` is a string not in the available schedules.
    TypeError
        If ``solver`` is neither a string nor a :class:`Solver` instance.
    TypeError
        If ``noise_schedule`` is neither a string, a :class:`NoiseScheduler`
        instance, nor ``None``.

    See Also
    --------
    :mod:`~physicsnemo.diffusion.samplers.solvers` : Available ODE/SDE solvers.
    :mod:`~physicsnemo.diffusion.noise_schedulers` : Available noise schedules.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> denoiser = lambda x, t: x * 0.9
    >>> xT = torch.randn(2, 3, 64, 64) * 80  # scaled by sigma_max
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

    # Validate and instantiate solver
    if isinstance(solver, str):
        if solver not in SOLVERS:
            available = ", ".join(f'"{k}"' for k in SOLVERS.keys())
            raise ValueError(
                f"Unknown solver '{solver}'. Available solvers: {available}."
            )
        solver_cls = SOLVERS[solver]
        solver_instance = solver_cls(denoiser, **solver_options)
    elif isinstance(solver, Solver):
        solver_instance = solver
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
        schedule_instance: Optional[NoiseScheduler] = schedule_cls(
            **noise_schedule_options
        )
    elif isinstance(noise_schedule, NoiseScheduler):
        schedule_instance = noise_schedule
    elif noise_schedule is None:
        schedule_instance = None
    else:
        raise TypeError(
            "noise_schedule must be a string, NoiseScheduler instance, "
            f"or None, got {type(noise_schedule).__name__}."
        )

    # Generate time-steps
    if isinstance(time_steps, int):
        if schedule_instance is None:
            raise ValueError(
                "noise_schedule must be provided when time_steps is an int."
            )
        t_steps = schedule_instance.timesteps(
            time_steps, device=xT.device, dtype=xT.dtype
        )
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
        x = solver_instance.step(x, t_cur_batch, t_next_batch)

        # Collect sample if requested
        if time_eval is not None and i in time_eval:
            samples.append(x.clone())

    # Return based on time_eval
    if time_eval is not None:
        return samples

    return x
