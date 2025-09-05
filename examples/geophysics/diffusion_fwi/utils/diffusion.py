# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import inspect
from typing import Any, Dict, List, Tuple, TypeAlias, Protocol
from collections.abc import Callable, Sequence

import torch
import nvtx

from physicsnemo.utils.diffusion import StackedRandomGenerator

import deepwave


class ModelBasedGuidance:
    r""" """

    # TODO: for each one of the scaling parameters, need explanations
    # + reference + make sure default values are sensible
    def __init__(
        self,
        guide_model: Callable[[torch.Tensor], torch.Tensor],
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 1,
    ):
        self.guide_model = torch.func.vmap(guide_model)
        self.std = std
        self.gamma = gamma
        self.mu = mu
        self.scale = scale
        self.power = power
        self.norm_ord = norm_ord

    def _log_likelihood(
        self,
        x_0_hat: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Compute L1 error between model prediction and observation
        # NOTE: for now only Tweedie's formula to estimate clean state x_0
        y_x0: torch.Tensor = self.guide_model(x_0_hat)  # (*_y,)
        if y_x0.shape != y.shape:
            raise ValueError(
                f"Expected 'guide_model' output and y to have same shape, "
                f"but got {y_x0.shape} and {y.shape}"
            )
        err1 = torch.abs((y - y_x0)) ** self.norm_ord  # (*_y,)

        # Compute log-likelihood p(y|x_0_hat)
        var = self.std**2 + self.gamma * (t / self.mu) ** 2  # (,)
        log_p = -0.5 * (err1 / var).sum()  # (,)
        return log_p

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        ndim = x.ndim

        # Parameters validation
        if t.shape != (B,):
            raise ValueError(f"Expected t to have shape {(B,)}, but got {t.shape}")
        if y.shape[0] != B:
            raise ValueError(f"Expected y to have batch size {B}, but got {y.shape[0]}")
        if x_0_hat.shape != x.shape:
            raise ValueError(
                f"Expected x_0_hat and x to have same shape, "
                f"but got {x_0_hat.shape} and {x.shape}"
            )

        # NOTE: tensor is detached without requires_grad to save memory
        # (not required with torch.func anyways)
        x_0_hat = x_0_hat.clone().detach().requires_grad_(False)  # (*_x,)

        # Compute likelihood score
        score = torch.func.vmap(
            torch.func.grad(
                self._log_likelihood,
                argnums=0,
            )
        )(x_0_hat, y, t)  # (B, *_x,)

        # Scale the likelihood score
        scale = torch.where(t < 1, self.scale * t.pow(self.power), self.scale).view(
            B, *([1] * (ndim - 1))
        )  # (B, 1, ..., 1)
        score_mag = torch.abs(score).mean(
            dim=tuple(range(1, ndim)), keepdim=True
        )  # (B, 1, ..., 1)
        score_scaled = (
            score * scale * t.view(B, *([1] * (ndim - 1))) / (1 + score_mag)
        )  # (B, *_x)

        return score_scaled


# Some type annotations
_Guidance: TypeAlias = (
    ModelBasedGuidance | DataConsistencyGuidance | Callable[..., torch.Tensor]
)
_SamplerFn: TypeAlias = Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]


class _DiffusionModel(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        *model_args: Any,
        **model_kwargs: Any,
    ) -> torch.Tensor: ...


def DiffusionAdapter(
    model: torch.nn.Module, args_map: Tuple[str, str, Dict[str, str]]
) -> _DiffusionModel:
    """
    Creates a thin wrapper around a module to convert it into a
    diffusion model compatible with other diffusion utilities.

    This wrapper modifies the signature of a model's forward method to match the
    expected interface for conditional diffusion models. It converts a model with
    an original signature ``model(arg1, ..., argN, kwarg1=val1, ..., kwargM=valM,
    **model_kwargs)`` into a model with signature
    ``wrapper(x, t, condition, wrapper_disabled=False, **wrapper_kwargs)``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to wrap with the diffusion adapter interface.
    args_map : Tuple[str, str, Dict[str, str]]
        A tuple containing 3 elements:
        - First element: the name of the parameter in the original model's forward
          method that the latent state `x` should be mapped to.
        - Second element: the name of the parameter in the original model's forward
          method that the diffusion time `t` should be mapped to.
        - Third element: a dictionary mapping keys in the `cond` dictionary
          to parameter names in the original model's forward method.

    Forward
    -------
    x : torch.Tensor
        The latent state vector of the diffusion model, typically of shape (B, *).
    t : torch.Tensor
        The diffusion time. Should be of shape (B,).
    cond : Dict[str, torch.Tensor]
        A dictionary of conditioning variables. Keys are strings identifying
        the conditioning variables names, and values are tensors used for
        conditioning. The keys are typically "x_0", "x_T", "sigma", "noise", etc.
    wrapper_disabled : bool, optional
        Flag to disable the wrapper functionality. When True, the forward method
        reverts to the original model's signature. Default is False.
    **wrapper_kwargs : Any, optional
        Additional arguments to pass to the original model's forward method.
        Should include all arguments from the original signature that are not
        referenced in `args_map`. This includes both positional and keyword
        arguments from the original signature, all converted to keyword
        arguments.

    Outputs
    -------
    output : Any
        The output from the wrapped model's forward method, with the same
        type and shape as the original model would return.

    Notes
    -----
    The wrapper is thin and only holds references to the original model's
    attributes. Any modification of attributes in the wrapper is reflected in the
    original model, and vice versa.

    Example
    -------
    >>> class Model(torch.nn.Module):
    >>>    def __init__(self):
    >>>        super().__init__()
    >>>        self.a = torch.tensor(10.0)
    >>>    def forward(self, x, y, z, u=4, v=5, w=6, **kwargs):
    >>>        return self.a * x, self.a * y, self.a * z, self.a * u, self.a * v, self.a * w
    >>> model = Model()
    >>> wrapper = DiffusionAdapter(
    >>>     model=model,
    >>>     args_map=("w", "u", {"j": "x", "k": "v"})
    >>> )
    >>> x = torch.tensor(1)
    >>> y = torch.tensor(2)
    >>> z = torch.tensor(3)
    >>> u = torch.tensor(-1)
    >>> v = torch.tensor(-2)
    >>> w = torch.tensor(-3)
    >>> model(x, y, z, u=u, v=v, w=w)
    (tensor(10.), tensor(20.), tensor(30.), tensor(-10.), tensor(-20.), tensor(-30.))
    >>> # Can be called with modified signature (x, t, cond, **wrapper_kwargs)
    >>> wrapper(x, w, {"j": y, "k": z}, z=u, y=v)
    (tensor(20.), tensor(-20.), tensor(-10.), tensor(-30.), tensor(30.), tensor(10.))
    >>> # Can be called with original signature with wrapper_disabled=True
    >>> wrapper(x, y, z, wrapper_disabled=True, u=u, v=v, w=w)
    (tensor(10.), tensor(20.), tensor(30.), tensor(-10.), tensor(-20.), tensor(-30.))
    """
    # Safety checks: make sure we don't map twice to the same argument (i.e.
    # targets in args_map are unique)
    if len(args_map[2]) != len(set(args_map[2].values())):
        raise ValueError(
            "Cannot map two values in 'cond' to the same target forward argument."
        )
    if any(arg_name == args_map[0] for arg_name in args_map[2].values()):
        raise ValueError(
            "Cannot map 'x' and a value in 'cond' to the same target forward argument."
        )
    if any(arg_name == args_map[1] for arg_name in args_map[2].values()):
        raise ValueError(
            "Cannot map 't' and a value in 'cond' to the same target forward argument."
        )

    # Unbound original origional forward method
    _orig_forward: Callable[..., Any] = model.__class__.forward

    # Signature of original forward method
    sig = inspect.signature(_orig_forward)

    # Placeholders
    _NoArg, _condArg, _kwArg = object(), object(), object()
    _xArg, _tArg = object(), object()

    # Process each parameter in the original forward method signature
    # and do the mapping if the parameter is a target specified  in args_map
    is_mapped: List[bool, bool, Dict[str, bool]] = [
        False,
        False,
        {k: False for k in args_map[2].keys()},
    ]
    sig_map: Dict[str, Tuple[int, object] | Tuple[int, object, str]] = {}
    for i, p in enumerate(sig.parameters.values()):
        # Skip 'self' argument
        if i == 0:
            continue
        # For now we don't support *args because it's not clear how to pass those
        # to the original forward method
        if p.kind == p.VAR_POSITIONAL:
            raise NotImplementedError("*args is not supported as a forward argument")
        # Avoid conflict with wrapper_disabled in the new forward
        elif p.name == "wrapper_disabled":
            raise ValueError(
                "'wrapper_disabled' kwarg is not supported as a forward argument"
            )
        # Skip **kwargs
        elif p.kind == p.VAR_KEYWORD:
            continue
        # Argument targetted for x (state vector)
        elif p.name == args_map[0]:
            sig_map[p.name] = (i - 1, _xArg)
            is_mapped[0] = True
        # Argument targetted for t (diffusion time)
        elif p.name == args_map[1]:
            sig_map[p.name] = (i - 1, _tArg)
            is_mapped[1] = True
        # Arguments targetted for condition
        elif p.name in args_map[2].values():
            cond_key = next(k for k, v in args_map[2].items() if v == p.name)
            sig_map[p.name] = (i - 1, _condArg, cond_key)
            is_mapped[2][cond_key] = True
        # Signature argument that is not a target in args_map
        else:
            sig_map[p.name] = (i - 1, _kwArg)
    # Safety check: make sure that we mapped all the variables in `args_map`
    if not is_mapped[0] or not is_mapped[1] or not all(is_mapped[2].values()):
        raise ValueError(
            f"Not all variables in 'args_map' were mapped to a forward argument. "
            f"Detail: {is_mapped}"
        )

    # Forward with modified signature
    def _forward(self, *args, wrapper_disabled=False, **kwargs):
        if wrapper_disabled:
            return _orig_forward(self, *args, **kwargs)
        # Extract x (state vector) and condition from args
        x, t, cond = args[0], args[1], args[2]

        # Build a list of arguments to pass to the original forward method
        args_and_kwargs = [_NoArg for _ in range(len(sig_map))]
        for param_name, (idx, arg_type, *cond_key) in sig_map.items():
            if arg_type is _xArg:
                args_and_kwargs[idx] = x
            elif arg_type is _tArg:
                args_and_kwargs[idx] = t
            elif arg_type is _condArg:
                args_and_kwargs[idx] = cond[cond_key[0]]
            elif arg_type is _kwArg:
                args_and_kwargs[idx] = kwargs.pop(param_name)

        # Safety checks
        if _NoArg in args_and_kwargs:
            raise ValueError("Some arguments are missing from 'args_map' or 'kwargs'")

        return _orig_forward(self, *args_and_kwargs, **kwargs)

    # Build a throw-away subclass that installs the override
    subclass = type(
        f"DiffusionAdapter{model.__class__.__name__}",
        (model.__class__,),
        {"forward": _forward},
    )

    # Allocate a blank instance of that subclass
    proxy = object.__new__(subclass)

    # Point its attribute storage at the original one (shared state)
    proxy.__dict__ = model.__dict__

    return proxy


def generate(
    sampler_fn: _SamplerFn,
    x_channels: int,
    x_resolution: Tuple[int, ...],
    rank_batches: List[List[int]] | List[torch.Tensor],
    cond: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    r"""
    Utility function to generate samples from the diffusion model. It starts by
    initializing a batch of noisy latent states :math:`\mathbf{x}_T` and then generates
    a batch of samples :math:`\mathbf{x}_0` by applying the ``sampler_fn`` function.
    It supports in addition generation minibatch by minibatch by splitting the
    seeds in ``rank_batches``.

    The ``sampler_fn`` function is expected to have the following signature:
    ``sampler_fn(x, cond)``, where ``x`` is the latent state and
    ``cond`` is the conditioning variables, as specified below. It should return
    a single tensor corresponding to a batch of generated samples.

    Parameters
    ----------
    sampler_fn : Callable
        Function used to generate samples from the diffusion model.
    x_channels : int
        Number of channels :math:`C_{\mathbf{x}}` for the latent state
        :math:`\mathbf{x}`.
    x_resolution : Tuple[int, ...]
        Spatial resolution :math:`\mathbf{x}`. For example, for a 2D image it
        should be of the form :math:`(H, W)`, where :math:`H` and :math:`W` are
        the height and width of the image, respectively.
    rank_batches : List[List[int]] | List[torch.Tensor]
        List of mini-batches of seeds to process. Each mini-batch is a list of
        seeds. The mini-batches are generated sequentially, and the final generated
        samples are concatenated across the batch dimension. This is typically used
        to generate large ensembles that do not fit in device memory.
    cond : Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditional generation. Can be set to ``{}`` for unconditioned
        generation.
    device : torch.device
        Device to perform computations.

    Returns
    -------
    torch.Tensor
        Generated samples. Has shape ``(B, x_channels, *x_resolution)``, where
        ``B`` is the total number of seeds in ``rank_batches``.
    """

    # Loop over batches
    x_generated = []
    for batch_seeds in rank_batches:
        with nvtx.annotate(f"generate {len(x_generated)}", color="rapids"):
            B = len(batch_seeds)
            if B == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            x_T = rnd.randn(
                (B, x_channels) + x_resolution,
                device=device,
            ).to(memory_format=torch.channels_last)

            with torch.inference_mode():
                x_0: torch.Tensor = sampler_fn(x_T, cond)
            x_generated.append(x_0)
    return torch.cat(x_generated)


def stochastic_sampler(
    model: _DiffusionModel,
    x: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    model_args: Tuple = (),
    model_kwargs: Dict[str, Any] = {},
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    guidance: _Guidance | Sequence[_Guidance] | None = None,
    guidance_args: Tuple | Sequence[Tuple] = (),
    guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]] = {},
) -> torch.Tensor:
    r"""
    EDM sampler with minor changes to enable posterior sampling.
    The sampler starts from a batch of noisy latent states :math:`\mathbf{x}_T`
    and generates a batch of samples :math:`\mathbf{x}_0` by iteratively denoising
    the noisy latent states.

    The diffusion model is expected to be called with:
    ``x_0_hat = model(x, t, cond, *model_args, **model_kwargs)``, where ``x`` is the
    latent state, ``t`` is the diffusion time, ``cond`` is the conditioning
    variables, and ``*model_args`` and ``**model_kwargs`` are additional
    arguments to pass to the model (see below for details on the expected
    arguments). It is expected to return a tensor :math:`\hat{\mathbf{x}}_0` of
    same shape as ``x``, that is an estimate of the clean latent state
    :math:`\mathbf{x}_0`.

    Guidance sampling (e.g. posterior sampling, classifier guidance, etc.) can be
    enabled by passing one or multiple ``guidance`` functions
    to the sampler. The outputs of the guidance functions are summed and added
    to the score function as a correction or drift term.
    Each guidance function must be an instance of the available guidance types (e.g.
    ``ModelBasedGuidance`` for posterior sampling based on consistency with a nonlinear
    model, ``DataConsistencyGuidance`` for guidance based on observed data, etc.)
    For example, in the case of posterior sampling, the guidance function
    should be an instance of ``ModelBasedGuidance`` that returns the
    likelihood score :math:`\nabla_{\mathbf{x}} \log p(\mathbf{y}|\mathbf{x}_t)`,
    :math:`\nabla_{\mathbf{x}} \log p(\mathbf{y}|\mathbf{x}_t)`, which is a
    tensor of same shape as ``x``, and where :math:`\mathbf{y}` is some
    conditioniong variable.

    Parameters
    ----------
    model: _DiffusionModel
        The denoising diffusion model to use in the sampling process. Should be
        an *:math:`\mathbf{x}_0`-prediction* model.
    x: torch.Tensor
        The noisy latent state used as the initial input for the sampler.
        Typically pure noise :math:`\mathbf{x}_T`.
        Should have shape :math:`(B, *)`, where :math:`B` is the batch size and
        :math:`*` is any number of dimensions.
    cond: Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditioning.
    model_args: Tuple, optional, default=()
        Additional positional arguments to pass to the model.
    model_kwargs: Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the model.
    num_steps: int, optional, default=18
        Number of time steps for the sampler.
    sigma_min: float, optional, default=0.002
        Minimum noise level. If the model has a ``sigma_min`` attribute, the
        larger value between the two will be used.
    sigma_max: float, optional, default=800
        Maximum noise level. If the model has a ``sigma_max`` attribute, the
        smaller value between the two will be used.
    rho: float, optional, default=7
        Exponent used in the time step discretization.
    S_churn: float, optional, default=0
        Churn parameter controlling the level of noise added in each step.
    S_min: float, optional, default=0
        Minimum time step for applying churn.
    S_max: float, optional, default=float("inf")
        Maximum time step for applying churn.
    S_noise: float, optional, default=1
        Noise scaling factor applied during the churn step.
    guidance: _Guidance | Sequence[_Guidance] | None, optional, default=None
        Guidance function that is added as a correction to the score function (computed by
        ``model``). Typically used for posterior sampling, classifier guidance,
        etc. Also support multiple guidance functions by passing a list or tuple.
    guidance_args: Tuple | Sequence[Tuple], optional, default=()
        Additional positional arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.
    guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]], optional, default={}
        Additional keyword arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler. Same shape
        :math:`(B, *)` as ``x``. It is typically a denoised latent state
        :math:`\mathbf{x}_0`.
    """

    # Set container structures for guidance functions
    if guidance is None:
        guidances = []
    elif not isinstance(guidance, (list, tuple)):
        guidances = [guidance]
        guidances_args = [guidance_args]
        guidances_kwargs = [guidance_kwargs]
    else:
        if not (len(guidance) == len(guidance_args) == len(guidance_kwargs)):
            raise ValueError(
                f"Number of guidance functions, arguments, and keyword "
                f"arguments must match, but got {len(guidance)}, "
                f"{len(guidance_args)}, {len(guidance_kwargs)}"
            )
        guidances = guidance
        guidances_args = guidance_args
        guidances_kwargs = guidance_kwargs

    B = x.shape[0]

    # Adjust noise levels based on what's supported by the network.
    # Proposed EDM sampler (Algorithm 2) with minor changes to enable
    # posterior sampling
    if hasattr(model, "sigma_min"):
        sigma_min = max(sigma_min, model.sigma_min)
    if hasattr(model, "sigma_max"):
        sigma_max = min(sigma_max, model.sigma_max)
    if hasattr(model, "round_sigma") and callable(model.round_sigma):
        round_sigma = model.round_sigma
    else:
        round_sigma = torch.as_tensor

    # Time step discretization.
    step_indices = torch.arange(num_steps, device=x.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = x * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # TODO: double check why there is a detach and requires_grad_
        x_cur = x_next.detach().requires_grad_()

        # Increase noise temporarily.
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = round_sigma(t_cur + gamma * t_cur)
        x_hat: torch.Tensor = (
            x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)
        ).to(x.device)

        # Move conditioning to the device
        for key, value in cond.items():
            cond[key] = value.to(x.device)

        x_0_hat = model(
            x_hat,
            t_hat.expand(
                B,
            ),
            cond,
            *model_args,
            **model_kwargs,
        )

        # Guidance (e.g. posterior sampling, etc...)
        guidance_sum = 0.0
        if guidances:
            for guidance, guidance_args, guidance_kwargs in zip(
                guidances, guidances_args, guidances_kwargs
            ):
                if isinstance(guidance, ModelBasedGuidance):
                    # TODO: why the guidance uses x_cur for the latent state
                    # instead of x_hat? (but it does use t_hat and not t_cur)
                    guidance_sum += guidance(
                        x_cur,
                        t_hat.expand(
                            B,
                        ),
                        x_0_hat,
                        *guidance_args,
                        **guidance_kwargs,
                    )
                elif isinstance(guidance, DataConsistencyGuidance):
                    pass
                else:
                    raise ValueError(f"Unsupported guidance type: {type(guidance)}")

        # TODO: why likelihood_score is not used to compute d_cur?
        d_cur = (x_hat - x_0_hat) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            x_next = x_next.to(x.device)
            x_0_hat_next = model(
                x_next,
                t_next.expand(
                    B,
                ),
                cond,
                *model_args,
                **model_kwargs,
            )
            d_prime = (x_next - x_0_hat_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime - guidance_sum
            )
    return x_next


def pi_conditioning(
    x_prev,
    x_0_hat,
    measurement,
    sigma,
    std=7.5e-2,
    gamma=5e-2,
    mu=1,
    scale=1,
    power=1,
    **kwargs,
):
    # TODO: need explanatiosn for hard-coded parameters + adapt to new
    # resolution (80x80) + avoid hard-coded values
    def pi_operator(vp, vs, rho):
        # ############### Denormalize #################
        vp_mean = 3035.069357508522
        vp_std = 890.3956
        vs_mean = 1712.469452191763
        vs_std = 551.9505919227604
        vp = vp_mean + vp_std * vp
        vs = vs_mean + vs_std * vs
        # ############### Acquisition Geometry #################
        ny, nx = 70, 70
        dx = 5.0
        nt = 1000
        dt = 0.001
        freq = 15
        peak_time = 1.5 / freq
        n_shots = 5
        source_depth = 1
        receiver_depth = 1
        n_receivers_per_shot = 69

        source_locations = torch.zeros(
            n_shots, 1, 2, dtype=torch.long, device=x_0_hat.device
        )
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = torch.arange(n_shots) * 17

        receiver_locations = torch.zeros(
            n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=x_0_hat.device
        )
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = torch.arange(n_receivers_per_shot).repeat(
            n_shots, 1
        )

        source_amplitudes = (
            deepwave.wavelets.ricker(freq, nt, dt, peak_time)
            .repeat(n_shots, 1, 1)
            .to(x_0_hat.device)
            * 100000.0
        )

        vz, vx = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            grid_spacing=dx,
            dt=dt,
            source_amplitudes_y=source_amplitudes,
            source_amplitudes_x=source_amplitudes,
            source_locations_y=source_locations,
            source_locations_x=source_locations,
            receiver_locations_y=receiver_locations,
            receiver_locations_x=receiver_locations,
            pml_freq=freq,
            pml_width=[20, 20, 20, 20],
        )[-2:]
        out = torch.cat([vx[None, :], vz[None, :]], dim=0)
        return out

    with torch.no_grad():
        out_true = pi_operator(measurement[0], measurement[1], measurement[2])
    # err1 = 0
    # for i in range(x_0_hat.shape[0]):
    #     out_x0 = pi_operator(x_0_hat[i,0],x_0_hat[i,1],x_0_hat[i,2])
    #     err1 += torch.abs(out_x0 - out_true)
    x_0_hat_mean = x_0_hat.mean(dim=0)
    out_x0 = pi_operator(x_0_hat_mean[0], x_0_hat_mean[1], x_0_hat_mean[2])
    err1 = torch.abs(out_x0 - out_true)

    var = std**2 + gamma * (sigma / mu) ** 2
    log_p = -(err1 / var).sum() / 2
    grad = torch.autograd.grad(outputs=log_p, inputs=x_prev)[0]
    if sigma < 1:
        scale = scale * sigma**power
    scaled_score = grad * scale * sigma / (1 + torch.abs(grad).mean())
    return scaled_score, err1.mean().item()
