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
from typing import Any, Callable, Dict, List, Tuple

import torch


def DiffusionAdapter(
    model: torch.nn.Module, args_map: Tuple[str, str, Dict[str, str]]
) -> torch.nn.Module:
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
    is_mapped: List[bool, bool, Dict[str, bool]] = [False, False, {k: False for k in args_map[2].keys()}]
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
