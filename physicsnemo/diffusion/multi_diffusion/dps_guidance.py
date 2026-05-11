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

"""Patch-local DPS guidance for multi-diffusion sampling."""

from typing import Callable, Protocol, Sequence, runtime_checkable

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from physicsnemo.diffusion.base import Predictor
from physicsnemo.diffusion.multi_diffusion.predictor import MultiDiffusionPredictor


@runtime_checkable
class MultiDiffusionDPSGuidance(Protocol):
    r"""Protocol for patch-local DPS guidance compatible with
    :class:`MultiDiffusionDPSScorePredictor`.

    Identical to the standard
    :class:`~physicsnemo.diffusion.guidance.DPSGuidance` protocol, with one
    extra optional argument ``slice_start`` that enables chunked
    evaluation. The semantics:

    - ``slice_start=None`` (default): the call processes the **whole**
      batch of patches at once. Inputs ``x``, ``t``, ``x_0`` should match
      the size of the pre-patched data stored on the guidance (i.e. the
      full :math:`P \times B`). The implementation may then optionally fuse
      the result back to the global resolution.
    - ``slice_start=s`` (an ``int``): the call processes a **single chunk**
      starting at row ``s`` of the pre-patched data. Inputs ``x``, ``t``,
      ``x_0`` are chunk-sized (:math:`K \leq chunk\_size`). The
      implementation slices its pre-patched data with ``[s : s+K]`` and
      returns a chunk-sized guidance term (no fusing).

    A guidance satisfying this protocol also satisfies
    :class:`~physicsnemo.diffusion.guidance.DPSGuidance` because the extra
    argument is optional.
    """

    def __call__(
        self,
        x: Float[Tensor, " *batch_dims"],
        t: Float[Tensor, " *batch_dims"],
        x_0: Float[Tensor, " *batch_dims"],
        slice_start: int | None = None,
    ) -> Float[Tensor, " *batch_dims"]: ...


class MultiDiffusionDPSScorePredictor(Predictor):
    r"""Score predictor that combines a
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionPredictor`
    with one or more DPS guidances, specialized for **patch-local**
    observation operators on large multi-diffusion domains.

    A guidance is called **patch-local** when the observation
    :math:`y` and the operator :math:`A` decompose along the multi-diffusion
    patch grid: each patch of :math:`y` only depends on the corresponding
    patch of :math:`x_0`. Inpainting with a spatial mask, sparse pointwise
    observations, and any operator that runs separately on each patch fall
    into this category. Cross-patch coupling (a global blur, a global
    Fourier observation) does not.

    When the guidance decomposes patch-locally, this predictor is more
    memory-efficient than
    :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` because it
    streams the per-patch computation:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{x})
        + \sum_i g_i(\mathbf{x}, t, \hat{\mathbf{x}}_0)
        \;=\;
        \mathrm{Fuse}\!\left[\, s^k + \sum_i g_i^k\, \right]_{k=1..P}

    where the superscript :math:`k` denotes the :math:`k`-th patch chunk and
    :math:`\mathrm{Fuse}` is the multi-diffusion fusing operator. Score
    contributions and guidance terms are summed in patch space and fused
    once at the end, which avoids materializing the full
    :math:`(P \times B, \dots)` activation tensor at any point.

    .. important::

        Use :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` for
        guidances that do **not** decompose patch-locally. For those
        operators, the gradient must be computed against the full global
        :math:`x_0`; passing them to this class produces incorrect results.

    All guidances must implement the
    :class:`MultiDiffusionDPSGuidance` protocol:

    .. code-block:: python

        def guidance(
            x: Tensor,                 # noisy patched slice, shape: (K, C, Hp, Wp)
            t: Tensor,                 # diffusion time slice, shape: (K,)
            x_0: Tensor,               # x0 estimate slice, shape: (K, C, Hp, Wp)
            slice_start: int | None,   # row index of the chunk in (P*B);
                                       # None means full-batch evaluation
        ) -> Tensor: ...               # guidance term, shape: (K, C, Hp, Wp)

    This predictor passes the chunk's ``slice_start`` from
    :meth:`MultiDiffusionPredictor.chunks` directly to each guidance, so the
    guidance reads the corresponding slice of its own pre-patched
    observations without any internal state.

    Parameters
    ----------
    x0_predictor : MultiDiffusionPredictor
        A trained predictor with ``chunk_size`` set, returning x0 estimates.
    x0_to_score_fn : callable
        Elementwise conversion ``(x0, x_t, t) -> score``. Typically obtained
        from a noise scheduler, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_score`.
    guidances : MultiDiffusionDPSGuidance or sequence of MultiDiffusionDPSGuidance
        One or more patch-local guidance objects.

    See Also
    --------
    :class:`MultiDiffusionDataConsistencyDPSGuidance` :
        Patch-local guidance for masked observations.
    :class:`MultiDiffusionModelConsistencyDPSGuidance` :
        Patch-local guidance for generic patch-local observation operators.
    :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` :
        Use for non-patch-local guidances.

    Examples
    --------
    **Example 1:** Bare-bone use with a minimal inline guidance and
    ``x0_to_score`` callback. This avoids the noise scheduler and the
    shipped guidance classes to keep the example self-contained:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> # Minimal x0_to_score (EDM convention: score = (x_0 - x) / t**2)
    >>> def x0_to_score_fn(x_0, x, t):
    ...     t_bc = t.reshape((-1,) + (1,) * (x.ndim - 1))
    ...     return (x_0 - x) / (t_bc ** 2)
    >>>
    >>> # Minimal patch-local guidance: gradient of L2 mismatch on a mask.
    >>> # mask and y_obs are pre-patched once and the guidance honours
    >>> # the optional slice_start to align with the predictor's chunks.
    >>> class MinimalInpaintGuidance:
    ...     def __init__(self, mask_patched, y_patched, gamma=0.1):
    ...         self.mask = mask_patched
    ...         self.y = y_patched
    ...         self.gamma = gamma
    ...     def __call__(self, x, t, x_0, slice_start=None):
    ...         if slice_start is None:
    ...             mask, y = self.mask, self.y
    ...         else:
    ...             K = x.shape[0]
    ...             mask = self.mask[slice_start : slice_start + K]
    ...             y = self.y[slice_start : slice_start + K]
    ...         return -self.gamma * mask * (x_0 - y)
    >>>
    >>> mask_patched = predictor.patch_fn(torch.ones(2, 3, 16, 16))
    >>> y_patched = predictor.patch_fn(torch.randn(2, 3, 16, 16))
    >>> guidance = MinimalInpaintGuidance(mask_patched, y_patched)
    >>>
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=x0_to_score_fn,
    ...     guidances=guidance,
    ... )
    >>> x = torch.randn(2, 3, 16, 16)
    >>> t = torch.tensor([1.0, 1.0])
    >>> dps(x, t).shape
    torch.Size([2, 3, 16, 16])

    **Example 2:** Use the shipped patch-local guidance classes for a
    more realistic inpainting setup:

    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDataConsistencyDPSGuidance,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> mask = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask[:, :, 4:, :] = True
    >>> y_obs = torch.randn(2, 3, 16, 16)
    >>>
    >>> guidance = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor, mask=mask, y=y_obs, std_y=0.1,
    ... )
    >>> dps = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance,
    ... )
    >>> dps(x, t).shape
    torch.Size([2, 3, 16, 16])

    **Example 3:** Plug into the standard sampling stack:

    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> denoiser = scheduler.get_denoiser(score_predictor=dps)
    >>> xN = torch.randn(2, 3, 16, 16)
    >>> x0 = sample(denoiser, xN, scheduler, num_steps=4)
    >>> x0.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        x0_predictor: MultiDiffusionPredictor,
        x0_to_score_fn: Callable[
            [
                Float[Tensor, " B C H W"],
                Float[Tensor, " B C H W"],
                Float[Tensor, " B"],
            ],
            Float[Tensor, " B C H W"],
        ],
        guidances: MultiDiffusionDPSGuidance | Sequence[MultiDiffusionDPSGuidance],
    ) -> None:
        if not isinstance(x0_predictor, MultiDiffusionPredictor):
            raise TypeError(
                f"x0_predictor must be a MultiDiffusionPredictor, "
                f"got {type(x0_predictor).__name__}."
            )
        if x0_predictor._chunk_size is None:
            raise ValueError(
                "x0_predictor must have chunk_size set. "
                "Pass chunk_size=<int> to MultiDiffusionPredictor.__init__."
            )
        self.x0_predictor = x0_predictor
        self.x0_to_score_fn = x0_to_score_fn
        if isinstance(guidances, Sequence) and not isinstance(guidances, str):
            self.guidances: list[MultiDiffusionDPSGuidance] = list(guidances)
        else:
            self.guidances = [guidances]  # type: ignore[list-item]

    def __call__(
        self,
        x: Float[Tensor, " B C H W"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B C H W"]:
        r"""Compute the guided score at the global resolution.

        Parameters
        ----------
        x : Tensor
            Noisy latent at global resolution, shape :math:`(B, C, H, W)`.
        t : Tensor
            Diffusion time, shape :math:`(B,)`.

        Returns
        -------
        Tensor
            Guided score at global resolution, shape :math:`(B, C, H, W)`.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionDPSScorePredictor requires autograd but torch "
                "inference mode is enabled. Wrap the calling code with "
                "'with torch.inference_mode(False):' or 'with torch.no_grad():' "
                "instead."
            )

        x = x.detach().requires_grad_(True)
        combined_list: list[Tensor] = []

        with torch.enable_grad():
            for s, x0_chunk, x_chunk, t_chunk in self.x0_predictor.chunks(x, t):
                g_chunk = torch.zeros_like(x0_chunk)
                for g in self.guidances:
                    g_chunk = g_chunk + g(x_chunk, t_chunk, x0_chunk, slice_start=s)
                score_chunk = self.x0_to_score_fn(x0_chunk, x_chunk, t_chunk)
                combined_list.append(score_chunk + g_chunk)

        combined_patched = torch.cat(combined_list, dim=0)  # (P*B, C, Hp, Wp)
        return self.x0_predictor.fuse_fn(combined_patched)


class MultiDiffusionModelConsistencyDPSGuidance:
    r"""Patch-local DPS guidance for generic observation operators.

    Multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance`,
    intended for cases where the observation operator decomposes along the
    multi-diffusion patch grid (cross-patch coupling is not supported, see
    :class:`MultiDiffusionDPSScorePredictor` for the global-coupling case).
    Implements the :class:`MultiDiffusionDPSGuidance` protocol.

    Computes the likelihood score under Gaussian measurement noise. Letting
    :math:`k` index the current patch chunk:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{y}^k | \mathbf{x}_t^k)
        = -\frac{1}{2 \left( \sigma_y^2 + \Gamma \frac{\sigma(t)^2}{\alpha(t)^2}
        \right)} \nabla_{\mathbf{x}^k}
        \| A(\hat{\mathbf{x}}_0^k) - \mathbf{y}^k \|^2

    Observations ``y`` are pre-patched once at construction using the
    predictor's :meth:`~MultiDiffusionPredictor.patch_fn`, so :meth:`__call__`
    does not pay the patching cost on every diffusion step. The L2 norm
    can be replaced by other Lp norms or a custom loss via the ``norm``
    parameter.

    The :meth:`__call__` operates in two modes selected by the
    ``slice_start`` argument:

    - ``slice_start=None``: process the whole batch of patches at once
      using the FULL pre-patched ``y``. Optionally fuse to the global
      resolution if ``fuse=True`` was passed at construction.
    - ``slice_start=s``: process the single chunk starting at row ``s``,
      slicing ``y`` with ``[s : s + K]``. Returns the patched chunk
      guidance (no fuse, regardless of ``fuse``).

    Parameters
    ----------
    predictor : MultiDiffusionPredictor
        Predictor used to pre-patch ``y`` and (optionally) fuse the
        guidance. Stored on ``self.predictor`` for later access.
    observation_operator : callable
        Patch-local observation operator ``A(x0_chunk) -> y_pred_chunk``.
        Must be differentiable.
    y : Tensor
        Global observations of shape :math:`(B, *obs\_dims)` matching the
        output of ``A`` applied at the global resolution.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm : int or callable, default=2
        Loss to apply to the residual. An ``int`` selects the corresponding
        Lp norm. A callable receives ``(y_pred, y_true)`` and returns a
        scalar loss per batch element.
    gamma : float, default=0.0
        SDA covariance scaling factor :math:`\Gamma`. Set to ``0`` for
        classical DPS without SDA scaling.
    sigma_fn : callable or None, default=None
        :math:`t \mapsto \sigma(t)`. Required when ``gamma > 0``.
    alpha_fn : callable or None, default=None
        :math:`t \mapsto \alpha(t)`. Defaults to :math:`\alpha(t) = 1`.
    fuse : bool, default=False
        Whether :meth:`__call__` fuses the guidance term to the global
        resolution when called without ``slice_start`` (full-batch mode).
        Ignored in chunked mode.
    retain_graph : bool, default=False
        Retain the computation graph after the gradient call. Required on
        all but the last guidance when combining multiple autograd-based
        guidances in a single :class:`MultiDiffusionDPSScorePredictor`.
    create_graph : bool, default=False
        Allow higher-order derivatives.

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance` :
        Use for non-patch-local observation operators.
    :class:`MultiDiffusionDPSScorePredictor` :
        Score predictor that consumes this guidance.

    Examples
    --------
    **Example 1:** Patch-local channel selection. The operator selects the
    first channel of each patch — clearly patch-local — so the multi-
    diffusion guidance is appropriate:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionModelConsistencyDPSGuidance,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> A = lambda x: x[:, :1]
    >>> y_obs = torch.randn(2, 1, 16, 16)
    >>>
    >>> guidance = MultiDiffusionModelConsistencyDPSGuidance(
    ...     predictor=predictor, observation_operator=A, y=y_obs, std_y=0.1,
    ... )
    >>> x_chunk = torch.randn(2, 3, 8, 8, requires_grad=True)
    >>> t_chunk = torch.tensor([1.0, 1.0])
    >>> x0_chunk = x_chunk * 0.9
    >>> guidance(x_chunk, t_chunk, x0_chunk, slice_start=0).shape
    torch.Size([2, 3, 8, 8])

    **Example 2:** Full guided sampling pipeline:

    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> md2 = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md2.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md2.eval()
    >>> predictor2 = MultiDiffusionPredictor(md2, chunk_size=2)
    >>> predictor2.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> A2 = lambda x: x[:, :1]
    >>> y_obs2 = torch.randn(2, 1, 16, 16)
    >>> guidance2 = MultiDiffusionModelConsistencyDPSGuidance(
    ...     predictor=predictor2, observation_operator=A2,
    ...     y=y_obs2, std_y=0.1,
    ... )
    >>> dps2 = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor2,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance2,
    ... )
    >>> denoiser2 = scheduler.get_denoiser(score_predictor=dps2)
    >>> xN2 = torch.randn(2, 3, 16, 16)
    >>> x0_2 = sample(denoiser2, xN2, scheduler, num_steps=4)
    >>> x0_2.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        predictor: MultiDiffusionPredictor,
        observation_operator: Callable[
            [Float[Tensor, " K C Hp Wp"]], Float[Tensor, " K *obs_dims"]
        ],
        y: Float[Tensor, " B *obs_dims"],
        std_y: float,
        norm: int
        | Callable[
            [Float[Tensor, " K *obs_dims"], Float[Tensor, " K *obs_dims"]],
            Float[Tensor, " K"],
        ] = 2,
        gamma: float = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        fuse: bool = False,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        if gamma > 0 and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when gamma > 0")
        self.predictor = predictor
        # Pre-patch observations once via the predictor's patch_fn.
        self._y_patched: Tensor = predictor.patch_fn(y)
        self.observation_operator = observation_operator
        self.std_y = std_y
        self.norm = norm
        self.gamma = gamma
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.fuse = fuse
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, " K C Hp Wp"],
        t: Float[Tensor, " K"],
        x_0: Float[Tensor, " K C Hp Wp"],
        slice_start: int | None = None,
    ) -> Float[Tensor, " K C Hp Wp"] | Float[Tensor, " B C H W"]:
        r"""Compute the patch-local likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy patched latent slice :math:`(K, C, H_p, W_p)` with
            ``requires_grad=True``.
        t : Tensor
            Diffusion time slice :math:`(K,)`.
        x_0 : Tensor
            Patched x0 estimate :math:`(K, C, H_p, W_p)` computed from ``x``.
        slice_start : int or None, default=None
            ``None`` processes the whole pre-patched batch at once (and
            optionally fuses if ``fuse=True``). An ``int`` ``s`` processes
            only the chunk starting at row ``s`` of the pre-patched
            observations, returning a patched chunk guidance with no fuse.

        Returns
        -------
        Tensor
            Patch-local guidance term of shape :math:`(K, C, H_p, W_p)`,
            or fused global guidance of shape :math:`(B, C, H, W)` when
            ``slice_start=None`` and ``fuse=True``.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionModelConsistencyDPSGuidance requires autograd "
                "but torch inference mode is enabled."
            )

        if slice_start is None:
            y_chunk = self._y_patched.to(dtype=x.dtype, device=x.device)
        else:
            K = x.shape[0]
            y_chunk = self._y_patched[slice_start : slice_start + K].to(
                dtype=x.dtype, device=x.device
            )

        with torch.enable_grad():
            y_pred = self.observation_operator(x_0)

            norm = self.norm
            if callable(norm):
                loss = norm(y_pred, y_chunk)
            else:
                residual = (y_pred - y_chunk).reshape(y_pred.shape[0], -1)
                loss = residual.abs().pow(norm).sum(dim=1)

            grads = torch.autograd.grad(
                outputs=loss.sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )

        grad_x = grads[0]

        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_bc = t.reshape(expected_shape)
        sigma_t = self.sigma_fn(t_bc)
        alpha_t = self.alpha_fn(t_bc)
        variance = self.std_y**2 + self.gamma * (sigma_t**2) / (alpha_t**2)

        g = -grad_x / (2 * variance)
        if slice_start is None and self.fuse:
            return self.predictor.fuse_fn(g)
        return g


class MultiDiffusionDataConsistencyDPSGuidance:
    r"""Patch-local DPS guidance for masked observations.

    Multi-diffusion counterpart of
    :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance`,
    intended for masked observations whose mask decomposes along the
    multi-diffusion patch grid (each patch's mask is independent of other
    patches). Use cases: inpainting, sparse pointwise data assimilation
    on large domains. Implements the :class:`MultiDiffusionDPSGuidance`
    protocol.

    Computes the likelihood score under Gaussian measurement noise. Letting
    :math:`k` index the current patch chunk:

    .. math::

        \nabla_{\mathbf{x}} \log p(\mathbf{y}^k | \mathbf{x}_t^k)
        = -\frac{1}{2 \left( \sigma_y^2 + \Gamma \frac{\sigma(t)^2}{\alpha(t)^2}
        \right)} \nabla_{\mathbf{x}^k}
        \| \mathbf{M}^k \odot (\hat{\mathbf{x}}_0^k - \mathbf{y}^k) \|^2

    Both ``mask`` and ``y`` are pre-patched once at construction via the
    predictor's :meth:`~MultiDiffusionPredictor.patch_fn`, so subsequent
    diffusion steps do not pay the patching cost. The L2 norm can be
    replaced by other Lp norms or a custom loss via the ``norm`` parameter.

    The :meth:`__call__` operates in two modes selected by the
    ``slice_start`` argument:

    - ``slice_start=None``: process the whole batch of patches at once
      using the FULL pre-patched ``mask`` and ``y``. Optionally fuse to the
      global resolution if ``fuse=True`` was passed at construction.
    - ``slice_start=s``: process the single chunk starting at row ``s``,
      slicing ``mask`` and ``y`` with ``[s : s + K]``. Returns the patched
      chunk guidance (no fuse, regardless of ``fuse``).

    Parameters
    ----------
    predictor : MultiDiffusionPredictor
        Predictor used to pre-patch ``mask`` and ``y`` and (optionally)
        fuse the guidance. Stored on ``self.predictor`` for later access.
    mask : Tensor
        Boolean mask of shape :math:`(B, *)`. ``True`` marks observed
        locations, ``False`` marks missing.
    y : Tensor
        Observed values of shape :math:`(B, *)`. Values at unobserved
        locations are ignored.
    std_y : float
        Standard deviation of the measurement noise :math:`\sigma_y`.
    norm : int or callable, default=2
        Loss to apply to the masked residual. An ``int`` selects the
        corresponding Lp norm. A callable receives
        ``(mask * x0, mask * y)`` and returns a scalar loss per batch element.
    gamma : float, default=0.0
        SDA covariance scaling factor :math:`\Gamma`. Set to ``0`` for
        classical DPS without SDA scaling.
    sigma_fn : callable or None, default=None
        :math:`t \mapsto \sigma(t)`. Required when ``gamma > 0``.
    alpha_fn : callable or None, default=None
        :math:`t \mapsto \alpha(t)`. Defaults to :math:`\alpha(t) = 1`.
    fuse : bool, default=False
        Whether :meth:`__call__` fuses the guidance term to the global
        resolution when called without ``slice_start`` (full-batch mode).
        Ignored in chunked mode.
    retain_graph : bool, default=False
        Retain the computation graph after the gradient call. Required on
        all but the last guidance when combining multiple autograd-based
        guidances in a single :class:`MultiDiffusionDPSScorePredictor`.
    create_graph : bool, default=False
        Allow higher-order derivatives.

    Note
    ----
    References:

    - DPS: `Diffusion Posterior Sampling for General Noisy Inverse Problems
      <https://arxiv.org/abs/2209.14687>`_
    - SDA: `Score-based Data Assimilation <https://arxiv.org/abs/2306.10574>`_

    See Also
    --------
    :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance` :
        Use for non-patch-local masks.
    :class:`MultiDiffusionDPSScorePredictor` :
        Score predictor that consumes this guidance.

    Examples
    --------
    **Example 1:** Inpainting on a large domain. The mask is a spatial
    pattern, so it decomposes along the patch grid:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D, MultiDiffusionPredictor,
    ...     MultiDiffusionDataConsistencyDPSGuidance,
    ... )
    >>>
    >>> class Backbone(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md.eval()
    >>> predictor = MultiDiffusionPredictor(md, chunk_size=2)
    >>> predictor.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> mask = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask[:, :, 4:, :] = True
    >>> y_obs = torch.randn(2, 3, 16, 16)
    >>>
    >>> guidance = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor, mask=mask, y=y_obs, std_y=0.1,
    ... )
    >>> x_chunk = torch.randn(2, 3, 8, 8, requires_grad=True)
    >>> t_chunk = torch.tensor([1.0, 1.0])
    >>> x0_chunk = x_chunk * 0.9
    >>> guidance(x_chunk, t_chunk, x0_chunk, slice_start=0).shape
    torch.Size([2, 3, 8, 8])

    **Example 2:** Full guided sampling pipeline:

    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionDPSScorePredictor,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.samplers import sample
    >>>
    >>> scheduler = EDMNoiseScheduler()
    >>> md2 = MultiDiffusionModel2D(Backbone(), global_spatial_shape=(16, 16))
    >>> md2.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> _ = md2.eval()
    >>> predictor2 = MultiDiffusionPredictor(md2, chunk_size=2)
    >>> predictor2.set_patching(overlap_pix=0, boundary_pix=0)
    >>>
    >>> mask2 = torch.zeros(2, 3, 16, 16, dtype=torch.bool)
    >>> mask2[:, :, 2, 3] = True
    >>> y_obs2 = torch.randn(2, 3, 16, 16)
    >>> guidance2 = MultiDiffusionDataConsistencyDPSGuidance(
    ...     predictor=predictor2, mask=mask2, y=y_obs2, std_y=0.1,
    ... )
    >>> dps2 = MultiDiffusionDPSScorePredictor(
    ...     x0_predictor=predictor2,
    ...     x0_to_score_fn=scheduler.x0_to_score,
    ...     guidances=guidance2,
    ... )
    >>> denoiser2 = scheduler.get_denoiser(score_predictor=dps2)
    >>> xN2 = torch.randn(2, 3, 16, 16)
    >>> x0_2 = sample(denoiser2, xN2, scheduler, num_steps=4)
    >>> x0_2.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        predictor: MultiDiffusionPredictor,
        mask: Bool[Tensor, " B *dims"],
        y: Float[Tensor, " B *dims"],
        std_y: float,
        norm: int
        | Callable[
            [Float[Tensor, " K *dims"], Float[Tensor, " K *dims"]],
            Float[Tensor, " K"],
        ] = 2,
        gamma: float = 0.0,
        sigma_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        alpha_fn: Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
        | None = None,
        fuse: bool = False,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        if gamma > 0 and sigma_fn is None:
            raise ValueError("sigma_fn must be provided when gamma > 0")
        self.predictor = predictor
        # Pre-patch mask and observations once via the predictor's patch_fn.
        patch = predictor.patch_fn
        self._mask_patched: Tensor = patch(mask.float())
        self._y_patched: Tensor = patch(y)
        self.std_y = std_y
        self.norm = norm
        self.gamma = gamma
        self.sigma_fn = (
            sigma_fn if sigma_fn is not None else lambda t: torch.zeros_like(t)
        )
        self.alpha_fn = (
            alpha_fn if alpha_fn is not None else lambda t: torch.ones_like(t)
        )
        self.fuse = fuse
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(
        self,
        x: Float[Tensor, " K C Hp Wp"],
        t: Float[Tensor, " K"],
        x_0: Float[Tensor, " K C Hp Wp"],
        slice_start: int | None = None,
    ) -> Float[Tensor, " K C Hp Wp"] | Float[Tensor, " B C H W"]:
        r"""Compute the patch-local likelihood score guidance term.

        Parameters
        ----------
        x : Tensor
            Noisy patched latent slice :math:`(K, C, H_p, W_p)` with
            ``requires_grad=True``.
        t : Tensor
            Diffusion time slice :math:`(K,)`.
        x_0 : Tensor
            Patched x0 estimate :math:`(K, C, H_p, W_p)` computed from ``x``.
        slice_start : int or None, default=None
            ``None`` processes the whole pre-patched batch at once (and
            optionally fuses if ``fuse=True``). An ``int`` ``s`` processes
            only the chunk starting at row ``s`` of the pre-patched mask
            and observations, returning a patched chunk guidance with no
            fuse.

        Returns
        -------
        Tensor
            Patch-local guidance term of shape :math:`(K, C, H_p, W_p)`,
            or fused global guidance of shape :math:`(B, C, H, W)` when
            ``slice_start=None`` and ``fuse=True``.
        """
        if not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "MultiDiffusionDataConsistencyDPSGuidance requires autograd "
                "but torch inference mode is enabled."
            )

        if slice_start is None:
            mask_chunk = self._mask_patched.to(dtype=x.dtype, device=x.device)
            y_chunk = self._y_patched.to(dtype=x.dtype, device=x.device)
        else:
            K = x.shape[0]
            mask_chunk = self._mask_patched[slice_start : slice_start + K].to(
                dtype=x.dtype, device=x.device
            )
            y_chunk = self._y_patched[slice_start : slice_start + K].to(
                dtype=x.dtype, device=x.device
            )

        with torch.enable_grad():
            y_pred = mask_chunk * x_0
            y_true = mask_chunk * y_chunk

            norm = self.norm
            if callable(norm):
                loss = norm(y_pred, y_true)
            else:
                residual = (y_pred - y_true).reshape(x_0.shape[0], -1)
                loss = residual.abs().pow(norm).sum(dim=1)

            grads = torch.autograd.grad(
                outputs=loss.sum(),
                inputs=x,
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )

        grad_x = grads[0]

        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_bc = t.reshape(expected_shape)
        sigma_t = self.sigma_fn(t_bc)
        alpha_t = self.alpha_fn(t_bc)
        variance = self.std_y**2 + self.gamma * (sigma_t**2) / (alpha_t**2)

        g = -grad_x / (2 * variance)
        if slice_start is None and self.fuse:
            return self.predictor.fuse_fn(g)
        return g
