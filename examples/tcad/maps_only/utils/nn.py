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

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.experimental.models.geotransolver import GeoTransolver
from physicsnemo.experimental.models.geotransolver.gale import GALE_block
from physicsnemo.experimental.models.geotransolver.geotransolver import (
    _normalize_tensor,
)
from physicsnemo.nn.module.embedding_layers import PositionalEmbedding


class TimeCondGALEBlock(GALE_block):
    """GALE block with Poseidon-style AdaLN-Zero modulation.

    Use this drop-in replacement for :class:`GALE_block` when you need every
    transformer block to be modulated by a continuous signal (typically a
    time / step-size embedding). A small per-block MLP maps the signal to
    scale, shift, and gate parameters for the two LayerNorms. The MLP is
    zero-initialized (AdaLN-Zero), so at the start of training the block
    collapses to an identity pass-through and the optimizer gradually learns
    to open the modulation branches.

    Examples
    --------
    >>> block = TimeCondGALEBlock(
    ...     time_emb_dim=128, num_heads=4, hidden_dim=128,
    ...     dropout=0.0, context_dim=16,
    ... )
    >>> fx = (torch.randn(2, 100, 128),)
    >>> time_emb = torch.randn(2, 128)
    >>> ctx = torch.randn(2, 4, 64, 16)
    >>> out = block(fx, time_emb, ctx)
    >>> out[0].shape
    torch.Size([2, 100, 128])

    Parameters
    ----------
    time_emb_dim : int
        Width of the time embedding consumed by the per-block modulation MLP.
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Per-token hidden dimension inside the block.
    dropout : float
        Dropout rate applied by the parent attention/FFN modules.
    act : str, optional
        Activation function name, default ``"gelu"``.
    mlp_ratio : int, optional
        Expansion ratio of the feed-forward layer, default ``4``.
    last_layer : bool, optional
        Marks the final block in the tower, default ``False``.
    out_dim : int, optional
        Output dim used when ``last_layer=True``, default ``1``.
    slice_num : int, optional
        Number of learnable physics-slice tokens, default ``32``.
    plus : bool, optional
        Enable Transolver++ slicing (Gumbel-softmax + per-point temperature),
        default ``False``.
    context_dim : int, optional
        Channel dim of the global context tensor used for cross-attention,
        default ``0``.
    attention_type : str, optional
        ``"GALE"`` or ``"GALE_FA"``, default ``"GALE"``.
    concrete_dropout : bool, optional
        Use learnable concrete dropout, default ``False``.
    """

    def __init__(
        self,
        time_emb_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        plus: bool = False,
        context_dim: int = 0,
        attention_type: str = "GALE",
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            last_layer=last_layer,
            out_dim=out_dim,
            slice_num=slice_num,
            use_te=False,  # AdaLN is incompatible with the fused te.LayerNormMLP
            plus=plus,
            context_dim=context_dim,
            attention_type=attention_type,
            concrete_dropout=concrete_dropout,
        )

        # Split parent's Sequential(LN, Mlp) so modulation lands between them.
        if not isinstance(self.ln_mlp1, nn.Sequential) or len(self.ln_mlp1) != 2:
            raise TypeError(
                f"Expected ln_mlp1 = Sequential(LayerNorm, Mlp); got "
                f"{type(self.ln_mlp1).__name__} with len={len(self.ln_mlp1)}"
            )
        self.ln_2 = self.ln_mlp1[0]
        self.mlp = self.ln_mlp1[1]
        del self.ln_mlp1

        # AdaLN modulation MLP → (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2).
        # Zero-init → block starts as identity (AdaLN-Zero).
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        fx: tuple[Float[Tensor, "B N C"], ...],
        time_emb: Float[Tensor, "B E"],
        global_context: Float[Tensor, "B H S Dc"],
    ) -> list[Float[Tensor, "B N C"]]:
        """Run the modulated block.

        Parameters
        ----------
        fx : tuple of Tensor of shape ``(B, N, C)``
            Per-stream hidden states at this block's input.
        time_emb : Tensor of shape ``(B, E)``
            Embedded conditioning signal produced once per forward pass.
        global_context : Tensor of shape ``(B, H, S, Dc)``
            Shared geometry/global context consumed by the inner GALE
            cross-attention.

        Returns
        -------
        list of Tensor of shape ``(B, N, C)``
            Updated hidden states, same shape as ``fx`` entries.
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(time_emb).chunk(
            6, dim=-1
        )
        gamma1, beta1, alpha1 = (
            v.unsqueeze(1) for v in (gamma1, beta1, alpha1)
        )  # (B, 1, C)
        gamma2, beta2, alpha2 = (
            v.unsqueeze(1) for v in (gamma2, beta2, alpha2)
        )  # (B, 1, C)

        # Attention branch with AdaLN.
        normed = [self.ln_1(_fx) * (1 + gamma1) + beta1 for _fx in fx]
        attn = self.Attn(tuple(normed), global_context)
        fx_out = [fx[i] + alpha1 * attn[i] for i in range(len(fx))]
        if self.attn_dropout is not None:
            fx_out = [self.attn_dropout(_fx) for _fx in fx_out]

        # FFN branch with AdaLN.
        normed2 = [self.ln_2(_fx) * (1 + gamma2) + beta2 for _fx in fx_out]
        mlp_out = [self.mlp(_fx) for _fx in normed2]
        fx_out = [fx_out[i] + alpha2 * mlp_out[i] for i in range(len(fx_out))]
        if self.ffn_dropout is not None:
            fx_out = [self.ffn_dropout(_fx) for _fx in fx_out]
        return fx_out


class TimeConditionedGeoTransolver(GeoTransolver):
    """GeoTransolver conditioned on time + thickness, with an Euler residual head.

    Use this for autoregressive time stepping on point-cloud fields. The model
    computes ``x_{t+1} = x_t[..., :out_dim] + dt * f(x_t, t, thickness, ...)``
    (toggleable via ``use_residual_head``). Time ``t`` is embedded once via a
    learnable :class:`PositionalEmbedding` and used in two places: (a) as the
    AdaLN modulation signal of every :class:`TimeCondGALEBlock`, and (b) as
    one half of the global-context input to the shared GALE cross-attention.
    Thickness is embedded separately with its own learnable
    :class:`PositionalEmbedding` and concatenated to the time embedding to form
    the global-context input — it does **not** feed the AdaLN path. ``dt`` is
    used **only** as the multiplicative scale of the Euler residual; it does
    not condition the model itself.

    Convention for the residual head: the first ``out_dim`` channels of
    ``local_embedding`` must be the predicted variables, in order. For the
    TCAD recipe that is ``[T, V]`` followed by ``[X, Y, Z]`` positions.

    The caller is responsible for normalizing ``t`` and ``thickness`` to
    dimensionless quantities before passing them in (see Notes).

    Examples
    --------
    >>> model = TimeConditionedGeoTransolver(
    ...     functional_dim=5, out_dim=2, geometry_dim=3,
    ...     n_layers=4, n_hidden=128, n_head=4, embed_channels=64,
    ... )
    >>> x = torch.randn(2, 100, 5)
    >>> pos = torch.randn(2, 100, 3)
    >>> t = torch.rand(2)               # already dimensionless: t / t_scale
    >>> dt = torch.rand(2) * 0.1
    >>> thickness = torch.tensor([[0.5], [1.0]])  # already dimensionless
    >>> y = model(x, t=t, dt=dt, thickness=thickness, geometry=pos)
    >>> y.shape
    torch.Size([2, 100, 2])

    Parameters
    ----------
    functional_dim : int or tuple of int
        Number of input features per point (single-stream only in this recipe).
        Must be ``>= out_dim`` so the residual head can slice it.
    out_dim : int or tuple of int
        Number of output features per point. The first ``out_dim`` channels of
        ``local_embedding`` are used as the residual base ``x_t``.
    embed_channels : int, optional
        Output dim shared by the time and thickness :class:`PositionalEmbedding`
        instances. Must be even. The global-context input has width
        ``2 * embed_channels`` (concat of time and thickness embeddings); the
        per-block AdaLN MLP consumes ``embed_channels`` (only the time half).
        Default ``64``.
    geometry_dim : int or None, optional
        Channel count of a per-point geometry tensor, or ``None`` to skip
        geometry projection. Default ``None``.
    n_layers : int, optional
        Number of GALE transformer blocks. Default ``4``.
    n_hidden : int, optional
        Hidden channels inside blocks (must be divisible by ``n_head``).
        Default ``128``.
    n_head : int, optional
        Number of attention heads. Default ``4``.
    slice_num : int, optional
        Number of learnable physics slices per block. Default ``32``.
    mlp_ratio : int, optional
        Expansion ratio of the FFN. Default ``4``.
    dropout : float, optional
        Dropout rate. Default ``0``.
    act : str, optional
        Activation function name. Default ``"gelu"``.
    plus : bool, optional
        Enable Transolver++ slicing. Default ``False``.
    attention_type : str, optional
        ``"GALE"`` or ``"GALE_FA"``. Default ``"GALE"``.
    concrete_dropout : bool, optional
        Use learnable concrete dropout. Default ``False``.
    use_residual_head : bool, optional
        When ``True`` (default), return ``x_t[..., :out_dim] + dt * f``. When
        ``False``, return the raw ``f`` (for ablation only — magnitudes and
        therefore losses are very different between the two modes).

    Notes
    -----
    Caller-side normalization. The caller is responsible for converting the
    raw physical inputs to dimensionless quantities:

    - ``t``        := ``t_raw / t_scale`` (so ``t`` is order O(1))
    - ``thickness``:= ``thickness_raw / coord_std`` (so it is order O(1))

    The model then applies its own internal ``* max_positions`` pre-scale to
    each, which matches the embedding's native input range and pairs with the
    embedding's ``max_positions`` knob (see :class:`PositionalEmbedding`).
    Both ``max_positions`` are hardcoded constants of this class:

    - ``_TIME_MAX_POSITIONS = 100000`` (time spans many decades)
    - ``_THICKNESS_MAX_POSITIONS = 100`` (thickness lands in ~``[1, 10]``)
    """

    # Hardcoded recommended max_positions (and pre-scales) for the two
    # PositionalEmbeddings. Tuned on TCAD distributions; not configurable on
    # purpose — these are part of the architecture.
    _TIME_MAX_POSITIONS = 100000
    _THICKNESS_MAX_POSITIONS = 100

    def __init__(
        self,
        functional_dim: int | tuple[int, ...],
        out_dim: int | tuple[int, ...],
        embed_channels: int = 64,
        geometry_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 128,
        n_head: int = 4,
        slice_num: int = 32,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        act: str = "gelu",
        plus: bool = False,
        attention_type: str = "GALE",
        concrete_dropout: bool = False,
        use_residual_head: bool = True,
    ) -> None:
        if embed_channels % 2 != 0:
            raise ValueError(f"embed_channels must be even, got {embed_channels}")

        # Global context = [time_embed | thickness_embed], so the parent's
        # global_tokenizer must be sized for the concatenated width.
        global_dim = 2 * embed_channels

        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            geometry_dim=geometry_dim,
            global_dim=global_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            use_te=False,  # required by TimeCondGALEBlock (AdaLN vs fused LN+MLP)
            plus=plus,
            attention_type=attention_type,
            concrete_dropout=concrete_dropout,
        )
        self.__name__ = "TimeConditionedGeoTransolver"

        effective_hidden = self.n_hidden
        context_dim = self.context_builder.get_context_dim()
        # AdaLN consumes ONLY the t embedding (thickness is global-context only,
        # dt no longer conditions).
        time_emb_dim = embed_channels

        # Rebuild blocks as time-conditioned variants with the same config.
        self.blocks = nn.ModuleList(
            [
                TimeCondGALEBlock(
                    time_emb_dim=time_emb_dim,
                    num_heads=n_head,
                    hidden_dim=effective_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    last_layer=(i == n_layers - 1),
                    slice_num=slice_num,
                    plus=plus,
                    context_dim=context_dim,
                    attention_type=attention_type,
                    concrete_dropout=concrete_dropout,
                )
                for i in range(n_layers)
            ]
        )

        # Learnable geometric-frequency embeddings for time and thickness.
        self.t_embed = PositionalEmbedding(
            num_channels=embed_channels,
            max_positions=self._TIME_MAX_POSITIONS,
            learnable=True,
        )
        self.thickness_embed = PositionalEmbedding(
            num_channels=embed_channels,
            max_positions=self._THICKNESS_MAX_POSITIONS,
            learnable=True,
        )

        self.use_residual_head = bool(use_residual_head)
        # Residual slice width (single-stream only).
        self.out_dim_total = out_dim if isinstance(out_dim, int) else int(sum(out_dim))

    def forward(
        self,
        local_embedding: Float[Tensor, "B N C"],
        *,
        t: Float[Tensor, " B"],
        dt: Float[Tensor, " B"],
        thickness: Float[Tensor, "B 1"] | Float[Tensor, " B"],
        geometry: Float[Tensor, "B N Cgeo"] | None = None,
        return_embedding_states: bool = False,
    ):
        """Run one time-conditioned forward pass with Euler residual head.

        Parameters
        ----------
        local_embedding : Tensor of shape ``(B, N, C)``
            Per-point input features. The first ``out_dim`` channels are taken
            as the current state ``x_t`` for the residual head.
        t : Tensor of shape ``(B,)``
            Dimensionless current time (caller passes ``t_raw / t_scale``).
            Used for AdaLN and (via the thickness-concatenated global input)
            the cross-attention context.
        dt : Tensor of shape ``(B,)``
            Dimensionless step size used **only** as the multiplicative scale
            of the residual. It does not condition the model.
        thickness : Tensor of shape ``(B, 1)`` or ``(B,)``
            Dimensionless thickness (caller passes ``thickness_raw / coord_std``).
        geometry : Tensor of shape ``(B, N, Cgeo)`` or None, optional
            Per-point geometry features for the geometry tokenizer.
        return_embedding_states : bool, optional
            If True, also return the built context for inspection.

        Returns
        -------
        Tensor of shape ``(B, N, out_dim)``, or ``(output, embedding_states)``
        when ``return_embedding_states=True``.
        """
        if not isinstance(local_embedding, torch.Tensor):
            raise TypeError(
                "TimeConditionedGeoTransolver only supports single-stream input; "
                f"got {type(local_embedding).__name__}"
            )
        local_embedding_tuple = _normalize_tensor(local_embedding)

        # Time embedding (used for both AdaLN and global context).
        t_in = t * self._TIME_MAX_POSITIONS
        t_emb = self.t_embed(t_in)  # (B, embed_channels)

        # Thickness embedding (only in global context).
        thickness_in = thickness.view(-1) * self._THICKNESS_MAX_POSITIONS
        thickness_emb = self.thickness_embed(thickness_in)  # (B, embed_channels)

        # Build the global-context input. Shape (B, 1, 2 * embed_channels).
        global_embedding = torch.cat([t_emb, thickness_emb], dim=-1).unsqueeze(1)

        embedding_states, _ = self.context_builder.build_context(
            local_embedding_tuple, None, geometry, global_embedding
        )

        x = [self.preprocess[0](local_embedding_tuple[0])]
        for block in self.blocks:
            x = block(tuple(x), t_emb, embedding_states)
        f = self.ln_mlp_out[0](x[0])  # (B, N, out_dim)

        if self.use_residual_head:
            # x_{t+1} = x_t[..., :out_dim] + dt * f
            dt_b = dt.view(-1, 1, 1)
            f = local_embedding[..., : self.out_dim_total] + dt_b * f

        if return_embedding_states:
            return f, embedding_states
        return f
