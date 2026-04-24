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
from physicsnemo.nn.module.embedding_layers import SinusoidalTimestepEmbedding


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
    """GeoTransolver conditioned on continuous time via per-block AdaLN.

    Use this when your task is unsteady (samples carry a current time and step
    size) and you want the model to stay aware of where it is on the
    trajectory. The current time ``t`` and the step size ``dt`` are passed
    through a :class:`SinusoidalTimestepEmbedding` each, concatenated, and fed
    to a per-block AdaLN MLP that modulates LayerNorms and residual gates.
    The raw ``(t, dt)`` pair is also expected to enter the model through
    ``global_embedding`` so that the shared GALE context carries the same
    signal as the modulation.

    The modulation MLPs are zero-initialized, so the model starts at
    parent-identity and the time signal is learned gradually.

    Examples
    --------
    >>> model = TimeConditionedGeoTransolver(
    ...     functional_dim=2, out_dim=2, global_dim=2,
    ...     n_layers=4, n_hidden=128, n_head=4,
    ...     time_embed_channels=64,
    ... )
    >>> x = torch.randn(2, 100, 2)
    >>> pos = torch.randn(2, 100, 3)
    >>> glob = torch.randn(2, 1, 2)
    >>> t = torch.rand(2)
    >>> dt = torch.rand(2) * 0.1
    >>> y = model(x, pos, glob, t=t, dt=dt)
    >>> y.shape
    torch.Size([2, 100, 2])

    Parameters
    ----------
    functional_dim : int or tuple of int
        Number of input features per point. Tuple enables multi-stream.
    out_dim : int or tuple of int
        Number of output features per point; must match the length of
        ``functional_dim`` when tuples are used.
    time_embed_channels : int, optional
        Output dim of each :class:`SinusoidalTimestepEmbedding` (one for
        ``t``, one for ``dt``). Must be even. The per-block MLP consumes
        ``2 * time_embed_channels`` after concatenation. Default ``64``.
    geometry_dim : int or None, optional
        Channel count of a per-point geometry tensor, or ``None`` to skip
        geometry projection. Default ``None``.
    global_dim : int or None, optional
        Channel count of a per-sample global tensor, or ``None`` to skip
        global projection. Default ``None``.
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
    """

    def __init__(
        self,
        functional_dim: int | tuple[int, ...],
        out_dim: int | tuple[int, ...],
        time_embed_channels: int = 64,
        geometry_dim: int | None = None,
        global_dim: int | None = None,
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
    ) -> None:
        if time_embed_channels % 2 != 0:
            raise ValueError(
                f"time_embed_channels must be even, got {time_embed_channels}"
            )

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
        time_emb_dim = 2 * time_embed_channels

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

        self.t_embed = SinusoidalTimestepEmbedding(num_channels=time_embed_channels)
        self.dt_embed = SinusoidalTimestepEmbedding(num_channels=time_embed_channels)

    def forward(
        self,
        local_embedding: (Float[Tensor, "B N C"] | tuple[Float[Tensor, "B N C"], ...]),
        local_positions: (
            Float[Tensor, "B N 3"] | tuple[Float[Tensor, "B N 3"], ...] | None
        ) = None,
        global_embedding: Float[Tensor, "B G Cg"] | None = None,
        *,
        t: Float[Tensor, " B"],
        dt: Float[Tensor, " B"],
        geometry: Float[Tensor, "B N Cgeo"] | None = None,
        return_embedding_states: bool = False,
    ):
        """Run one time-conditioned forward pass.

        Parameters
        ----------
        local_embedding : Tensor of shape ``(B, N, C)`` or tuple of such
            Per-point input features (single-stream or multi-stream).
        local_positions : Tensor of shape ``(B, N, 3)`` or tuple or None, optional
            Per-point spatial coordinates.
        global_embedding : Tensor of shape ``(B, G, Cg)`` or None, optional
            Per-sample global features. The training loop passes
            ``[t_norm, dt_norm]`` here so the shared context also carries the
            time signal.
        t : Tensor of shape ``(B,)``
            Normalized current time, one scalar per sample.
        dt : Tensor of shape ``(B,)``
            Normalized step size, one scalar per sample.
        geometry : Tensor of shape ``(B, N, Cgeo)`` or None, optional
            Per-point geometry features.
        return_embedding_states : bool, optional
            If True, also return the built context for inspection.

        Returns
        -------
        Tensor of shape ``(B, N, out_dim)``, or tuple of such for multi-stream,
        or ``(output, embedding_states)`` when ``return_embedding_states=True``.
        """
        single_input = isinstance(local_embedding, torch.Tensor)
        local_embedding = _normalize_tensor(local_embedding)
        if local_positions is not None:
            local_positions = _normalize_tensor(local_positions)

        time_emb = torch.cat([self.t_embed(t), self.dt_embed(dt)], dim=-1)
        # (B, 2 * time_embed_channels)

        embedding_states, _ = self.context_builder.build_context(
            local_embedding, local_positions, geometry, global_embedding
        )

        x = [self.preprocess[i](le) for i, le in enumerate(local_embedding)]

        for block in self.blocks:
            x = block(tuple(x), time_emb, embedding_states)

        x = [self.ln_mlp_out[i](x[i]) for i in range(len(x))]

        if single_input:
            x = x[0]
        else:
            x = tuple(x)
        if return_embedding_states:
            return x, embedding_states
        return x
