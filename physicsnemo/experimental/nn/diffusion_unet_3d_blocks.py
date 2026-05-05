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

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn.functional import silu

from einops import rearrange

from physicsnemo.nn.module.fully_connected_layers import Linear
from physicsnemo.nn.module.attention_layers import AttentionOp
from physicsnemo.nn.module.utils.weight_init import _weight_init


class Conv3d(torch.nn.Module):
    """
    3D convolution layer with optional upsampling and downsampling.

    This layer implements a 3D convolution operation with optional bilinear
    resampling (upsampling or downsampling) capabilities. It supports both
    fused and non-fused resampling modes for efficiency.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel : int
        Kernel size for the convolution (applied uniformly across all spatial dimensions).
    bias : bool, optional
        Whether to include a learnable bias term, by default True.
    up : bool, optional
        Whether to apply 2x upsampling before/after convolution, by default False.
    down : bool, optional
        Whether to apply 2x downsampling before/after convolution, by default False.
    resample_filter : List[int], optional
        1D filter coefficients for bilinear resampling, by default [1, 1].
        The 3D filter is constructed as outer product of this 1D filter.
    fused_resample : bool, optional
        Whether to fuse resampling with convolution for efficiency, by default False.
    init_mode : str, optional
        Weight initialization mode, by default "kaiming_normal".
    init_weight : float, optional
        Multiplier for weight initialization, by default 1.0.
    init_bias : float, optional
        Multiplier for bias initialization, by default 0.0.

    Raises
    ------
    ValueError
        If both `up` and `down` are set to True simultaneously.

    Note
    ----
    When `fused_resample=True`, the resampling operation is combined with
    convolution for improved computational efficiency. The resample filter
    is constructed as a 3D separable filter from the 1D coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel * kernel,
            fan_out=out_channels * kernel * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init(
                    [out_channels, in_channels, kernel, kernel, kernel], **init_kwargs
                )
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = (f.ger(f).unsqueeze(2) * f.view(1, 1, -1)).unsqueeze(0).unsqueeze(
            1
        ) / f.sum().pow(3)  # for 3D, should be ^3
        self.register_buffer("resample_filter", f.contiguous() if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose3d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv3d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv3d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv3d(
                x,
                f.tile([self.out_channels, 1, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose3d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv3d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv3d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.

    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if self.training:
            # Use default torch implementation of GroupNorm for training
            # This does not support channels last memory format
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=self.weight.to(x.dtype),
                bias=self.bias.to(x.dtype),
                eps=self.eps,
            )
        else:
            # Use custom GroupNorm implementation that supports channels last
            # memory layout for inference
            dtype = x.dtype
            x = x.float()
            x = rearrange(x, "b (g c) d h w -> b g c d h w", g=self.num_groups)

            mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)  # added 5th dim
            var = x.var(dim=[2, 3, 4, 5], keepdim=True)

            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c d h w -> b (g c) d h w")

            weight = rearrange(self.weight, "c -> 1 c 1 1 1")
            bias = rearrange(self.bias, "c -> 1 c 1 1 1")
            x = x * weight + bias

            x = x.type(dtype)
        return x


class UNetBlock3D(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional
        If True, applies upsampling in the forward pass. By default False.
    down : bool, optional
        If True, applies downsampling in the forward pass. By default False.
    attention : bool, optional
        If True, enables the self-attention mechanism in the block. By default False.
    num_heads : int, optional
        Number of attention heads. If None, defaults to `out_channels // 64`.
    channels_per_head : int, optional
        Number of channels per attention head. By default 64.
    dropout : float, optional
        Dropout probability. By default 0.0.
    skip_scale : float, optional
        Scale factor applied to skip connections. By default 1.0.
    eps : float, optional
        Epsilon value used for normalization layers. By default 1e-5.
    resample_filter : List[int], optional
        Filter for resampling layers. By default [1, 1].
    resample_proj : bool, optional
        If True, resampling projection is enabled. By default False.
    adaptive_scale : bool, optional
        If True, uses adaptive scaling in the forward pass. By default True.
    init : dict, optional
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional
        Initialization parameters with zero weights for certain layers. By default
        {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv3d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        # torch.cuda.nvtx.range_push("UNetBlock3D")
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        # torch.cuda.nvtx.range_pop()
        return x
