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

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import (
    FourierEmbedding,
    Linear,
    PositionalEmbedding,
)
from physicsnemo.experimental.nn import UNetBlock3D, Conv3d, GroupNorm


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class DiffusionUNet3D(Module):
    """
    3D U-Net diffusion backbone for volumetric data generation.

    This architecture extends the DDPM++ and NCSN++ models to 3D volumetric data,
    implementing a U-Net variant with optional self-attention, embeddings, and
    encoder-decoder components for generating 3D volumes.

    The model supports both conditional and unconditional generation with flexible
    architectural choices for encoder/decoder types, embedding types, and attention
    mechanisms. It can be configured for various 3D diffusion tasks including medical
    imaging, scientific simulations, and volumetric content generation.

    Architecture Overview
    ---------------------
    The model processes 3D volumetric inputs through:

    1. **Embedding Generation**: Maps noise levels, class labels, and augmentation
       labels to embeddings that condition the generation process.

    2. **U-Net Encoder**: A hierarchical encoder with multiple levels, where each level:
       - Downsamples spatial resolution by 2x (D, H, W dimensions)
       - Applies ``num_blocks`` residual blocks with conditioning
       - Optionally applies 3D self-attention at specified resolutions
       - Caches features for skip connections

    3. **U-Net Decoder**: Mirror of the encoder that:
       - Upsamples spatial resolution by 2x at each level
       - Combines features via skip connections from encoder
       - Produces the final denoised 3D volume

    Conditioning Mechanism
    ----------------------
    - **Noise labels**: Condition on diffusion timestep/noise level
    - **Class labels**: Optional vector-valued class conditioning
    - **Augmentation labels**: Optional data augmentation conditioning
    - **Image conditioning**: Concatenate conditioning volumes to input channels

    Parameters
    ----------
    img_resolution : Union[List[int], int]
        Spatial resolution of the volumetric data. Can be a single int for uniform
        resolution (D=H=W) or a list [D, H, W] for non-uniform dimensions.
        Note: Model can process different resolutions at inference, except when
        ``additive_pos_embed=True``.
    in_channels : int
        Number of input channels. Includes both latent channels and any additional
        channels for image-based conditioning. For unconditional models, should
        equal ``out_channels``.
    out_channels : int
        Number of output channels. Should match the number of channels in the
        latent state being denoised.
    label_dim : int, optional
        Dimension of vector-valued class labels for conditional generation.
        Set to 0 for unconditional generation, by default 0.
    augment_dim : int, optional
        Dimension of vector-valued augmentation labels. Set to 0 for no
        augmentation conditioning, by default 0.
    model_channels : int, optional
        Base channel multiplier for the network. Determines the number of
        channels at the first level, by default 128.
    channel_mult : List[int], optional
        Channel multipliers at each U-Net level. Length determines the number
        of levels. At level i, channels = ``channel_mult[i] * model_channels``,
        by default [1, 2, 2, 2].
    channel_mult_emb : int, optional
        Multiplier for embedding vector channels. Embedding dimension is
        ``model_channels * channel_mult_emb``, by default 4.
    num_blocks : int, optional
        Number of residual blocks at each U-Net level, by default 4.
    attn_resolutions : List[int], optional
        Spatial resolutions at which to apply 3D self-attention. Attention is
        applied when the feature map resolution matches these values exactly,
        by default [16].
    dropout : float, optional
        Dropout probability for intermediate activations in U-Net blocks,
        by default 0.10.
    label_dropout : float, optional
        Dropout probability for class labels, typically used for classifier-free
        guidance during training, by default 0.0.
    embedding_type : str, optional
        Noise level embedding type. Options: 'positional' (DDPM++), 'fourier'
        (NCSN++), or 'zero' (no embedding), by default "positional".
    channel_mult_noise : int, optional
        Channel multiplier for noise level embeddings. Noise embedding dimension
        is ``model_channels * channel_mult_noise``, by default 1.
    encoder_type : str, optional
        Encoder architecture variant. Options: 'standard' (DDPM++), 'residual'
        (NCSN++), or 'skip' (skip connections), by default "standard".
    decoder_type : str, optional
        Decoder architecture variant. Options: 'standard' or 'skip' (skip
        connections), by default "standard".
    resample_filter : List[int], optional
        1D filter coefficients for resampling operations. Use [1, 1] for DDPM++
        or [1, 3, 3, 1] for NCSN++, by default [1, 1].
    checkpoint_level : int, optional
        Number of levels to use gradient checkpointing. Higher values trade
        memory for computation. 0 disables checkpointing, by default 0.
    additive_pos_embed : bool, optional
        If True, adds learnable positional embeddings encoding spatial position
        (separate from temporal diffusion embeddings). When enabled, input
        resolution must match ``img_resolution``, by default False.

    Raises
    ------
    ValueError
        If ``embedding_type`` is not one of ['fourier', 'positional', 'zero'].
    ValueError
        If ``encoder_type`` is not one of ['standard', 'skip', 'residual'].
    ValueError
        If ``decoder_type`` is not one of ['standard', 'skip'].

    Note
    ----
    This is a 3D extension of the SongUNet architecture. The primary differences
    from the 2D version are:
    - All convolutions and attention operations work on 3D volumes (B, C, D, H, W)
    - Resampling filters are constructed as 3D separable filters
    - Self-attention operates on flattened 3D spatial dimensions

    See Also
    --------
    SongUNet : 2D variant of this architecture for image generation.
    EDMPrecond3D : Preconditioning wrapper for 3D diffusion models.

    References
    ----------
    .. [1] Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion
           probabilistic models. ICML 2021.
    .. [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
           & Poole, B. (2021). Score-based generative modeling through stochastic
           differential equations. ICLR 2021.

    Examples
    --------
    >>> # Create unconditional 3D diffusion model for 64^3 volumes
    >>> model = SongUNet3D(
    ...     img_resolution=64,
    ...     in_channels=4,
    ...     out_channels=4,
    ...     model_channels=128,
    ...     channel_mult=[1, 2, 2, 2],
    ...     num_blocks=4,
    ... )
    >>>
    >>> # Forward pass with noise conditioning
    >>> x = torch.randn(2, 4, 64, 64, 64)  # Noisy volumes
    >>> noise_labels = torch.randn(2, 128)  # Noise level embeddings
    >>> denoised = model(x, noise_labels)
    >>> denoised.shape
    torch.Size([2, 4, 64, 64, 64])
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        super().__init__(meta=MetaData())
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise

        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))

        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Handle image resolution (now 3D)
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_z = self.img_shape_y = self.img_shape_x = img_resolution
        elif len(img_resolution) == 2:
            self.img_shape_y, self.img_shape_x = img_resolution
            self.img_shape_z = img_resolution[0]  # Default to same as y
        else:
            self.img_shape_z, self.img_shape_y, self.img_shape_x = img_resolution[:3]

        # Set checkpoint threshold based on resolution
        max_dimension = max(self.img_shape_x, self.img_shape_y, self.img_shape_z)
        self.checkpoint_threshold = (max_dimension >> checkpoint_level) + 1

        # Optional additive learned position embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(
                    1,
                    model_channels,
                    self.img_shape_z,
                    self.img_shape_y,
                    self.img_shape_x,
                )
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(in_features=label_dim, out_features=noise_channels, **init)
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels, out_features=emb_channels, **init
            )
            self.map_layer1 = Linear(
                in_features=emb_channels, out_features=emb_channels, **init
            )

        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv3d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv3d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv3d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv3d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv3d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv3d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping
        if self.embedding_type != "zero":
            emb = self.map_noise(noise_labels)
            emb = (
                emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            )  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (
                        torch.rand([x.shape[0], 1], device=x.device)
                        >= self.label_dropout
                    ).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = F.silu(self.map_layer0(emb))
            emb = F.silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels), device=x.device
            )

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                if isinstance(block, UNetBlock3D):
                    if x.shape[-1] > self.checkpoint_threshold:
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(F.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                if (x.shape[-1] > self.checkpoint_threshold and "_block" in name) or (
                    x.shape[-1] > (self.checkpoint_threshold / 2) and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)
        return aux
