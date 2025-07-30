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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from physicsnemo.models.diffusion import UNetBlock
from physicsnemo.models.diffusion.layers import Conv2d
from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd
from physicsnemo.models.fno.fno import FNO
from physicsnemo.models.geophysics.elastic_net import _center_crop
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


@torch.no_grad()
def _get_output_dimensions(
    module: torch.nn.Module, input_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Determines the output dimensions by passing a dummy tensor through the module.
    Works with both Conv2D and UNetBlock.
    """
    dummy_input = torch.zeros([1, module.in_channels] + list(input_shape))
    output = module(dummy_input)
    del dummy_input
    return tuple(output.shape[-2:])


@dataclass
class ResNetEncoderMetaData(ModelMetaData):
    """
    Metadata for the ResNetEncoder model.
    """

    name: str = "ResNetEncoder"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = True
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


# TODO: make the embedding number of channels configurable. This will require
# another phase where the resolution does not change and only the number of
# channels changes until reaching the target.
class ResNetEncoder(Module):
    """
    ResNetEncoder model that encodes concatenated seismic data (ux and uz)
    into a latent representation using UNetBlocks.

    Parameters
    ----------
    - nb_sources : int
        Number of input channels for each input image (ux and uz). The
        number of input channels `in_channels` is nb_sources * 2.
    - nb_timesteps : int
        Height of the input images (number of timesteps in the seismic data).
    - nb_receivers : int
        Width of the input image (number of receivers in the seismic data).
    - embedding_dimension : Tuple[int, int]
        Shape of the latent representation (height, width).
    - initial_channels : Optional[int]
        Number of channels after the first convolution. If None, computes as the
        next power of 2 greater than in_channels (2*nb_sources). Default: None
    - checkpointing_level : Optional[int]
        Level of gradient checkpointing to apply. None means no checkpointing.
        Higher values mean checkpointing is applied to more layers. Default: None

    Forward
    -------
    Should be called with `output = model(ux, uz)`.
    Input:
        - ux: torch.Tensor, shape (B, nb_sources, nb_timesteps, nb_receivers)
          Input tensor for u_x component.
        - uz: torch.Tensor, shape (B, nb_sources, nb_timesteps, nb_receivers)
          Input tensor for u_z component.

    Output:
        - torch.Tensor, shape (B, out_channels, embedding_dimension[0], embedding_dimension[1])
          Encoded latent representation.
    """

    def __init__(
        self,
        nb_sources: int,
        nb_timesteps: int,
        nb_receivers: int,
        embedding_dimension: Tuple[int, int],
        initial_channels: Optional[int] = None,
        checkpointing_level: Optional[int] = None,
    ):
        super().__init__()
        self.meta = ResNetEncoderMetaData()
        self.nb_timesteps = nb_timesteps
        self.nb_receivers = nb_receivers
        self.embedding_dimension = embedding_dimension
        self.checkpointing_level = checkpointing_level

        # Double the input channels since we're concatenating ux and uz
        in_channels = nb_sources * 2
        if initial_channels is None:
            initial_channels = 2 ** math.ceil(math.log2(in_channels))
        self.initial_channels = initial_channels

        # Set the threshold for checkpointing based on image resolution
        if self.checkpointing_level is not None:
            self.checkpoint_threshold = (
                max(nb_timesteps, nb_receivers) >> self.checkpointing_level
            ) + 1
        else:
            self.checkpoint_threshold = 0

        self.encoder_blocks, self.out_channels = self._make_encoder(
            in_channels,
            nb_timesteps,
            nb_receivers,
            initial_channels,
            embedding_dimension,
        )

    def _make_encoder(
        self,
        in_channels: int,
        height: int,
        width: int,
        initial_channels: int,
        target_shape: Tuple[int, int],
    ) -> Tuple[nn.ModuleList, int]:
        """
        Create encoder blocks that progressively reduce the spatial dimensions
        until reaching the target shape.

        Parameters
        ----------
        - in_channels : int
            Number of input channels (combined ux and uz).
        - height : int
            Height of the input image.
        - width : int
            Width of the input image.
        - initial_channels : int
            Number of channels after the first convolution.
        - target_shape : Tuple[int, int]
            Target shape for the latent representation.

        Returns
        -------
        - Tuple[nn.ModuleList, int]
            A tuple containing:
            - The encoder blocks as nn.ModuleList
            - The number of output channels in the final encoder block
        """
        encoder_blocks = nn.ModuleList()
        current_shape = (height, width)
        current_channels = in_channels

        initial_conv = Conv2d(
            in_channels=current_channels,
            out_channels=initial_channels,
            kernel=3,
        )
        encoder_blocks.append(initial_conv)
        current_shape = _get_output_dimensions(initial_conv, current_shape)
        current_channels = initial_channels

        # Identify which dimension is larger
        is_height_larger = current_shape[0] > current_shape[1]
        idx_larger = 0 if is_height_larger else 1
        idx_smaller = 1 if is_height_larger else 0

        # Phase 1: Make the image more square by downsampling the largest dimension
        # Also ensure the downsampled dimension doesn't go below target shape
        while (
            current_shape[idx_larger] > 2 * current_shape[idx_smaller]
            and current_shape[idx_larger] > 2 * target_shape[idx_larger]
        ):
            next_channels = current_channels * 2
            encoder_block = UNetBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                emb_channels=0,  # Not used when use_embedding=False
                down=[is_height_larger, not is_height_larger],
                use_embedding=False,
            )
            encoder_blocks.append(encoder_block)
            current_shape = _get_output_dimensions(encoder_block, current_shape)
            current_channels = next_channels

        # Phase 2: If any dimension is below target, upsample it by adding
        # blocks that upsample until we reach the target shape
        while current_shape[0] < target_shape[0] or current_shape[1] < target_shape[1]:
            encoder_block = UNetBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                emb_channels=0,
                up=[
                    current_shape[0] < target_shape[0],
                    current_shape[1] < target_shape[1],
                ],
                use_embedding=False,
            )
            encoder_blocks.append(encoder_block)
            current_shape = _get_output_dimensions(encoder_block, current_shape)

        # Final block after center cropping
        final_block = UNetBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            emb_channels=0,
            use_embedding=False,
        )
        encoder_blocks.append(final_block)

        return encoder_blocks, current_channels

    def checkpointed_forward(self, layer, x):
        """
        Apply gradient checkpointing to a layer if the feature map is large
        enough.
        """
        if self.checkpointing_level is None:
            return layer(x)
        if min(x.shape[-1], x.shape[-2]) > self.checkpoint_threshold:
            return torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, ux: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNetEncoder model.
        """
        x = torch.cat([ux, uz], dim=1)

        for i, encoder in enumerate(self.encoder_blocks[:-1]):
            if i == 0:
                x = self.checkpointed_forward(encoder, x)
            else:
                x = self.checkpointed_forward(encoder, x)

        x = _center_crop(x, self.embedding_dimension[0], self.embedding_dimension[1])

        x = self.checkpointed_forward(self.encoder_blocks[-1], x)

        return x


@dataclass
class DiffusionFWIUNetMetaData(ModelMetaData):
    """
    Metadata for the DiffusionFWIUNet model.
    """

    name: str = "DiffusionFWIUNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


# TODO: enable conditioning concatenation at deeper level of the UNet, insteda
# of input concatenation.
class DiffusionFWIUNet(Module):
    """
    DiffusionUNet combines a ResNetEncoder for seismic data conditioning with a
    SongUNetPosEmbd for diffusion modeling. This model processes seismic data
    (ux and uz) into conditioning embeddings that are concatenated with the
    state vector of the diffusion model, either at the input level or at a
    specified resolution level. Also supports conditioning on the mean
    prediction from a regression model.

    Parameters
    ----------
    - nb_sources : int
        Number of input channels/sources for each input image (ux and uz
        seismic data) used to condition the diffusion model.
    - nb_timesteps : int
        Height of the input images used to condition the diffusion model
        (number of timesteps in the seismic data).
    - nb_receivers : int
        Width of the input image used to condition the diffusion model
        (number of receivers in the seismic data).
    - img_resolution : Union[List[int], int]
        Resolution of the diffusion model state vector. Can be a single int for
        square images or a list [height, width] for rectangular images.
    - x_mean_conditioning: bool
        Whether to condition the diffusion model on the mean prediction from a
        regression model. Default: True.
    - conditioning_model_type: str
        Type of conditioning model to use, either "ResNet" or "FNO". Default: "ResNet".
    - conditioning_model_kwargs: dict
        Additional keyword arguments for the conditioning model. Default: {}.
    - unet_kwargs: dict
        Additional keyword arguments to pass to SongUNetPosEmbd. Default: {}.

    Forward
    -------
    Should be called with `output = model(x, ux, uz, x_mean, noise)`.

    Input:
        - x: torch.Tensor, shape (B, 2, img_resolution[0], img_resolution[1])
          Input tensor to the diffusion model that represents the state vector.
        - ux: torch.Tensor, shape (B, nb_sources, nb_timesteps, nb_receivers)
          Input tensor for u_x velocity component. Used as conditioning.
        - uz: torch.Tensor, shape (B, nb_sources, nb_timesteps, nb_receivers)
          Input tensor for u_z velocity component. Used as conditioning.
        - x_mean: Union[torch.Tensor, None]. Deterministic prediction from a
          regression model. Used as conditioning, only if x_mean_conditioning is
          True. If not None, shape should be (B, 2, img_resolution[0], img_resolution[1]).
        - noise: torch.Tensor, shape (B,)
          Noise level. Used as conditioning.

    Output:
        - torch.Tensor, shape (B, 2, img_resolution[0], img_resolution[1])
          Output tensor from the diffusion model. Represents the updated state vector.

    Note
    ----
    This model uses :class:`physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd` as its
    diffusion backbone. For more details on the diffusion model parameters, refer to the
    SongUNetPosEmbd documentation.
    """

    def __init__(
        self,
        nb_sources: int,
        nb_timesteps: int,
        nb_receivers: int,
        img_resolution: Union[List[int], int],
        x_mean_conditioning: bool = True,
        conditioning_model_type: str = "ResNet",
        conditioning_model_kwargs: dict = {},
        unet_kwargs: dict = {},
    ):
        super().__init__()
        self.meta = DiffusionFWIUNetMetaData()
        self.conditioning_model_type = conditioning_model_type
        self.nb_timesteps = nb_timesteps
        self.nb_receivers = nb_receivers

        if isinstance(img_resolution, int):
            img_resolution = (img_resolution, img_resolution)

        # Set default UNet parameters
        if unet_kwargs is None:
            unet_kwargs = {}
        if "gridtype" not in unet_kwargs:
            unet_kwargs["gridtype"] = "learnable"
        if "N_grid_channels" not in unet_kwargs:
            unet_kwargs["N_grid_channels"] = 32
        if "cond_concat_resolution" not in unet_kwargs:
            unet_kwargs["cond_concat_resolution"] = img_resolution[0]

        self.cond_concat_resolution = unet_kwargs["cond_concat_resolution"]
        self.concat_at_input_level = (
            self.cond_concat_resolution is None
            or self.cond_concat_resolution == img_resolution[0]
        )

        if isinstance(self.cond_concat_resolution, int):
            cond_embedding_dimension = (
                self.cond_concat_resolution,
                self.cond_concat_resolution,
            )
        else:
            cond_embedding_dimension = tuple(self.cond_concat_resolution)
        self.cond_embedding_dimension = cond_embedding_dimension

        self.x_mean_conditioning = x_mean_conditioning

        # Configure the conditioning model
        if conditioning_model_type == "ResNet":
            self.cond_encoder = ResNetEncoder(
                nb_sources=nb_sources,
                nb_timesteps=nb_timesteps,
                nb_receivers=nb_receivers,
                embedding_dimension=cond_embedding_dimension,
                **conditioning_model_kwargs,
            )
            self.cond_channels = self.cond_encoder.out_channels
        elif conditioning_model_type == "FNO":
            out_channels = conditioning_model_kwargs.pop("out_channels", 32)
            # Additional input channels in the conditioning model for
            # positional embeddings
            C_emb = unet_kwargs["N_grid_channels"]
            self.pos_embd_channels = max(
                conditioning_model_kwargs.pop("pos_embd_channels", 16), C_emb
            )
            self.pos_embd_proj = nn.Linear(
                img_resolution[0], self.pos_embd_channels // C_emb
            )
            self.cond_encoder = FNO(
                in_channels=nb_sources * 2 + C_emb * self.pos_embd_proj.out_features,
                out_channels=out_channels,
                dimension=1,
                decoder_layers=conditioning_model_kwargs.pop("decoder_layers", 2),
                decoder_layer_size=conditioning_model_kwargs.pop(
                    "decoder_layer_size", 128
                ),
                latent_channels=conditioning_model_kwargs.pop("latent_channels", 128),
                num_fno_modes=conditioning_model_kwargs.pop("num_fno_modes", 32),
                **conditioning_model_kwargs,
            )
            self.cond_channels = out_channels
        else:
            raise ValueError(
                f"Invalid conditioning_model_type: {conditioning_model_type}. "
                f"Must be one of ['ResNet', 'FNO']"
            )

        # Configure the diffusion backbone
        state_channels = 2  # vp and vs
        if self.concat_at_input_level:
            # Concatenate at input level: include both original input
            # and conditioning channels
            in_channels = (
                state_channels + self.cond_channels + unet_kwargs["N_grid_channels"]
            )
            if self.x_mean_conditioning:
                in_channels += state_channels

            self.diffusion_backbone = SongUNetPosEmbd(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=2,
                **unet_kwargs,
            )
        else:
            # Concatenate at a deeper level in the network
            in_channels = state_channels + unet_kwargs["N_grid_channels"]
            if self.x_mean_conditioning:
                in_channels += state_channels

            self.diffusion_backbone = SongUNetPosEmbd(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=2,
                cond_channels=self.cond_channels,
                **unet_kwargs,
            )

    def forward(
        self,
        x: torch.Tensor,
        ux: torch.Tensor,
        uz: torch.Tensor,
        x_mean: Union[torch.Tensor, None],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the DiffusionFWIUNet model.
        """
        # Note: (H, W) = img_resolution, T = nb_timesteps, R = nb_receivers,
        # S = nb_sources, B = x.shape[0]
        # Generate conditioning embeddings from seismic data
        if self.conditioning_model_type == "ResNet":
            cond_y = self.cond_encoder(ux, uz)
        elif self.conditioning_model_type == "FNO":
            B = ux.shape[0]

            # Process the positional embeddings to add them to the conditioning
            # input
            pos_embd = (
                self.diffusion_backbone.pos_embd.to(ux.device)
                .to(ux.dtype)[None]
                .expand((B, -1, -1, -1))
            )  # (B, C_emb, H, W)
            # Handle dimension mismatch between state width and nb_receivers
            if pos_embd.shape[-1] != self.nb_receivers:
                pos_embd = F.interpolate(
                    pos_embd,
                    size=(pos_embd.shape[-2], self.nb_receivers),
                    mode="bilinear",
                    align_corners=False,
                )  # (B, C_emb, H, R)
            # Reduce dimensionality of positional embeddings with a linear
            # projection
            pos_embd = rearrange(pos_embd, "b c h r -> (b r) c h")  # (B*R, C_emb, H)
            pos_embd = self.pos_embd_proj(
                pos_embd
            )  # (B*R, C_emb, pos_embd_channels // C_emb)
            pos_embd = rearrange(
                pos_embd, "br c h -> br (c h) 1"
            )  # (B*R, pos_embd_channels, 1)
            pos_embd = pos_embd.expand(
                -1, -1, self.nb_timesteps
            )  # (B*R, pos_embd_channels, T)

            # Create full conditioning input: ux, uz, and positional embeddings
            cond_input = torch.cat([ux, uz], dim=1)  # (B, 2*S, T, R)
            cond_input = rearrange(cond_input, "b c t r -> (b r) c t")  # (B*R, 2*S, T)
            cond_input = torch.cat(
                [cond_input, pos_embd], dim=1
            )  # (B*R, 2*S+pos_embd_channels, T)

            # Create the conditioning output
            cond_y = self.cond_encoder(cond_input)  # (B*R, cond_channels, H)
            cond_y = rearrange(
                cond_y, "(b r) c t -> b c t r", b=B, r=self.nb_receivers
            )  # (B, cond_channels, H, R)
            # Handle dimension mismatch for embedding dimension
            if (
                cond_y.shape[2] != self.cond_embedding_dimension[0]
                or cond_y.shape[3] != self.cond_embedding_dimension[1]
            ):
                cond_y = F.interpolate(
                    cond_y,
                    size=self.cond_embedding_dimension,
                    mode="bilinear",
                    align_corners=False,
                )  # (B, cond_channels, H, W)

        # Handle conditioning on x_mean
        if self.x_mean_conditioning and x_mean is None:
            raise ValueError("x_mean tensor required when x_mean_conditioning is True.")
        if not self.x_mean_conditioning and x_mean is not None:
            raise ValueError("x_mean conditioning ignored.")

        if self.concat_at_input_level:
            if self.x_mean_conditioning and x_mean is not None:
                x = torch.cat([x, cond_y, x_mean], dim=1)
            else:
                x = torch.cat([x, cond_y], dim=1)
            output = self.diffusion_backbone(
                x,
                noise,
                class_labels=None,
            )
        else:
            if self.x_mean_conditioning and x_mean is not None:
                x = torch.cat([x, x_mean], dim=1)
            output = self.diffusion_backbone(
                x,
                noise,
                class_labels=None,
                y=cond_y,
            )

        return output
