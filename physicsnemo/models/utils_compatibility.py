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

import importlib
from typing import Any, Dict, Type

# -- Diffusion UNet prior to 327d9928abc17983ad7aa3df94da9566c197c468 -- #
# For args renaming/deletion
old_to_new_args_UNet_327d9928 = {
    "img_channels": None,
    "sigma_min": None,
    "sigma_max": None,
    "sigma_data": None,
}
# For unpacked kwargs dict
old_to_new_kwargs_UNet_327d9928 = {}  # No unpacked kwargs for UNet

# -- EDMPrecondSuperResolution prior to 327d9928abc17983ad7aa3df94da9566c197c468 -- #
# For args renaming/deletion
old_to_new_args_EDMPrecondSuperResolution_327d9928 = {"img_channels": None}
# For unpacked kwargs dict
old_to_new_kwargs_EDMPrecondSuperResolution_327d9928 = {}
# For class renaming
old_to_new_class_name_EDMPrecondSuperResolution_327d9928 = {
    "__name__": "EDMPrecondSuperResolution"
}


def _update_args(args, old_to_new_args):
    for k, v in old_to_new_args.items():
        if v is not None:
            args[v] = args.pop(k)
        else:
            del args[k]


def _update_init_args(cls: Type, args: Dict[str, Any]):
    """Update arguments passed to instantiation of a class for backward
    compatibility. Handles arguments that have been deprecated or renamed.

    Parameters
    ----------
    - cls : type
        The class to filter arguments for.
    - args : dict
        The arguments passed to cls.__init__ that need to be filtered.
    """
    # Diffusion UNet prior to 327d9928abc17983ad7aa3df94da9566c197c468
    diffusion_module = importlib.import_module("physicsnemo.models.diffusion")
    if cls is diffusion_module.UNet and all(
        k in args for k in old_to_new_args_UNet_327d9928
    ):
        _update_args(args, old_to_new_args_UNet_327d9928)
        return
    # EDMPrecondSuperResolution prior to 327d9928abc17983ad7aa3df94da9566c197c468
    if cls is diffusion_module.EDMPrecondSuperResolution and all(
        k in args for k in old_to_new_args_EDMPrecondSuperResolution_327d9928
    ):
        _update_args(args, old_to_new_args_EDMPrecondSuperResolution_327d9928)
        _update_args(args, old_to_new_kwargs_EDMPrecondSuperResolution_327d9928)
        return


def _update_class_name(arg_dict: Dict[str, Any]):
    """Update the class name of classes that have been renamed.

    Parameters
    ----------
    arg_dict : dict
        The argument dictionary to update. It should contain a "__name__" key
        that represents the class name and and "__args__" key that represents
        the arguments passed to the class constructor.
    """
    # EDMPrecondSuperResolution prior to 327d9928abc17983ad7aa3df94da9566c197c468
    if arg_dict["__name__"] == "EDMPrecondSR" and all(
        k in arg_dict["__args__"]
        for k in old_to_new_args_EDMPrecondSuperResolution_327d9928
    ):
        arg_dict["__name__"] = old_to_new_class_name_EDMPrecondSuperResolution_327d9928[
            "__name__"
        ]
        return
