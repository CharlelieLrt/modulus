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

from pathlib import Path

import pytest
import torch

import physicsnemo


class MockModel(physicsnemo.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class NewMockModel(physicsnemo.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class MockModelNoOverride(physicsnemo.Module):
    """Fake model"""

    def __init__(self, value1, value2):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))


class MockModelWithOverride(physicsnemo.Module):
    """Fake model"""

    _overridable_args = {"w2"}

    def __init__(self, value1, value2):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("LoadModel", [MockModel, NewMockModel])
def test_from_checkpoint_custom(device, LoadModel):
    """Test checkpointing custom physicsnemo module"""
    torch.manual_seed(0)

    # Construct Mock Model and save it
    mock_model = MockModel().to(device)
    mock_model.save("checkpoint.mdlus")

    # Load from checkpoint using class
    LoadModel.from_checkpoint("checkpoint.mdlus")
    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_checkpoint_override(device):
    """Test checkpointing custom physicsnemo module with override"""
    torch.manual_seed(0)

    # Model with no overrides, loading without overrides
    mock_model = MockModelNoOverride(1, 2).to(device)
    mock_model.save("checkpoint.mdlus")
    MockModelWithOverride.from_checkpoint("checkpoint.mdlus")

    # Model with no overrides, loading with overrides (should fail)
    with pytest.raises(ValueError):
        MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"w2": 3}
        )
    Path("checkpoint.mdlus").unlink(missing_ok=False)

    # Model with overrides, loading without overrides
    mock_model = MockModelWithOverride(1, 2).to(device)
    mock_model.save("checkpoint.mdlus")
    MockModelWithOverride.from_checkpoint("checkpoint.mdlus")
    Path("checkpoint.mdlus").unlink(missing_ok=False)

    # Model with overrides, loading with allowed overrides (should work)
    MockModelWithOverride.from_checkpoint("checkpoint.mdlus", override_args={"w2": 3})
    assert torch.equal(mock_model.w2, torch.tensor(3, dtype=torch.float32))
    Path("checkpoint.mdlus").unlink(missing_ok=False)

    # Model with overrides, loading with disallowed overrides (should fail)
    with pytest.raises(ValueError):
        MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"w1": 3, "w2": 4}
        )
    Path("checkpoint.mdlus").unlink(missing_ok=False)
