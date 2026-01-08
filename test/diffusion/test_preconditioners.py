# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for diffusion preconditioners."""

from typing import Any, Dict, Tuple

import pytest
import torch

from physicsnemo.core import Module
from physicsnemo.diffusion.preconditioners import (
    BasePreconditioner,
    EDMPreconditioner,
    IDDPMPreconditioner,
    VEPreconditioner,
    VPPreconditioner,
)

from .helpers import (
    compare_outputs,
    generate_batch_data,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
)

# =============================================================================
# Test Model Definition
# =============================================================================


class SimpleModel(Module):
    """Simple model for testing preconditioners with deterministic init."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        # Simple convolution that preserves shape
        self.net = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Constants and Preconditioner Configurations
# =============================================================================


# Default test shape: (batch_size, channels, height, width)
TEST_SHAPE: Tuple[int, ...] = (4, 3, 16, 16)

# Preconditioner configurations for parameterized tests
PRECOND_CONFIGS = [
    (
        VPPreconditioner,
        {"beta_d": 19.9, "beta_min": 0.1, "M": 1000},
        "vp_precond",
    ),
    (
        VEPreconditioner,
        {},
        "ve_precond",
    ),
    (
        IDDPMPreconditioner,
        {"C_1": 0.001, "C_2": 0.008, "M": 100},
        "iddpm_precond",
    ),
    (
        EDMPreconditioner,
        {"sigma_data": 0.5},
        "edm_precond",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model with deterministic parameters."""
    return instantiate_model_deterministic(SimpleModel, seed=0, channels=TEST_SHAPE[1])


@pytest.fixture
def batch_data(device):
    """Create deterministic batch data for testing."""
    return generate_batch_data(shape=TEST_SHAPE, seed=42, device=device)


def create_preconditioner(precond_cls, precond_kwargs):
    """Create a preconditioner with deterministic model."""
    model = instantiate_model_deterministic(SimpleModel, seed=0, channels=TEST_SHAPE[1])
    return precond_cls(model, **precond_kwargs)


# =============================================================================
# VPPreconditioner Tests
# =============================================================================


class TestVPPreconditioner:
    """Tests for VPPreconditioner."""

    @pytest.mark.parametrize(
        "config,beta_d,beta_min,M",
        [
            ("default", 19.9, 0.1, 1000),
            ("custom", 10.0, 0.05, 500),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, beta_d, beta_min, M):
        """Test VPPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = VPPreconditioner(simple_model)
            assert precond.beta_d == 19.9
            assert precond.beta_min == 0.1
            assert precond.M == 1000
        else:
            # Test with custom values - verify against passed arguments
            precond = VPPreconditioner(
                simple_model, beta_d=beta_d, beta_min=beta_min, M=M
            )
            assert precond.beta_d == beta_d
            assert precond.beta_min == beta_min
            assert precond.M == M

        assert precond.model is simple_model
        assert isinstance(precond, BasePreconditioner)

    def test_forward_input_validation(self, simple_model, device):
        """Test forward validates input shapes."""
        precond = VPPreconditioner(simple_model).to(device)
        x = torch.randn(*TEST_SHAPE, device=device)
        t_wrong = torch.rand(2, device=device)  # Wrong batch size

        with pytest.raises(ValueError, match="Expected t to have shape"):
            precond(x, t_wrong, {})


# =============================================================================
# VEPreconditioner Tests
# =============================================================================


class TestVEPreconditioner:
    """Tests for VEPreconditioner."""

    def test_constructor_attributes(self, simple_model):
        """Test VEPreconditioner constructor and attributes."""
        precond = VEPreconditioner(simple_model)

        assert precond.model is simple_model
        assert isinstance(precond, BasePreconditioner)


# =============================================================================
# IDDPMPreconditioner Tests
# =============================================================================


class TestIDDPMPreconditioner:
    """Tests for IDDPMPreconditioner."""

    @pytest.mark.parametrize(
        "config,C_1,C_2,M",
        [
            ("default", 0.001, 0.008, 1000),
            ("custom", 0.002, 0.01, 500),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, C_1, C_2, M):
        """Test IDDPMPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = IDDPMPreconditioner(simple_model)
            assert precond.C_1 == 0.001
            assert precond.C_2 == 0.008
            assert precond.M == 1000
            expected_M = 1000
        else:
            # Test with custom values - verify against passed arguments
            precond = IDDPMPreconditioner(simple_model, C_1=C_1, C_2=C_2, M=M)
            assert precond.C_1 == C_1
            assert precond.C_2 == C_2
            assert precond.M == M
            expected_M = M

        assert hasattr(precond, "u")
        assert precond.u.shape == (expected_M + 1,)
        assert isinstance(precond, BasePreconditioner)


# =============================================================================
# EDMPreconditioner Tests
# =============================================================================


class TestEDMPreconditioner:
    """Tests for EDMPreconditioner."""

    @pytest.mark.parametrize(
        "config,sigma_data",
        [
            ("default", 0.5),
            ("custom", 1.0),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, sigma_data):
        """Test EDMPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = EDMPreconditioner(simple_model)
            assert precond.sigma_data == 0.5
        else:
            # Test with custom values - verify against passed arguments
            precond = EDMPreconditioner(simple_model, sigma_data=sigma_data)
            assert precond.sigma_data == sigma_data

        assert precond.model is simple_model
        assert isinstance(precond, BasePreconditioner)


# =============================================================================
# Non-Regression Tests (Parameterized Across All Preconditioners)
# =============================================================================


@pytest.mark.parametrize(
    "precond_cls,precond_kwargs,name",
    PRECOND_CONFIGS,
    ids=["VP", "VE", "iDDPM", "EDM"],
)
class TestNonRegression:
    """Non-regression tests parameterized across all preconditioner types."""

    def test_sigma_non_regression(self, device, precond_cls, precond_kwargs, name):
        """Test sigma(t) against reference data."""
        precond = create_preconditioner(precond_cls, precond_kwargs).to(device)

        def compute_reference():
            t = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0])
            precond_cpu = create_preconditioner(precond_cls, precond_kwargs)
            sigma = precond_cpu.sigma(t)
            return {"t": t, "sigma": sigma}

        ref_data = load_or_create_reference(f"{name}_sigma.pth", compute_reference)

        t = ref_data["t"].to(device)
        sigma = precond.sigma(t)

        compare_outputs(sigma, ref_data["sigma"], atol=1e-6, rtol=1e-6)

    def test_sigma_from_checkpoint(self, device, precond_cls, precond_kwargs, name):
        """Test sigma(t) from loaded checkpoint matches reference."""

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs)

        precond = load_or_create_checkpoint(f"{name}.mdlus", create_fn).to(device)

        ref_data = load_or_create_reference(f"{name}_sigma.pth", None)

        t = ref_data["t"].to(device)
        sigma = precond.sigma(t)

        compare_outputs(sigma, ref_data["sigma"], atol=1e-6, rtol=1e-6)

    def test_compute_coefficients_non_regression(
        self, device, precond_cls, precond_kwargs, name
    ):
        """Test compute_coefficients against reference data."""
        precond = create_preconditioner(precond_cls, precond_kwargs).to(device)

        def compute_reference():
            sigma = torch.tensor([0.5, 1.0, 2.0, 5.0]).reshape(4, 1, 1, 1)
            precond_cpu = create_preconditioner(precond_cls, precond_kwargs)
            c_in, c_noise, c_out, c_skip = precond_cpu.compute_coefficients(sigma)
            return {
                "sigma": sigma,
                "c_in": c_in,
                "c_noise": c_noise,
                "c_out": c_out,
                "c_skip": c_skip,
            }

        ref_data = load_or_create_reference(
            f"{name}_coefficients.pth", compute_reference
        )

        sigma = ref_data["sigma"].to(device)
        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(sigma)

        compare_outputs(c_in, ref_data["c_in"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_noise, ref_data["c_noise"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_out, ref_data["c_out"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_skip, ref_data["c_skip"], atol=1e-5, rtol=1e-5)

    def test_compute_coefficients_from_checkpoint(
        self, device, precond_cls, precond_kwargs, name
    ):
        """Test compute_coefficients from checkpoint matches reference."""

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs)

        precond = load_or_create_checkpoint(f"{name}.mdlus", create_fn).to(device)

        ref_data = load_or_create_reference(f"{name}_coefficients.pth", None)

        sigma = ref_data["sigma"].to(device)
        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(sigma)

        compare_outputs(c_in, ref_data["c_in"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_noise, ref_data["c_noise"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_out, ref_data["c_out"], atol=1e-5, rtol=1e-5)
        compare_outputs(c_skip, ref_data["c_skip"], atol=1e-5, rtol=1e-5)

    def test_forward_non_regression(self, device, precond_cls, precond_kwargs, name):
        """Test forward pass against reference data."""
        precond = create_preconditioner(precond_cls, precond_kwargs).to(device)

        def compute_reference():
            data = generate_batch_data(shape=TEST_SHAPE, seed=42, device="cpu")
            precond_cpu = create_preconditioner(precond_cls, precond_kwargs)
            out = precond_cpu(data["x"], data["t"], data["condition"])
            return {"x": data["x"], "t": data["t"], "out": out}

        ref_data = load_or_create_reference(f"{name}_forward.pth", compute_reference)

        x = ref_data["x"].to(device)
        t = ref_data["t"].to(device)
        out = precond(x, t, {})

        compare_outputs(out, ref_data["out"], atol=1e-5, rtol=1e-5)

    def test_forward_from_checkpoint(self, device, precond_cls, precond_kwargs, name):
        """Test forward pass from loaded checkpoint matches reference."""

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs)

        precond = load_or_create_checkpoint(f"{name}.mdlus", create_fn).to(device)

        ref_data = load_or_create_reference(f"{name}_forward.pth", None)

        x = ref_data["x"].to(device)
        t = ref_data["t"].to(device)
        out = precond(x, t, {})

        compare_outputs(out, ref_data["out"], atol=1e-5, rtol=1e-5)


# =============================================================================
# Other tests for all preconditioner types
# =============================================================================


@pytest.mark.parametrize(
    "precond_cls,precond_kwargs,name",
    PRECOND_CONFIGS,
    ids=["VP", "VE", "iDDPM", "EDM"],
)
class TestAllPreconditioners:
    """Tests that apply to all preconditioner types."""

    def test_forward_dtype_preservation(
        self, simple_model, batch_data, device, precond_cls, precond_kwargs, name
    ):
        """Test forward preserves input dtype."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]

        output = precond(x, t, condition)

        assert output.dtype == x.dtype

    def test_condition_batch_validation(
        self, simple_model, device, precond_cls, precond_kwargs, name
    ):
        """Test condition tensor batch size validation."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = torch.randn(*TEST_SHAPE, device=device)
        t = torch.rand(TEST_SHAPE[0], device=device)
        # Wrong batch size in condition
        condition = {"cond": torch.randn(2, 10, device=device)}

        with pytest.raises(ValueError, match="batch size"):
            precond(x, t, condition)

    def test_gradient_flow(
        self, simple_model, batch_data, device, precond_cls, precond_kwargs, name
    ):
        """Test gradients flow through the preconditioner."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"].clone().requires_grad_(True)
        t = batch_data["t"]
        condition = batch_data["condition"]

        output = precond(x, t, condition)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
