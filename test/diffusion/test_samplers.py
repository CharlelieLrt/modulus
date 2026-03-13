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

"""Tests for diffusion model sampling interface."""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.samplers.solvers import (
    EulerSolver,
    HeunSolver,
)

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    Conv3dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    gpu_rng_roundtrip,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_samplers_"
BATCH = 2
NUM_STEPS = 4
NUM_STEPS_SHORT = 2

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VENoiseScheduler, {}, "ve"),
    (VPNoiseScheduler, {}, "vp"),
]


class _CustomEulerSolver:
    """User-defined solver implementing the Solver protocol from scratch."""

    def __init__(self, denoiser):
        self.denoiser = denoiser

    def step(self, x, t_cur, t_next):
        t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
        t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))
        d = self.denoiser(x, t_cur)
        return x + (t_next_bc - t_cur_bc) * d


# (solver_key, solver_options, sampler_name, uses_rng)
# "_custom_euler" is handled specially to create a _CustomEulerSolver instance.
SAMPLER_CONFIGS = [
    ("euler", {}, "euler", False),
    ("heun", {}, "heun", False),
    ("heun", {"alpha": 0.5}, "heun_midpoint", False),
    ("_custom_euler", {}, "custom_euler", False),
    (
        "edm_stochastic_euler",
        {"S_churn": 20, "num_steps": NUM_STEPS},
        "stoch_euler",
        True,
    ),
    (
        "edm_stochastic_heun",
        {"S_churn": 20, "num_steps": NUM_STEPS},
        "stoch_heun",
        True,
    ),
]

TIME_EVAL_INDICES = [0, 1, 3]


def _make_sampling_components(
    sched_cls,
    sched_kwargs,
    shape,
    predictor_cls,
    predictor_kwargs,
    device,
    seed=0,
    num_steps=NUM_STEPS,
):
    """Create scheduler, model, denoiser, and initial latents."""
    scheduler = sched_cls(**sched_kwargs)
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    denoiser = scheduler.get_denoiser(x0_predictor=model, denoising_type="ode")
    t_steps = scheduler.timesteps(num_steps, device=device)
    tN = t_steps[0].expand(shape[0])
    xN = make_input(shape, seed=200, device=device) * tN.view(
        -1, *([1] * (len(shape) - 1))
    )
    return scheduler, model, denoiser, xN


def _make_solver_arg(solver_key, solver_options, denoiser):
    """Build the solver argument for sample() from config fields."""
    if solver_key == "_custom_euler":
        return _CustomEulerSolver(denoiser), None
    return solver_key, solver_options or None


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestSampleNonRegression:
    """Non-regression tests for sample() across all sampler configs."""

    def test_sample(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        if "cuda" in str(device) and uses_rng:

            def fn():
                return sample(
                    denoiser,
                    xN,
                    scheduler,
                    NUM_STEPS,
                    solver=solver_arg,
                    solver_options=opts,
                )

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
        elif uses_rng:
            x0 = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
            )
            assert x0.shape == shape
            assert torch.isfinite(x0).all()
        else:
            x0 = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
            )
            assert x0.shape == shape
            assert torch.isfinite(x0).all()
            ref_file = f"{REF_PREFIX}{sampler_name}_{sched_name}_{spatial_name}.pth"
            ref = load_or_create_reference(ref_file, lambda: {"x0": x0.cpu()})
            compare_outputs(x0, ref["x0"], **tolerances)

    def test_sample_with_time_eval(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        if "cuda" in str(device) and uses_rng:

            def fn():
                results = sample(
                    denoiser,
                    xN,
                    scheduler,
                    NUM_STEPS,
                    solver=solver_arg,
                    solver_options=opts,
                    time_eval=TIME_EVAL_INDICES,
                )
                return torch.stack(results)

            stacked = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
        elif uses_rng:
            results = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
                time_eval=TIME_EVAL_INDICES,
            )
            stacked = torch.stack(results)
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
            assert torch.isfinite(stacked).all()
        else:
            results = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=solver_arg,
                solver_options=opts,
                time_eval=TIME_EVAL_INDICES,
            )
            stacked = torch.stack(results)
            assert stacked.shape == (len(TIME_EVAL_INDICES), *shape)
            assert torch.isfinite(stacked).all()
            ref_file = (
                f"{REF_PREFIX}{sampler_name}_{sched_name}_{spatial_name}_teval.pth"
            )
            ref = load_or_create_reference(ref_file, lambda: {"stacked": stacked.cpu()})
            compare_outputs(stacked, ref["stacked"], **tolerances)


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestSampleConsistency:
    """Tests that equivalent argument combinations produce identical results."""

    def test_time_steps_vs_num_steps(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing explicit time_steps from scheduler.timesteps(N) should match
        passing num_steps=N to let sample() generate them internally."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )
        t_steps = scheduler.timesteps(NUM_STEPS_SHORT, device=device, dtype=xN.dtype)

        x0_via_num_steps = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
        x0_via_time_steps = sample(
            denoiser,
            xN,
            scheduler,
            num_steps=0,
            time_steps=t_steps,
            solver="euler",
        )
        compare_outputs(x0_via_time_steps, x0_via_num_steps, atol=1e-6, rtol=1e-6)

    def test_solver_string_vs_instance(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing solver="euler" should match passing solver=EulerSolver(denoiser)."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )

        x0_via_string = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
        x0_via_instance = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=EulerSolver(denoiser),
        )
        compare_outputs(x0_via_instance, x0_via_string, atol=1e-6, rtol=1e-6)

    def test_solver_options_vs_instance(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Passing solver="heun" + solver_options={"alpha": 0.5} should match
        passing solver=HeunSolver(denoiser, alpha=0.5)."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )

        x0_via_options = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="heun",
            solver_options={"alpha": 0.5},
        )
        x0_via_instance = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=HeunSolver(denoiser, alpha=0.5),
        )
        compare_outputs(x0_via_instance, x0_via_options, atol=1e-6, rtol=1e-6)

    def test_custom_solver_vs_euler(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """User-defined _CustomEulerSolver should match built-in EulerSolver."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )

        x0_builtin = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver="euler",
        )
        x0_custom = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=_CustomEulerSolver(denoiser),
        )
        compare_outputs(x0_custom, x0_builtin, **tolerances)


# =============================================================================
# Validation / Error Tests
# =============================================================================


class TestSampleValidation:
    """Tests for sample() argument validation and error handling."""

    def test_solver_options_with_instance_raises(self, device):
        shape = (BATCH, 3, 8, 6)
        scheduler, _, denoiser, xN = _make_sampling_components(
            EDMNoiseScheduler,
            {},
            shape,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
        )
        with pytest.raises(ValueError, match="solver_options"):
            sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS,
                solver=EulerSolver(denoiser),
                solver_options={"alpha": 0.5},
            )

    def test_unknown_solver_string_raises(self, device):
        shape = (BATCH, 3, 8, 6)
        scheduler, _, denoiser, xN = _make_sampling_components(
            EDMNoiseScheduler,
            {},
            shape,
            Conv2dX0Predictor,
            {"channels": 3},
            device,
        )
        with pytest.raises(ValueError, match="Unknown solver"):
            sample(denoiser, xN, scheduler, NUM_STEPS, solver="nonexistent")


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestSampleCompile:
    """torch.compile tests: compiled denoiser passed to sample()."""

    def test_compiled_denoiser_in_sample(
        self,
        deterministic_settings,
        device,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Sampling with a compiled denoiser matches eager sampling."""
        scheduler, _, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )
        compiled_denoiser = torch.compile(denoiser, fullgraph=True)

        solver_eager, opts_eager = _make_solver_arg(
            solver_key,
            solver_options,
            denoiser,
        )
        solver_compiled, opts_compiled = _make_solver_arg(
            solver_key,
            solver_options,
            compiled_denoiser,
        )

        with torch.no_grad():
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_eager = sample(
                denoiser,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_eager,
                solver_options=opts_eager,
            )
            torch.manual_seed(GLOBAL_SEED)
            if "cuda" in str(device):
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            x0_compiled = sample(
                compiled_denoiser,
                xN,
                scheduler,
                NUM_STEPS_SHORT,
                solver=solver_compiled,
                solver_options=opts_compiled,
            )
        torch.testing.assert_close(x0_eager, x0_compiled, atol=1e-3, rtol=1e-3)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_key,solver_options,sampler_name,uses_rng",
    SAMPLER_CONFIGS,
    ids=[c[2] for c in SAMPLER_CONFIGS],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestGradientFlow:
    """Tests that gradients flow through the sampling loop to model parameters."""

    def test_backward_through_sampling(
        self,
        device,
        solver_key,
        solver_options,
        sampler_name,
        uses_rng,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        scheduler, model, denoiser, xN = _make_sampling_components(
            sched_cls,
            sched_kwargs,
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
            num_steps=NUM_STEPS_SHORT,
        )
        solver_arg, opts = _make_solver_arg(solver_key, solver_options, denoiser)

        x0 = sample(
            denoiser,
            xN,
            scheduler,
            NUM_STEPS_SHORT,
            solver=solver_arg,
            solver_options=opts,
        )
        loss = x0.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad
