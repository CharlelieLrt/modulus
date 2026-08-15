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

"""Tests for diffusion ODE/SDE solvers."""

import math

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler, VPNoiseScheduler
from physicsnemo.diffusion.samplers import (
    EDMStochasticEulerSolver,
    EDMStochasticHeunSolver,
    EulerSolver,
    ExponentialAB2Solver,
    HeunSolver,
    Solver,
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

REF_PREFIX = "test_solvers_"
BATCH = 2

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

# (solver_cls, solver_kwargs, solver_name, uses_rng)
# The solver constructor receives solver_kwargs after `denoiser`.
# Keys starting with "_use_" are sentinels handled by _make_solver: they
# select EDM schedule callbacks, the linear coefficient of the semi-linear
# split, or the DPM-Solver++(2M) change of variables.
SOLVER_CONFIGS = [
    (EulerSolver, {}, "euler", False),
    (HeunSolver, {}, "heun", False),
    (HeunSolver, {"alpha": 0.5}, "heun_midpoint", False),
    (EDMStochasticEulerSolver, {"S_churn": 0}, "stoch_euler_nochurn", False),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_euler_churn",
        True,
    ),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10, "_use_edm_sigma_fns": True},
        "stoch_euler_sigmafns",
        True,
    ),
    (EDMStochasticHeunSolver, {"S_churn": 0}, "stoch_heun_nochurn", False),
    (
        EDMStochasticHeunSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_heun_churn",
        True,
    ),
    (
        ExponentialAB2Solver,
        {"_use_linear_coefficient": True},
        "exponential_ab2",
        False,
    ),
    (
        ExponentialAB2Solver,
        {"_use_dpmpp_chart": True},
        "exponential_ab2_dpmpp",
        False,
    ),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 0, "renoise": 1.0},
        "stoch_euler_renoise",
        True,
    ),
]


def _make_denoiser(shape, predictor_cls, predictor_kwargs, device, seed=0):
    """Create a deterministic ODE denoiser from an x0-predictor via EDM scheduler."""
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    scheduler = EDMNoiseScheduler()
    return scheduler.get_denoiser(x0_predictor=model, denoising_type="ode"), model


def _identity_denoiser(x, t):
    return x


def _minus_one_coeff(t):
    return -torch.ones_like(t)


def _make_solver(
    solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device, seed=0
):
    """Create a solver with its denoiser, resolving "_use_*" sentinels."""
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    scheduler = EDMNoiseScheduler()
    kwargs = dict(solver_kwargs)
    if kwargs.pop("_use_edm_sigma_fns", False):
        kwargs["sigma_fn"] = scheduler.sigma
        kwargs["sigma_inv_fn"] = scheduler.sigma_inv
        kwargs["diffusion_fn"] = scheduler.diffusion
    if kwargs.pop("_use_linear_coefficient", False):
        kwargs["linear_fn"] = scheduler.get_linear_denoiser(x0_predictor=model)
    if kwargs.pop("_use_dpmpp_chart", False):
        # DPM-Solver++(2M): x/alpha state, half log-SNR clock, linear
        # coefficient from the scheduler, midpoint slope weight
        kwargs["linear_fn"] = scheduler.get_linear_denoiser(x0_predictor=model)
        kwargs.setdefault("slope_variant", "midpoint")
        kwargs["x_scale_fn"] = scheduler.alpha
        kwargs["x_scale_dot_fn"] = scheduler.alpha_dot
        kwargs["time_fn"] = lambda t: torch.log(scheduler.alpha(t) / scheduler.sigma(t))
        kwargs["time_dot_fn"] = lambda t: (
            scheduler.alpha_dot(t) / scheduler.alpha(t)
            - scheduler.sigma_dot(t) / scheduler.sigma(t)
        )
    denoiser = scheduler.get_denoiser(x0_predictor=model, denoising_type="ode")
    return solver_cls(denoiser, **kwargs)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestEulerSolverConstructor:
    """Tests for EulerSolver constructor."""

    def test_attributes(self):
        solver = EulerSolver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)

    def test_chart_callback_validation(self):
        with pytest.raises(ValueError, match="x_scale_fn and x_scale_dot_fn"):
            EulerSolver(_identity_denoiser, x_scale_fn=lambda t: t)
        with pytest.raises(ValueError, match="time_fn and time_dot_fn"):
            EulerSolver(_identity_denoiser, time_dot_fn=lambda t: t)


class TestHeunSolverConstructor:
    """Tests for HeunSolver constructor."""

    def test_default_alpha(self):
        solver = HeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)

    def test_custom_alpha(self):
        solver = HeunSolver(_identity_denoiser, alpha=0.5)
        assert solver.alpha == pytest.approx(0.5)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=1.5)


class TestEDMStochasticEulerSolverConstructor:
    """Tests for EDMStochasticEulerSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticEulerSolver(_identity_denoiser)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.S_noise == pytest.approx(1.0)
        assert solver.num_steps == 18
        assert solver.renoise == pytest.approx(0.0)

    def test_sigma_fn_validation(self):
        def sigma_only(t):
            return t

        with pytest.raises(ValueError, match="sigma_fn and sigma_inv_fn"):
            EDMStochasticEulerSolver(_identity_denoiser, sigma_fn=sigma_only)

    def test_invalid_renoise(self):
        with pytest.raises(ValueError, match="renoise"):
            EDMStochasticEulerSolver(_identity_denoiser, renoise=1.5)
        with pytest.raises(ValueError, match="renoise"):
            EDMStochasticEulerSolver(_identity_denoiser, renoise=-0.1)


class TestEDMStochasticHeunSolverConstructor:
    """Tests for EDMStochasticHeunSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticHeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.renoise == pytest.approx(0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            EDMStochasticHeunSolver(_identity_denoiser, alpha=0.0)

    def test_invalid_renoise(self):
        with pytest.raises(ValueError, match="renoise"):
            EDMStochasticHeunSolver(_identity_denoiser, renoise=1.5)


class TestExponentialAB2SolverConstructor:
    """Tests for ExponentialAB2Solver constructor."""

    def test_default_attributes(self):
        solver = ExponentialAB2Solver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)
        assert solver.slope_variant == "heun"
        t = torch.tensor([2.0, 3.0])
        assert torch.all(solver.linear_fn(t) == 0)
        assert torch.all(solver.x_scale_fn(t) == 1.0)
        assert torch.all(solver.time_fn(t) == t)

    def test_custom_linear_fn(self):
        solver = ExponentialAB2Solver(_identity_denoiser, linear_fn=_minus_one_coeff)
        assert solver.linear_fn is _minus_one_coeff


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestStepNonRegression:
    """Non-regression tests for solver step() across all solver configs."""

    def test_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        solver = _make_solver(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        ref_file = f"{REF_PREFIX}{solver_name}_{spatial_name}_step.pth"
        if "cuda" in str(device) and uses_rng:

            def fn():
                return solver.step(x, t_cur, t_next)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["x_next"].shape
        else:
            x_next = solver.step(x, t_cur, t_next)
            assert x_next.shape == shape
            ref = load_or_create_reference(ref_file, lambda: {"x_next": x_next.cpu()})
            compare_outputs(x_next, ref["x_next"], **tolerances)

    def test_step_to_zero(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Step to t=0 should produce finite output."""
        solver = _make_solver(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )

        x = make_input(shape, seed=101, device=device)
        t_cur = torch.tensor([1.0] * shape[0], device=device)
        t_next = torch.tensor([0.0] * shape[0], device=device)

        x_next = solver.step(x, t_cur, t_next)
        assert x_next.shape == shape
        assert torch.isfinite(x_next).all()

    def test_zero_churn_matches_deterministic(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Stochastic solvers with S_churn=0 should match their deterministic counterpart."""
        if solver_name == "stoch_euler_nochurn":
            det_cls = EulerSolver
        elif solver_name == "stoch_heun_nochurn":
            det_cls = HeunSolver
        else:
            pytest.skip("Only applies to zero-churn stochastic configs")

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        stoch_solver = _make_solver(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )
        det_solver = det_cls(denoiser)

        x = make_input(shape, seed=120, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_stoch = stoch_solver.step(x, t_cur, t_next)
        x_det = det_solver.step(x, t_cur, t_next)
        compare_outputs(x_stoch, x_det, **tolerances)


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestConsistency:
    """Equivalences between solvers and between solver and scheduler methods."""

    def test_ab2_reset_restores_first_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """reset() clears the history: the next step reproduces a fresh
        instance's first step."""
        solver = _make_solver(
            ExponentialAB2Solver,
            {"_use_dpmpp_chart": True},
            shape,
            predictor_cls,
            predictor_kwargs,
            device,
        )

        x = make_input(shape, seed=133, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_first = solver.step(x, t_cur, t_next)
        solver.step(x_first, t_next, torch.tensor([1.0] * shape[0], device=device))
        solver.reset()
        x_after_reset = solver.step(x, t_cur, t_next)
        compare_outputs(x_after_reset, x_first, **tolerances)

    def test_chart_euler_matches_ddim_on_vp(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """On a VP schedule, the Euler step on the probability-flow ODE
        right-hand side under the (alpha, sigma/alpha) change of variables
        equals the DDIM update, expressed through the scheduler conversion
        methods."""
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        scheduler = VPNoiseScheduler()

        def ntsr(t):
            return scheduler.sigma(t) / scheduler.alpha(t)

        def ntsr_dot(t):
            return ntsr(t) * (
                scheduler.sigma_dot(t) / scheduler.sigma(t)
                - scheduler.alpha_dot(t) / scheduler.alpha(t)
            )

        ddim = EulerSolver(
            scheduler.get_denoiser(x0_predictor=model),
            x_scale_fn=scheduler.alpha,
            x_scale_dot_fn=scheduler.alpha_dot,
            time_fn=ntsr,
            time_dot_fn=ntsr_dot,
        )

        x = make_input(shape, seed=134, device=device)
        t_cur = torch.tensor([0.6] * shape[0], device=device)
        t_next = torch.tensor([0.3] * shape[0], device=device)

        x0 = model(x, t_cur)
        eps = scheduler.x0_to_epsilon(x0, x, t_cur)
        expected_shape = (-1,) + (1,) * (x.ndim - 1)
        t_next_bc = t_next.reshape(expected_shape)
        x_ddim = scheduler.alpha(t_next_bc) * x0 + scheduler.sigma(t_next_bc) * eps
        compare_outputs(ddim.step(x, t_cur, t_next), x_ddim, **tolerances)

    def test_renoise_full_restart_returns_data_prediction_at_zero_noise(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """At t_next = 0 the arrival noise level is zero, so the fully
        re-noised step returns the data prediction exactly."""
        denoiser, model = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = EDMStochasticEulerSolver(denoiser, S_churn=0, renoise=1.0)

        x = make_input(shape, seed=135, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([0.0] * shape[0], device=device)
        compare_outputs(solver.step(x, t_cur, t_next), model(x, t_cur), **tolerances)

    def test_dpmpp_chart_variants_share_first_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Both DPM-Solver++(2M) slope variants coincide on the first step,
        where no history exists."""
        common = (shape, predictor_cls, predictor_kwargs, device)
        midpoint = _make_solver(
            ExponentialAB2Solver, {"_use_dpmpp_chart": True}, *common
        )
        heun = _make_solver(
            ExponentialAB2Solver,
            {"_use_dpmpp_chart": True, "slope_variant": "heun"},
            *common,
        )

        x = make_input(shape, seed=136, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)
        compare_outputs(
            midpoint.step(x, t_cur, t_next), heun.step(x, t_cur, t_next), **tolerances
        )


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestDiagnostics:
    """Order-of-accuracy diagnostics on a closed-form semi-linear ODE.

    The exponential solver is a general-purpose integrator: the checks below
    integrate dx/dt = -x + cos(t), passing the full RHS as the denoiser with
    linear coefficient -1. The exact solution is
    x(t) = C exp(-t) + (cos(t) + sin(t))/2.
    """

    @staticmethod
    def _full_rhs(x, t):
        return -x + torch.cos(t).view(-1, *([1] * (x.ndim - 1))).expand_as(x)

    def _integrate(self, solver, x_init, num_steps):
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
        x = x_init.clone()
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x = solver.step(x, t_cur.expand(1), t_next.expand(1))
        return x

    def _errors(self, make_solver, x_init, x_exact):
        return [
            abs(self._integrate(make_solver(), x_init, n).item() - x_exact)
            for n in (10, 20, 40)
        ]

    def test_convergence_orders(self):
        x_init = torch.tensor([[2.0]], dtype=torch.float64)
        c_int = (x_init.item() - (math.cos(1.0) + math.sin(1.0)) / 2) * math.exp(1.0)
        x_exact = c_int + 0.5  # Exact solution at t = 0

        errs_ab2 = self._errors(
            lambda: ExponentialAB2Solver(self._full_rhs, linear_fn=_minus_one_coeff),
            x_init,
            x_exact,
        )

        orders_ab2 = [math.log2(errs_ab2[i] / errs_ab2[i + 1]) for i in range(2)]

        assert min(orders_ab2) > 1.7

    def test_ab2_exact_for_linear_ode(self):
        """With a zero nonlinear part and constant linear coefficient, each
        step integrates the ODE exactly regardless of the step size."""
        solver = ExponentialAB2Solver(lambda x, t: -x, linear_fn=_minus_one_coeff)
        x_init = torch.tensor([[2.0]], dtype=torch.float64)
        x_final = self._integrate(solver, x_init, 4)
        assert x_final.item() == pytest.approx(2.0 * math.exp(1.0), rel=1e-12)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
@pytest.mark.usefixtures("nop_compile")
class TestStepCompile:
    """Double-call compile tests for solver step()."""

    def test_compiled_step(
        self,
        deterministic_settings,
        device,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Compiled step traces without error and the second call reuses the graph."""
        torch._dynamo.config.error_on_recompile = True

        solver = _make_solver(
            solver_cls, solver_kwargs, shape, predictor_cls, predictor_kwargs, device
        )

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        # Multistep solvers take a history-free branch on their first step and
        # a history branch afterwards. Prime the history eagerly so the
        # compiled graph traces only the steady-state branch, and re-prime
        # identically before each call so the results are comparable.
        x_prev = make_input(shape, seed=99, device=device)
        t_prev = torch.tensor([7.5] * shape[0], device=device)

        def prime(s):
            if isinstance(s, ExponentialAB2Solver):
                s.reset()
                with torch.no_grad():
                    s.step(x_prev, t_prev, t_cur)

        compiled_step = torch.compile(solver.step, fullgraph=True)

        prime(solver)
        with torch.no_grad():
            out_compiled = compiled_step(x, t_cur, t_next)
        assert out_compiled.shape == shape
        assert torch.isfinite(out_compiled).all()

        # Second call — must reuse the graph
        prime(solver)
        with torch.no_grad():
            out_compiled_2 = compiled_step(x, t_cur, t_next)
        assert out_compiled_2.shape == shape
        assert torch.isfinite(out_compiled_2).all()

        # For deterministic solvers, also verify eager-vs-compiled match
        if not uses_rng:
            prime(solver)
            with torch.no_grad():
                out_eager = solver.step(x, t_cur, t_next)
            torch.testing.assert_close(out_eager, out_compiled)
