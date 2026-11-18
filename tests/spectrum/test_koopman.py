# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for Koopman Operator Decomposition Effect."""

import pytest
import torch
import numpy as np

from tam.model.spectrum._koopman import KoopmanEffect
from tam.model.spectrum._factory import create_effects_from_parsed_terms
from tam.common.utils import parse_formula_to_terms


def test_koopman_brunton_kutz_slow_manifold():
    """Verifies exact linearization of the Brunton-Kutz 2D slow manifold ODE."""
    torch.manual_seed(42)
    # Dynamics: x1_dot = mu * x1, x2_dot = lambda_val * (x2 - x1^2)
    # Discrete time: x1_{t+1} = exp(mu*dt) * x1_t
    # x2_{t+1} = exp(lambda_val*dt)*x2_t + (1 - exp(lambda_val*dt))*x1_t^2 ...
    mu = -0.05
    lam = -1.0
    dt = 0.1

    T = 200
    x1 = torch.zeros(T, dtype=torch.float32)
    x2 = torch.zeros(T, dtype=torch.float32)
    x1[0] = 1.0
    x2[0] = 1.0

    for t in range(T - 1):
        x1[t + 1] = x1[t] + dt * (mu * x1[t])
        x2[t + 1] = x2[t] + dt * (lam * (x2[t] - x1[t] ** 2))

    X = torch.stack([x1, x2], dim=-1)

    effect = KoopmanEffect(feature_name=["x1", "x2"], k=6, n_lift=32, delay_embed=1)
    effect.initialize(X)
    phi = effect.build_feature_map(X)
    P = effect.build_penalty_matrix()

    assert phi.shape[0] == T
    assert phi.shape[1] == effect.get_n_coeffs()
    assert P.shape == (effect.get_n_coeffs(), effect.get_n_coeffs())
    assert torch.all(torch.diag(P) >= 1.0)

    # Linear regression from phi onto target x2(t+1)
    # Koopman coordinates should span x2 with high R^2
    target = x2[1:]
    phi_train = phi[:-1]
    w = torch.linalg.lstsq(phi_train, target).solution
    pred = phi_train @ w
    ss_tot = torch.sum((target - target.mean()) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)).item()

    assert r2 > 0.99


def test_koopman_limit_cycle_eigenvalues():
    """Verifies that harmonic oscillator dynamics produce eigenvalues with |mu| approx 1."""
    torch.manual_seed(42)
    dt = 0.05
    omega = 1.5
    t = torch.arange(0, 300 * dt, dt, dtype=torch.float32)
    x = torch.sin(omega * t).unsqueeze(-1)

    effect = KoopmanEffect(feature_name="x", k=8, n_lift=32, delay_embed=4)
    effect.initialize(x)
    phi = effect.build_feature_map(x)
    P = effect.build_penalty_matrix()

    assert phi.shape[0] == len(t)
    assert not torch.isnan(phi).any()
    assert not torch.isnan(P).any()


def test_koopman_causality():
    """Verifies that changing future inputs t > t0 does not alter phi[t0, :]."""
    torch.manual_seed(42)
    T = 60
    x = torch.randn(T, 1, dtype=torch.float32)

    effect = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    effect.initialize(x)

    t0 = 25
    x_perturbed = x.clone()
    x_perturbed[t0 + 1 :] = x_perturbed[t0 + 1 :] + 50.0

    effect_clean = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    effect_clean._W_proj = effect._W_proj.clone()
    effect_clean._penalty_diag = effect._penalty_diag.clone()
    effect_clean._x_mean = effect._x_mean.clone() if effect._x_mean is not None else None
    effect_clean._x_std = effect._x_std.clone() if effect._x_std is not None else None
    phi_clean = effect_clean.build_feature_map(x)

    effect_pert = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    effect_pert._W_proj = effect._W_proj.clone()
    effect_pert._penalty_diag = effect._penalty_diag.clone()
    effect_pert._x_mean = effect._x_mean.clone() if effect._x_mean is not None else None
    effect_pert._x_std = effect._x_std.clone() if effect._x_std is not None else None
    phi_pert = effect_pert.build_feature_map(x_perturbed)

    assert torch.allclose(phi_clean[: t0 + 1], phi_pert[: t0 + 1], atol=1e-5)


def test_koopman_state_continuation():
    """Verifies that chunked execution with state continuation matches batch execution."""
    torch.manual_seed(42)
    x = torch.randn(100, 1, dtype=torch.float32)

    effect_batch = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    effect_batch.initialize(x)
    phi_batch = effect_batch.build_feature_map(x)

    # Chunked evaluation
    effect_chunked = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    effect_chunked._W_proj = effect_batch._W_proj.clone()
    effect_chunked._penalty_diag = effect_batch._penalty_diag.clone()
    effect_chunked._x_mean = effect_batch._x_mean.clone() if effect_batch._x_mean is not None else None
    effect_chunked._x_std = effect_batch._x_std.clone() if effect_batch._x_std is not None else None

    phi_chunk1 = effect_chunked.build_feature_map(x[:60])
    phi_chunk2 = effect_chunked.build_feature_map(x[60:], state_in=effect_chunked._state_out)
    phi_concat = torch.cat([phi_chunk1, phi_chunk2], dim=0)

    assert torch.allclose(phi_batch, phi_concat, atol=1e-5)

    # Also verify lazy initialization inside build_feature_map records the state to carry
    effect_lazy = KoopmanEffect(feature_name="x", k=6, n_lift=24, delay_embed=3)
    _ = effect_lazy.build_feature_map(x[:60])
    assert effect_lazy._state_out["hist"].shape == (1, 2, 1)
    assert (phi_batch.std(dim=0) > 1e-5).all(), "Zero constant columns allowed in KoopmanEffect"


def test_koopman_factory_integration():
    """Verifies formula parsing and factory construction of Koopman effect."""
    formula = "Load ~ koopman(temp, k=8, delay_embed=4, ap=2.0)"
    target, parsed = parse_formula_to_terms(formula)
    effects = create_effects_from_parsed_terms(
        parsed_terms=parsed,
        token_values={},
        default_alpha_p=0.0,
        include_offset=True,
    )

    koopman_effects = [e for e in effects if isinstance(e, KoopmanEffect)]
    assert len(koopman_effects) == 1
    ke = koopman_effects[0]
    assert ke.k == 8
    assert ke.delay_embed == 4
    assert abs(ke.lambda_p - 100.0) < 1e-4


def test_koopman_penalty_scales_with_ap_and_is_symmetric_psd():
    """Verifies Koopman penalty scales by 10 per +1 ap increment and is symmetric PSD."""
    torch.manual_seed(0)
    x = torch.sin(torch.linspace(0.0, 12.0, 120, dtype=torch.float64)).unsqueeze(-1)

    _, parsed_0 = parse_formula_to_terms("Load ~ koopman(x, k=6, delay_embed=2, ap=0)")
    _, parsed_1 = parse_formula_to_terms("Load ~ koopman(x, k=6, delay_embed=2, ap=1)")
    ke_0 = [
        e for e in create_effects_from_parsed_terms(parsed_0, {}, 0.0, include_offset=False)
        if isinstance(e, KoopmanEffect)
    ][0]
    ke_1 = [
        e for e in create_effects_from_parsed_terms(parsed_1, {}, 0.0, include_offset=False)
        if isinstance(e, KoopmanEffect)
    ][0]

    ke_0.initialize(x)
    ke_1.initialize(x)

    p0 = ke_0.build_penalty_matrix()
    p1 = ke_1.build_penalty_matrix()

    assert torch.allclose(p1, 10.0 * p0, rtol=1e-5, atol=1e-6)
    assert torch.allclose(p1, p1.T, atol=1e-8)
    eigvals = torch.linalg.eigvalsh(p1)
    assert torch.all(eigvals >= 0.0)


def test_koopman_has_exactly_k_columns_even_when_the_data_are_rank_deficient():
    """The number of columns is fixed by k: a constant input gets orthonormal filler directions, penalized heavily."""
    x = torch.ones(2, 80, 1, dtype=torch.float64)
    effect = KoopmanEffect("x", k=6, n_lift=24, delay_embed=2)
    effect.initialize(x)
    assert effect.get_n_coeffs() == 6
    assert effect._W_proj.shape[-1] == 6 and effect.build_penalty_matrix().shape == (6, 6)
    assert sum(effect._mode_sizes) == 6
    assert effect.build_feature_map(x).shape == (2, 80, 6)
    W = effect._W_proj
    fillers = [j for j in range(6) if float(effect._penalty_diag[j]) >= KoopmanEffect.FILLER_PENALTY]
    if fillers:
        Wf = W[:, fillers]
        assert torch.allclose(Wf.T @ Wf, torch.eye(len(fillers), dtype=W.dtype), atol=1e-8)


def test_koopman_never_splits_a_conjugate_pair():
    """Complex modes enter as (real, imaginary) neighbours: whatever k, no pair is cut in two."""
    t = torch.arange(0, 400, dtype=torch.float64) * 0.05
    x = (torch.sin(1.5 * t) + 0.5 * torch.sin(0.4 * t + 0.3)).reshape(1, -1, 1)
    for k in (3, 4, 5, 7):
        effect = KoopmanEffect("x", k=k, n_lift=32, delay_embed=4)
        effect.initialize(x)
        assert effect.get_n_coeffs() == k == effect._W_proj.shape[-1]
        assert sum(effect._mode_sizes) == k and set(effect._mode_sizes) <= {1, 2}
        start = 0
        for size in effect._mode_sizes:
            if size == 2:                       # the two parts of a pair carry the same transition residual and |mu|
                assert effect._penalty_diag[start] == effect._penalty_diag[start + 1]
            start += size


def test_koopman_others_are_read_as_columns():
    _, parsed = parse_formula_to_terms("Load ~ koopman(x, others='z|w', k=4, delay_embed=2)")
    effect = [e for e in create_effects_from_parsed_terms(parsed, {}, 0.0, include_offset=False) if isinstance(e, KoopmanEffect)][0]
    assert effect.input_features == ["x", "z", "w"] and effect.multi_input is True
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([effect]) == {"x", "z", "w"}


def test_koopman_groups_are_independent_and_trajectories_never_cross():
    """Snapshot pairs are formed inside a group: the fit is the same whether the groups are stacked or the series concatenated."""
    g = torch.Generator().manual_seed(0)
    x = torch.sin(torch.cumsum(torch.randn(3, 120, 1, dtype=torch.float64, generator=g) * 0.2, 1))
    effect = KoopmanEffect("x", k=5, n_lift=24, delay_embed=3)
    effect.initialize(x)
    whole = effect.build_feature_map(x)
    for b in range(3):
        assert torch.allclose(effect.build_feature_map(x[b:b + 1])[0], whole[b], atol=1e-10)


def test_koopman_inner_dictionary_does_not_depend_on_the_dtype():
    effect = KoopmanEffect("x", k=4, n_lift=40, delay_embed=1)
    effect._setup_lifting_dictionary(2)
    omega = effect._omega.clone()
    assert omega.dtype == torch.float64 and omega.device.type == "cpu"
    other = KoopmanEffect("x", k=4, n_lift=40, delay_embed=1)
    other._setup_lifting_dictionary(2)
    assert torch.equal(omega, other._omega)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_koopman_runs_on_the_gpu():
    x = torch.sin(torch.linspace(0, 12, 200, dtype=torch.float64)).reshape(1, -1, 1).cuda()
    effect = KoopmanEffect("x", k=5, n_lift=24, delay_embed=2)
    effect.initialize(x)
    assert effect.build_feature_map(x).device.type == "cuda"
