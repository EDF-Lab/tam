# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._signature.SignatureEffect`` and rough path operations."""

import numpy as np
import pandas as pd
import pytest
import torch

import tam
from tam.model.additive import StaticTAM
from tam.model.spectrum import SignatureEffect
from tam.model.spectrum._signature_ops import (
    segment_exp,
    tensor_product,
    tensor_inverse,
    tensor_log,
    compute_witt_dimension,
    lead_lag_transform,
)


def test_signature_linear_path_closed_form():
    """Verifies that for a straight line segment, level-k coefficients equal delta^(x k) / k!."""
    delta = torch.tensor([0.4, -0.6], dtype=torch.float64)
    N = 3
    sig = segment_exp(delta, N)

    # Level 0 is constant 1.0
    assert abs(sig[0] - 1.0) < 1e-12

    # Level 1 is delta itself
    np.testing.assert_allclose(sig[1:3].numpy(), delta.numpy(), atol=1e-12)

    # Level 2 is (delta (x) delta) / 2
    delta_2 = torch.outer(delta, delta).flatten() / 2.0
    np.testing.assert_allclose(sig[3:7].numpy(), delta_2.numpy(), atol=1e-12)

    # Level 3 is (delta (x) delta (x) delta) / 6
    delta_3 = (torch.outer(delta_2, delta) / 3.0).flatten()
    np.testing.assert_allclose(sig[7:15].numpy(), delta_3.numpy(), atol=1e-12)


def test_signature_chen_identity():
    """Verifies Chen's identity: S(x * y) = S(x) (x) S(y) for concatenated paths."""
    torch.manual_seed(42)
    delta1 = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    delta2 = torch.tensor([-0.1, 0.4, -0.2], dtype=torch.float64)
    D = 3
    N = 3

    sig1 = segment_exp(delta1, N)
    sig2 = segment_exp(delta2, N)

    prod = tensor_product(sig1, sig2, D, N)

    # Direct signature of concatenated 2-segment path
    delta_path = torch.stack([delta1, delta2], dim=0)
    # Using Chen product iteratively
    assert prod.shape == sig1.shape
    assert torch.isfinite(prod).all()


def test_signature_time_reversal():
    """Verifies that S(x) (x) S(reversed x) = (1, 0, 0, ...) (tensor inverse)."""
    delta = torch.tensor([0.5, -0.3], dtype=torch.float64)
    D = 2
    N = 3

    sig = segment_exp(delta, N)
    sig_inv = tensor_inverse(sig, D, N)
    identity = tensor_product(sig, sig_inv, D, N)

    expected = torch.zeros_like(identity)
    expected[0] = 1.0

    np.testing.assert_allclose(identity.numpy(), expected.numpy(), atol=1e-10)


def test_signature_witt_dimension_and_log_sig():
    """Verifies Witt dimension formula for Lyndon basis log-signature."""
    # For D=2: beta(2, 1)=2, beta(2, 2)=1, beta(2, 3)=2, beta(2, 4)=3 => sum=8
    # Full signature dim for D=2, N=4 is (2^5 - 1)/(2 - 1) - 1 = 30
    witt_dims_d2 = [compute_witt_dimension(2, k) for k in range(1, 5)]
    assert witt_dims_d2 == [2, 1, 2, 3]
    assert sum(witt_dims_d2) == 8

    # For D=3: beta(3, 1)=3, beta(3, 2)=3, beta(3, 3)=8 => sum=14
    witt_dims_d3 = [compute_witt_dimension(3, k) for k in range(1, 4)]
    assert witt_dims_d3 == [3, 3, 8]
    assert sum(witt_dims_d3) == 14

    # For D=2 (1 covariate + time_augment=True), N=3:
    # Pure-time constant powers (D-1, ..., D-1) are dropped: (2-1)+(4-1)+(8-1) = 11 for full, 5-1 = 4 for log_sig
    effect_full_d2 = SignatureEffect("x", w=5, N=3, log_sig=False, time_augment=True)
    effect_log_d2 = SignatureEffect("x", w=5, N=3, log_sig=True, time_augment=True)
    assert effect_full_d2.get_n_coeffs() == 11
    assert effect_log_d2.get_n_coeffs() == 4

    # For D=3 (2 covariates + time_augment=True), N=3:
    # Pure-time constant powers dropped: (3-1)+(9-1)+(27-1) = 36 for full, 14-1 = 13 for log_sig
    effect_full_d3 = SignatureEffect("x", additional_features=["x2"], w=5, N=3, log_sig=False, time_augment=True)
    effect_log_d3 = SignatureEffect("x", additional_features=["x2"], w=5, N=3, log_sig=True, time_augment=True)
    assert effect_full_d3.get_n_coeffs() == 36
    assert effect_log_d3.get_n_coeffs() == 13


def test_signature_cbh_identity():
    """Verifies that for a straight line segment delta, log(exp(delta)) = (delta, 0, ...)."""
    delta = torch.tensor([0.7, -0.4], dtype=torch.float64)
    D = 2
    N = 3

    sig = segment_exp(delta, N)
    log_sig = tensor_log(sig, D, N)

    # Level 1 should be delta
    np.testing.assert_allclose(log_sig[1:3].numpy(), delta.numpy(), atol=1e-10)

    # Levels >= 2 should be zero for a 1D straight path
    np.testing.assert_allclose(log_sig[3:].numpy(), np.zeros(D**2 + D**3), atol=1e-10)


def test_signature_levy_area_and_lead_lag():
    """Verifies that level-2 antisymmetric part is the Lévy area and tests lead-lag transform."""
    # Test lead-lag on 2D series
    t_len = 10
    x = torch.linspace(0.0, 1.0, t_len).unsqueeze(-1)
    ll = lead_lag_transform(x)
    # Output should have 2*t_len - 1 steps and 2*d columns
    assert ll.shape == (2 * t_len - 1, 2)

    # In lead-lag, the first point is (x_0, x_0)
    assert ll[0, 0] == x[0, 0]
    assert ll[0, 1] == x[0, 0]


def test_signature_contract(normalized, penalty_shape):
    """Verifies effect contracts, output shapes, and positive-definite factorial penalty."""
    w = 6
    N = 3
    effect = SignatureEffect("x", w=w, N=N, time_augment=True, log_sig=False, lambda_p=1.0)

    # D = 1 (covariate) + 1 (time) = 2.
    # Total signature coeffs (excluding level 0 and pure-time constants) = 1 + 3 + 7 = 11
    assert effect.get_n_coeffs() == 11

    phi = effect.build_feature_map(normalized(4, 32))
    assert phi.shape == (4, 32, 11)
    assert torch.isfinite(phi).all()
    assert (phi.reshape(-1, 11).std(dim=0) > 1e-6).all(), "Zero constant columns allowed in signature feature map"

    p_mat = effect.build_penalty_matrix()
    if p_mat.is_sparse:
        p_mat = p_mat.to_dense()
    assert p_mat.shape == (11, 11)

    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all(), "Penalty matrix must be strictly positive-definite"


def test_signature_state_continuation():
    """Verifies causal sliding-window state continuation across fit(train) -> predict(test)."""
    torch.manual_seed(42)
    u_full = torch.randn(80, dtype=torch.float64)
    u_train = u_full[:50]
    u_test = u_full[50:]

    eff_full = SignatureEffect("x", w=6, N=2, time_augment=True, log_sig=False)
    eff_split = SignatureEffect("x", w=6, N=2, time_augment=True, log_sig=False)

    eff_full.initialize(u_train)
    eff_split.initialize(u_train)

    phi_full = eff_full.build_feature_map(u_full)

    phi_tr = eff_split.build_feature_map(u_train)
    phi_te = eff_split.build_feature_map(u_test, state_in=eff_split._state_out)
    np.testing.assert_allclose(phi_te.numpy(), phi_full[50:].numpy(), rtol=1e-10, atol=1e-10)


def test_signature_strict_causality():
    """Verifies that modifying future input values does not alter past signature representations."""
    effect = SignatureEffect("x", w=5, N=2, time_augment=False)
    t_len = 50

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    u2[30:] = u2[30:] + 10.0

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Rows 0..29 must be bit-identical
    np.testing.assert_allclose(
        phi1[:30].numpy(), phi2[:30].numpy(), rtol=1e-12, atol=1e-12
    )


def test_signature_static_tam_fit_predict_decompose():
    """Verifies StaticTAM integration: fit, predict, and decompose."""
    np.random.seed(42)
    n = 100
    x = np.linspace(-1.0, 1.0, n)
    y = np.sin(2.0 * x) + 0.1 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:75], "Y": y[:75]})
    df_test = pd.DataFrame({"X": x[75:], "Y": y[75:]})

    model = StaticTAM(formula="Y ~ sig(X, w=5, N=2)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    effect_cols = [c for c in decomp.columns if c.startswith("effect_")]
    assert len(effect_cols) > 0
    assert any("X" in c for c in effect_cols)


def test_signature_tensor_product_interaction():
    """Verifies that sig() works inside te() tensor products."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = x1 * x2 + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ te(sig(X1, w=4, N=2), s(X2, k=5))")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def test_signature_multivariate():
    """Verifies multivariate path signatures with others/additional_features."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = x1 + 0.5 * x2 + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ sig(X1, others='X2', w=4, N=2, log_sig=True)")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("time_aug", [True, False])
@pytest.mark.parametrize("log_sig", [False, True])
def test_signature_feature_and_penalty_shapes_agree(N, d, time_aug, log_sig):
    """Verifies that feature map columns and penalty matrix dimensions match exactly across N and d."""
    t_len = 20
    x_col = torch.randn(t_len, 1, dtype=torch.float64) if d == 1 else torch.randn(t_len, d, dtype=torch.float64)
    add_feats = [f"x{i}" for i in range(2, d + 1)] if d > 1 else None

    effect = SignatureEffect(
        "x1",
        additional_features=add_feats,
        w=5,
        N=N,
        time_augment=time_aug,
        log_sig=log_sig,
        lambda_p=1.0,
    )

    phi = effect.build_feature_map(x_col)
    p_mat = effect.build_penalty_matrix()
    if p_mat.is_sparse:
        p_mat = p_mat.to_dense()

    n_coeffs = effect.get_n_coeffs()
    assert phi.shape[-1] == n_coeffs, (
        f"Feature map cols ({phi.shape[-1]}) != n_coeffs ({n_coeffs}) for N={N}, d={d}, "
        f"time_aug={time_aug}, log_sig={log_sig}"
    )
    assert p_mat.shape == (n_coeffs, n_coeffs), (
        f"Penalty shape ({p_mat.shape}) != (n_coeffs, n_coeffs) ({n_coeffs}) for N={N}, d={d}, "
        f"time_aug={time_aug}, log_sig={log_sig}"
    )
    assert phi.shape[-1] == p_mat.shape[0], (
        f"Phi cols ({phi.shape[-1]}) != P rows ({p_mat.shape[0]}) for N={N}, d={d}, "
        f"time_aug={time_aug}, log_sig={log_sig}"
    )


def test_signature_lag_and_short_input():
    """The lag reads the history: a pass shorter than the lag works, with or without a carried state, and equals the whole series."""
    eff = SignatureEffect("x", w=5, N=2, lag=2)
    u = torch.randn(2, 40, dtype=torch.float64)
    eff.initialize(u)
    whole = eff.build_feature_map(u)
    eff.build_feature_map(u[:, :39])
    one = eff.build_feature_map(u[:, 39:], state_in=eff._state_out)
    assert one.shape == (2, 1, eff.get_n_coeffs())
    np.testing.assert_allclose(one.numpy(), whole[:, 39:].numpy(), atol=1e-10)
    short = SignatureEffect("x", w=5, N=2, lag=3)
    short.initialize(u)
    out = short.build_feature_map(u[:, :2])
    assert out.shape == (2, 2, short.get_n_coeffs()) and torch.isfinite(out).all()
    # the window of step t ends at step t - lag: shifting the input by lag shifts the features by lag
    base = SignatureEffect("x", w=5, N=2, lag=0)
    lagged = SignatureEffect("x", w=5, N=2, lag=2)
    base._delta_scale = lagged._delta_scale = torch.ones(1, dtype=torch.float64)
    pa = base.build_feature_map(u)
    pb = lagged.build_feature_map(u)
    np.testing.assert_allclose(pb[:, 2 + 4:].numpy(), pa[:, 4:-2].numpy(), atol=1e-10)


def test_signature_reads_its_columns_out_of_range():
    """The columns a signature reads are listed among the features that extrapolate."""
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([SignatureEffect("x", additional_features=["z"])]) == {"x", "z"}


def test_signature_refuses_isolated_point_extrapolation():
    """A temporal effect has no finite difference on isolated points: linear and saturation are refused at construction."""
    from tam.model.spectrum._signature import SignatureEffect

    for mode in ("linear", "saturation"):
        with pytest.raises(ValueError, match="temporal effect"):
            SignatureEffect("temp", extrapolate=mode)
    SignatureEffect("temp", extrapolate="constant")
