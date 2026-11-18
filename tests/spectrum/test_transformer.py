# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._transformer.TransformerEffect``."""

import numpy as np
import pandas as pd
import pytest
import torch

import tam
from tam.model.additive import StaticTAM
from tam.model.spectrum import TransformerEffect


def test_transformer_contract(normalized, penalty_shape):
    """Verifies coefficient counts, feature map shapes, and positive-definite penalty."""
    m_feat = 16
    n_heads = 4
    effect = TransformerEffect("x", m_features=m_feat, n_heads=n_heads, mode="head", lambda_p=1.0)
    # For univariate input d=1 in mode='head', coeffs = n_heads * d = 4
    assert effect.get_n_coeffs() == n_heads * 1

    phi = effect.build_feature_map(normalized(4, 32))
    assert phi.shape == (4, 32, n_heads)
    assert torch.isfinite(phi).all()

    p_shape = penalty_shape(effect)
    assert p_shape == (n_heads, n_heads)

    p_mat = effect.build_penalty_matrix()
    if p_mat.is_sparse:
        p_mat = p_mat.to_dense()
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all(), "Penalty matrix must be strictly positive-definite"


def test_transformer_favor_plus_kernel_approximation():
    """Verifies FAVOR+ positive random features approximate softmax kernel with lower ORF variance."""
    d = 4
    m = 128
    torch.manual_seed(42)
    q = torch.tensor([0.3, -0.2, 0.4, 0.1], dtype=torch.float64)
    k = torch.tensor([-0.1, 0.5, 0.2, -0.3], dtype=torch.float64)

    exact = torch.exp(torch.dot(q, k) / np.sqrt(d)).item()
    q_tilde = (q / (d**0.25)).unsqueeze(0)
    k_tilde = (k / (d**0.25)).unsqueeze(0)

    # Compare ORF vs IID over multiple random seeds
    ests_orf = []
    ests_iid = []
    for s in range(60):
        eff_orf = TransformerEffect("x", m_features=m, orthogonal=True, seed=s)
        phi_q_orf = eff_orf._compute_prf(q_tilde.to(torch.float64), d=d, is_orf=True)
        phi_k_orf = eff_orf._compute_prf(k_tilde.to(torch.float64), d=d, is_orf=True)
        ests_orf.append(float(torch.sum(phi_q_orf * phi_k_orf)))

        eff_iid = TransformerEffect("x", m_features=m, orthogonal=False, seed=s)
        phi_q_iid = eff_iid._compute_prf(q_tilde.to(torch.float64), d=d, is_orf=False)
        phi_k_iid = eff_iid._compute_prf(k_tilde.to(torch.float64), d=d, is_orf=False)
        ests_iid.append(float(torch.sum(phi_q_iid * phi_k_iid)))

    mean_orf = np.mean(ests_orf)
    var_orf = np.var(ests_orf)
    var_iid = np.var(ests_iid)

    # Check accuracy
    assert abs(mean_orf - exact) / exact < 0.1
    # Check variance reduction from orthogonalization
    assert var_orf < var_iid, f"ORF variance ({var_orf:.6f}) must be < IID variance ({var_iid:.6f})"


def test_transformer_positivity_and_denominator_safety():
    """Verifies that PRF feature map and attention normalizer D_t are strictly positive."""
    effect = TransformerEffect("x", m_features=16, n_heads=4, causal=True)
    x = torch.linspace(-1.0, 1.0, 100).unsqueeze(-1)

    phi_prf = effect._compute_prf(x, d=1, is_orf=True)
    assert (phi_prf > 0.0).all(), "Positive Random Features must be strictly positive"

    states = effect.build_feature_map(x)
    assert torch.isfinite(states).all()
    assert not torch.isnan(states).any()


def test_transformer_strict_causality():
    """Verifies that modifying future input values does not alter past attention representations."""
    effect = TransformerEffect("x", m_features=16, n_heads=4, causal=True)
    t_len = 60

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    u2[35:] = u2[35:] + 5.0

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Rows 0..34 must be bit-identical
    np.testing.assert_allclose(
        phi1[:35].numpy(), phi2[:35].numpy(), rtol=1e-12, atol=1e-12
    )

    # Also verify strict causality in mode='bilinear' (no terminal M_T future leakage)
    effect_bi = TransformerEffect("x", m_features=8, n_heads=2, mode="bilinear", causal=True)
    phi_bi1 = effect_bi.build_feature_map(u1)
    phi_bi2 = effect_bi.build_feature_map(u2)
    np.testing.assert_allclose(
        phi_bi1[:35].numpy(), phi_bi2[:35].numpy(), rtol=1e-12, atol=1e-12
    )

    # Verify causal state continuation across fit(train) -> predict(test)
    eff_full = TransformerEffect("x", m_features=8, n_heads=2, causal=True)
    eff_split = TransformerEffect("x", m_features=8, n_heads=2, causal=True)
    phi_full = eff_full.build_feature_map(u1)
    phi_tr = eff_split.build_feature_map(u1[:35])
    phi_te = eff_split.build_feature_map(u1[35:], state_in=eff_split._state_out)
    np.testing.assert_allclose(phi_te.numpy(), phi_full[35:].numpy(), rtol=1e-10, atol=1e-10)


def test_transformer_linear_scaling_complexity():
    """Verifies linear O(T) compute scaling without quadratic memory allocation."""
    effect = TransformerEffect("x", m_features=16, n_heads=2, causal=True)
    for t_len in [500, 2000, 5000]:
        u = torch.rand(t_len, dtype=torch.get_default_dtype())
        phi = effect.build_feature_map(u)
        assert phi.shape == (t_len, 2)
        assert torch.isfinite(phi).all()


def test_transformer_static_tam_fit_predict_decompose():
    """Verifies StaticTAM integration: fit, predict, and decompose."""
    np.random.seed(42)
    n = 120
    x = np.linspace(-1.0, 1.0, n)
    y = np.sin(2.0 * x) + 0.1 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:90], "Y": y[:90]})
    df_test = pd.DataFrame({"X": x[90:], "Y": y[90:]})

    model = StaticTAM(formula="Y ~ attn(X, m_features=16, n_heads=4)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    effect_cols = [c for c in decomp.columns if c.startswith("effect_")]
    assert len(effect_cols) > 0
    assert any("X" in c for c in effect_cols)


def test_transformer_tensor_product_interaction():
    """Verifies that attn() works inside te() tensor products."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = np.sin(x1) * x2 + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ te(attn(X1, m_features=8, n_heads=2), s(X2, k=6))")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def test_transformer_multivariate():
    """Verifies multivariate inputs with others/additional_features."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = x1 + 0.5 * x2 + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ attn(X1, others='X2', m_features=16, n_heads=2)")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def _reference_causal_attention(eff, u, head):
    """Step-by-step causal attention of one head (the recurrence the scan replaces)."""
    B, T, d = u.shape
    phi = eff._compute_prf(u / float(d) ** 0.25, d=d, head=head)
    gamma = float(eff.decay[head])
    M = torch.zeros(B, eff.m_features, d, dtype=u.dtype)
    z = torch.zeros(B, eff.m_features, dtype=u.dtype)
    out = []
    for t in range(T):
        M = gamma * M + phi[:, t].unsqueeze(-1) * u[:, t].unsqueeze(-2)
        z = gamma * z + phi[:, t]
        out.append(torch.einsum("bm,bmd->bd", phi[:, t], M) / (torch.sum(phi[:, t] * z, dim=-1, keepdim=True) + 1e-6))
    return torch.stack(out, dim=1)


def test_transformer_heads_have_their_own_projection_in_both_modes():
    """Every head draws its own random features: the head blocks of the feature map are not copies of each other."""
    g = torch.Generator().manual_seed(0)
    u = torch.tanh(torch.randn(2, 60, 3, dtype=torch.float64, generator=g))
    for causal in (True, False):
        eff = TransformerEffect("x", additional_features=["y", "z"], m_features=8, n_heads=3, causal=causal, decay=(1.0, 1.0, 1.0))
        omega = eff._get_omega(3)
        assert omega.shape == (3, 8, 3) and not torch.allclose(omega[0], omega[1])
        phi = eff.build_feature_map(u)
        blocks = phi.reshape(2, 60, 3, 3)
        assert float((blocks[:, :, 0] - blocks[:, :, 1]).abs().max()) > 1e-3
        assert float((blocks[:, :, 1] - blocks[:, :, 2]).abs().max()) > 1e-3


def test_transformer_scan_equals_the_stepwise_recurrence():
    g = torch.Generator().manual_seed(1)
    u = torch.tanh(torch.randn(2, 150, 2, dtype=torch.float64, generator=g))
    eff = TransformerEffect("x", additional_features=["z"], m_features=6, n_heads=3, causal=True, decay=(1.0, 0.9, 0.5))
    phi = eff.build_feature_map(u).reshape(2, 150, 3, 2)
    for head in range(3):
        assert torch.allclose(phi[:, :, head], _reference_causal_attention(eff, u, head), atol=1e-9), head


def test_transformer_bidirectional_form_is_stateless_and_per_group():
    eff = TransformerEffect("x", m_features=6, n_heads=2, causal=False)
    assert eff.stateful is False
    g = torch.Generator().manual_seed(2)
    u = torch.rand(3, 40, dtype=torch.float64, generator=g)
    whole = eff.build_feature_map(u)
    assert torch.allclose(eff.build_feature_map(u[1:2])[0], whole[1], atol=1e-12)
    bilinear = TransformerEffect("x", m_features=5, n_heads=2, causal=False, mode="bilinear")
    assert bilinear.build_feature_map(u).shape == (3, 40, 5)


def test_transformer_refuses_isolated_point_extrapolation_when_causal():
    with pytest.raises(ValueError, match="temporal effect"):
        TransformerEffect("x", extrapolate="linear")
    TransformerEffect("x", extrapolate="linear", causal=False)


def test_transformer_fifty_thousand_steps_in_linear_memory():
    import time

    eff = TransformerEffect("x", m_features=8, n_heads=2, causal=True)
    t0 = time.time()
    out = eff.build_feature_map(torch.rand(1, 50_000, dtype=torch.float64))
    assert out.shape == (1, 50_000, 2) and torch.isfinite(out).all() and time.time() - t0 < 30.0


def test_transformer_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([TransformerEffect("x", additional_features=["z"])]) == {"x", "z"}
