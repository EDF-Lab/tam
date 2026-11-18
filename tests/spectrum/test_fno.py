# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam.model.additive import StaticTAM
from tam.model.spectrum._fno import OperatorEffect

def test_fno_super_resolution_invariance():
    """Verifies that Fourier Neural Operator evaluates continuously across grid resolutions and is causal by default."""
    fno = OperatorEffect("x", modes=8, width=4, causal=False, seed=42)

    # Coarse grid: T = 128
    t_coarse = torch.linspace(0, 1, 128, dtype=torch.float64)
    v_coarse = torch.sin(2 * np.pi * 3 * t_coarse) + 0.5 * torch.cos(2 * np.pi * 5 * t_coarse)
    phi_coarse = fno.build_feature_map(v_coarse)
    assert phi_coarse.shape == (128, 4)

    # Fine grid: T = 512
    t_fine = torch.linspace(0, 1, 512, dtype=torch.float64)
    v_fine = torch.sin(2 * np.pi * 3 * t_fine) + 0.5 * torch.cos(2 * np.pi * 5 * t_fine)
    phi_fine = fno.build_feature_map(v_fine)
    assert phi_fine.shape == (512, 4)

    # Linear interpolation of coarse to fine resolution
    phi_coarse_interp = torch.nn.functional.interpolate(
        phi_coarse.unsqueeze(0).permute(0, 2, 1), size=512, mode="linear", align_corners=True
    ).squeeze(0).permute(1, 0)

    # Relative discretization discrepancy should be negligible (< 0.05)
    max_diff = torch.max(torch.abs(phi_coarse_interp - phi_fine)).item()
    assert max_diff < 0.05

    # Verify default causal=True mode enforces strict causality and state continuation
    fno_causal = OperatorEffect("x", modes=6, width=4, window=16, causal=True, seed=42)
    u1 = torch.randn(80, dtype=torch.float64)
    u2 = u1.clone()
    u2[50:] = u2[50:] + 20.0
    phi_c1 = fno_causal.build_feature_map(u1)
    phi_c2 = fno_causal.build_feature_map(u2)
    np.testing.assert_allclose(phi_c1[:50].numpy(), phi_c2[:50].numpy(), rtol=1e-12, atol=1e-12)

    fno_split = OperatorEffect("x", modes=6, width=4, window=16, causal=True, seed=42)
    phi_tr = fno_split.build_feature_map(u1[:50])
    phi_te = fno_split.build_feature_map(u1[50:], state_in=fno_split._state_out)
    np.testing.assert_allclose(phi_te.numpy(), phi_c1[50:].numpy(), rtol=1e-12, atol=1e-12)



def test_fno_contract_and_penalty():
    """Verifies FNO coefficient dimension and positive definite penalty."""
    modes = 10
    width = 6
    fno = OperatorEffect("x", modes=modes, width=width, lambda_p=2.0)
    assert fno.get_n_coeffs() == width

    p = fno.build_penalty_matrix()
    assert p.shape == (width, width)
    diag = torch.diagonal(p)
    assert (diag > 0).all()


def test_fno_statictam_integration():
    np.random.seed(42)
    n = 128
    x = np.sin(np.linspace(0, 10, n))
    y = 0.5 * x + 0.1 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:90], "Y": y[:90]})
    df_test = pd.DataFrame({"X": x[90:], "Y": y[90:]})
    model = StaticTAM(formula="Y ~ s(X, k=6) + fno(X, modes=6, width=4)")
    model.fit(df_train)
    preds = model.predict(df_test)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()
    assert any("fno" in c for c in model.decompose_prediction(df_test).columns)


def test_fno_window_and_causal_are_parsed_and_used():
    """`fno(x, window=, causal=)` reaches the effect and changes the features."""
    from tam.common.utils import parse_formula_to_terms
    from tam.model.spectrum._factory import create_effects_from_parsed_terms

    def build(formula):
        _, parsed = parse_formula_to_terms(formula)
        return [e for e in create_effects_from_parsed_terms(parsed, {}, 0.0, include_offset=False) if isinstance(e, OperatorEffect)][0]

    default = build("y ~ fno(x)")
    assert default.window == 32 and default.causal is True
    custom = build("y ~ fno(x, window=16, causal=False)")
    assert custom.window == 16 and custom.causal is False and custom.stateful is False
    assert build("y ~ fno(x, W=24)").window == 24
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 200, dtype=torch.float64, generator=g)
    a, b = OperatorEffect("x", window=8, seed=1), OperatorEffect("x", window=24, seed=1)
    assert float((a.build_feature_map(x) - b.build_feature_map(x)).abs().max()) > 1e-3


def test_fno_groups_dtype_and_state():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 150, dtype=torch.float64, generator=g)
    eff = OperatorEffect("x", modes=6, width=4, window=16)
    whole = eff.build_feature_map(x)
    assert whole.shape == (3, 150, 4)
    assert torch.allclose(eff.build_feature_map(x[2:3])[0], whole[2], atol=1e-12)
    assert eff.build_feature_map(x.float()).dtype == torch.float32
    eff.build_feature_map(x[:, :100])
    assert eff._state_out["hist"].shape == (3, 15, 1)
    rest = eff.build_feature_map(x[:, 100:], state_in=eff._state_out)
    assert torch.allclose(rest, whole[:, 100:], atol=1e-10)
    with pytest.raises(ValueError, match="temporal effect"):
        OperatorEffect("x", extrapolate="linear")
    OperatorEffect("x", extrapolate="linear", causal=False)


def test_fno_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([OperatorEffect("x")]) == {"x"}
