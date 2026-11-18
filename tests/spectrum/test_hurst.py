# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam.model.additive import StaticTAM
from tam.model.spectrum._hurst import HurstEffect

def test_hurst_effect_fbm_penalties():
    """Verifies Hurst exponent power-law penalties for persistence and anti-persistence."""
    k = 16
    m = k // 2

    # H = 0.5: Brownian motion, flat spectral penalty
    hurst_flat = HurstEffect("x", k=k, H=0.5, lambda_p=1.0)
    p_flat = hurst_flat.build_penalty_matrix()
    diag_flat = torch.diagonal(p_flat)
    assert torch.allclose(diag_flat, torch.ones_like(diag_flat))

    # H = 0.85: Persistent long-memory, penalizes high frequencies strictly more
    hurst_pers = HurstEffect("x", k=k, H=0.85, lambda_p=1.0)
    diag_pers = torch.diagonal(hurst_pers.build_penalty_matrix())
    assert diag_pers[0] < diag_pers[m - 1]

    # H = 0.25: Anti-persistent rough paths, penalizes high frequencies strictly less
    hurst_rough = HurstEffect("x", k=k, H=0.25, lambda_p=1.0)
    diag_rough = torch.diagonal(hurst_rough.build_penalty_matrix())
    assert diag_rough[0] > diag_rough[m - 1]


def test_hurst_statictam_integration():
    np.random.seed(42)
    n = 128
    x = np.sin(np.linspace(0, 10, n))
    y = 0.5 * x + 0.1 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:90], "Y": y[:90]})
    df_test = pd.DataFrame({"X": x[90:], "Y": y[90:]})
    model = StaticTAM(formula="Y ~ s(X, k=6) + hurst(X, k=8, H=0.75)")
    model.fit(df_train)
    preds = model.predict(df_test)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()
    assert any("hurst" in c for c in model.decompose_prediction(df_test).columns)


def test_hurst_is_pointwise_per_group_and_keeps_the_dtype():
    """A pointwise Fourier basis of the value: the layouts agree, the groups do not interact, the training span is frozen."""
    eff = HurstEffect("x", k=8, H=0.7)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 50, dtype=torch.float64, generator=g) * 4.0
    eff.initialize(x)
    span = eff._fitted_period
    whole = eff.build_feature_map(x)
    assert whole.shape == (3, 50, 8) and eff.stateful is False
    assert torch.allclose(eff.build_feature_map(x.unsqueeze(-1)), whole)
    assert torch.allclose(eff.build_feature_map(x[1:2])[0], whole[1], atol=1e-12)
    assert eff.build_feature_map(x * 10.0).shape == (3, 50, 8) and eff._fitted_period == span
    assert eff.build_feature_map(x.float()).dtype == torch.float32
    assert eff.build_feature_map(x[0]).shape == (50, 8)


def test_hurst_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([HurstEffect("x")]) == {"x"}
