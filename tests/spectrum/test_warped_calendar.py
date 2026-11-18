# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import StaticTAM
from tam.model.spectrum._inertia import WarpedCalendarEffect

def test_warped_calendar_effect():
    """Verifies WarpedCalendarEffect feature map, penalty, and StaticTAM formula integration."""
    wcal = WarpedCalendarEffect("tod", regime_feature="holiday", period=48.0, n_harmonics=4)
    assert wcal.get_n_coeffs() == 6 * 4  # 24

    p_mat = wcal.build_penalty_matrix()
    assert p_mat.shape == (24, 24)
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all()

    # Test with StaticTAM
    np.random.seed(42)
    n = 96
    tod = np.tile(np.arange(48), 2).astype(float)
    holiday = np.array([0] * 48 + [1] * 48, dtype=float)
    y = 50.0 + 20.0 * np.sin(2 * np.pi * tod / 48.0) - 10.0 * holiday + np.random.randn(n)

    df = pd.DataFrame({"tod": tod, "holiday": holiday, "y": y})
    # Verify WarpedCalendarEffect has full column rank 6K = 24
    x_wcal = torch.tensor(np.column_stack([2.0 * tod / 48.0 - 1.0, holiday]), dtype=torch.float64)   # the phase is normalized from [0, period] to [-1, 1]
    phi_wcal = wcal.build_feature_map(x_wcal)
    assert phi_wcal.shape == (96, 24)
    rank_wcal = int(torch.linalg.matrix_rank(phi_wcal).item())
    assert rank_wcal == 24, f"Expected full column rank 24 for WarpedCalendarEffect, got {rank_wcal}"

    formula = "y ~ wcal(tod, by=holiday, period=48.0, k=4, ap=-4)"
    model = StaticTAM(formula=formula)
    model.fit(df)
    preds = model.predict(df)
    assert not preds["Estimatedy"].isna().any()
    assert len(preds) == n



def test_warped_calendar_fixed_range_when_grouped_by_tod():
    """Verifies wcal() is exempted by fixed_ranges() so group_col='tod' does not zero its input."""
    from tam.model._data import _transform_data_stacked
    from tam.model.spectrum import fixed_ranges

    rng = np.random.default_rng(7)
    n_days = 10
    tod = np.tile(np.arange(48, dtype=float), n_days)
    regime = np.repeat(np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=float), 48)
    temp = rng.normal(size=len(tod))
    phase = 2.0 * np.pi * tod / 48.0
    y = 100.0 + 25.0 * np.sin(phase) + 15.0 * regime * np.cos(phase + 0.5) + 4.0 * temp + rng.normal(scale=0.1, size=len(tod))
    df = pd.DataFrame({"tod": tod, "regime": regime, "temp": temp, "y": y})

    formula_no = "y ~ s(temp, k=6, ap=-5)"
    formula_wcal = "y ~ s(temp, k=6, ap=-5) + wcal(tod, by=regime, period=48.0, k=4, ap=-5)"

    m_no = StaticTAM(formula=formula_no, group_col="tod").fit(df)
    m_wcal = StaticTAM(formula=formula_wcal, group_col="tod").fit(df)

    # 1. fixed_ranges() must exempt wcal with (0, period)
    assert fixed_ranges(m_wcal.effects_list_) == {"tod": (0.0, 48.0)}

    # 2. On a frame grouped by tod, wcal columns must not collapse to constant 0 within a group or across groups
    x_stacked, _ = _transform_data_stacked(
        df,
        m_wcal.features_config_["features"],
        group_col="tod",
        norm_params=m_wcal.norm_params_,
        unique_groups=m_wcal.unique_groups_,
    )
    wcal_eff = [e for e in m_wcal.effects_list_ if isinstance(e, WarpedCalendarEffect)][0]
    idx = [m_wcal.features_config_["features"].index(f) for f in wcal_eff.required_features]
    phi_wcal = wcal_eff.transform(x_stacked[..., idx])  # (n_groups=48, n_days=10, 6*K=24)
    k = wcal_eff.n_harmonics
    # Within group tod=10 (a phase where no harmonic k <= K has a zero cosine or sine; the regime varies across days),
    # all 4*K warp & amplitude columns must vary
    assert (phi_wcal[10, :, 2 * k :].std(dim=0) > 1e-6).all()
    # Across tod groups, all 2*K base harmonic columns must vary
    assert (phi_wcal[:, 0, : 2 * k].std(dim=0) > 1e-6).all()

    # 3. Fit with wcal must differ from fit without it
    p_no = m_no.predict(df)["Estimatedy"].to_numpy()
    p_wcal = m_wcal.predict(df)["Estimatedy"].to_numpy()
    assert float(np.max(np.abs(p_wcal - p_no))) > 1.0


def test_warped_calendar_base_columns_are_the_daily_harmonics():
    """On the normalized phase of tod = 0..47 the base columns are cos(2 pi k tod / 48) and sin(2 pi k tod / 48)."""
    k = 5
    wcal = WarpedCalendarEffect("tod", period=48.0, n_harmonics=k)
    tod = torch.arange(48, dtype=torch.float64)
    phi = wcal.build_feature_map(2.0 * tod / 48.0 - 1.0)
    harmonics = torch.arange(1, k + 1, dtype=torch.float64)
    expected = torch.cat([torch.cos(2 * np.pi * harmonics * tod[:, None] / 48.0), torch.sin(2 * np.pi * harmonics * tod[:, None] / 48.0)], dim=-1)
    assert torch.allclose(phi, expected, atol=1e-12)
    # with a regime: the envelope of the warp columns is the normalized phase itself
    reg = WarpedCalendarEffect("tod", regime_feature="h", period=48.0, n_harmonics=k)
    x = torch.stack([2.0 * tod / 48.0 - 1.0, torch.ones(48, dtype=torch.float64)], dim=-1)
    full = reg.build_feature_map(x)
    d_cols = torch.cat([-harmonics * expected[:, k:], harmonics * expected[:, :k]], dim=-1)
    assert torch.allclose(full[:, 2 * k:4 * k], (2.0 * tod / 48.0 - 1.0)[:, None] * d_cols, atol=1e-12)
    assert torch.allclose(full[:, 4 * k:], expected, atol=1e-12)


def test_warped_calendar_periodic_extrapolation_wraps_the_phase():
    """A phase one period later (normalized value + 2) gives the same columns, and no error."""
    wcal = WarpedCalendarEffect("tod", period=48.0, n_harmonics=4)
    x = torch.linspace(-0.9, 0.9, 20, dtype=torch.float64)
    assert torch.allclose(wcal.transform(x + 2.0), wcal.transform(x), atol=1e-10)
    assert torch.allclose(wcal.transform(x - 4.0), wcal.transform(x), atol=1e-10)


def test_warped_calendar_regime_keyword_is_a_model_feature():
    """`regime_feature=` is read as a column like `by=`."""
    df = pd.DataFrame({"tod": np.tile(np.arange(48.0), 4), "hol": np.repeat([0.0, 1.0, 0.0, 1.0], 48)})
    df["y"] = np.sin(2 * np.pi * df["tod"] / 48.0) + 0.5 * df["hol"]
    m = StaticTAM("y ~ wcal(tod, regime_feature=hol, period=48.0, k=3, ap=-4)").fit(df)
    assert "hol" in m.features_config_["features"]
    assert float(np.sqrt(np.mean((m.predict(df)["Estimatedy"].to_numpy() - df["y"].to_numpy()) ** 2))) < 0.2


def test_warped_calendar_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([WarpedCalendarEffect("tod", regime_feature="h")]) == {"tod", "h"}
