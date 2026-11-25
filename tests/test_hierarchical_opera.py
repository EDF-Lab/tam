# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import HierarchicalOperaTAM, OperaTAM

def test_hierarchical_operatam_and_boa():
    rng = np.random.default_rng(9)
    T = 150
    t = np.linspace(0, 4 * np.pi, T)
    y = np.sin(t) + 0.5 * np.cos(2 * t)
    df = pd.DataFrame({
        "y": y,
        "s1": y + 0.10 * rng.standard_normal(T),
        "s2": y + 0.25 * rng.standard_normal(T),
        "k1": y + 0.08 * rng.standard_normal(T),
        "k2": y + 0.20 * rng.standard_normal(T),
        "t1": y + 0.15 * rng.standard_normal(T),
    })

    families = {
        "Splines": ["s1", "s2"],
        "KAN": ["k1", "k2"],
        "Trees": ["t1"],
    }
    h_opera = HierarchicalOperaTAM(
        families=families,
        target_col="y",
        level1_algorithm="MLPOL",
        level2_algorithm="BOA",
    )
    res_online = h_opera.predict_online(df.iloc[:100].reset_index(drop=True))
    assert "prediction_opera" in res_online.columns
    assert "family_weight_Splines" in res_online.columns
    assert "family_weight_KAN" in res_online.columns

    # Composite expert weights sum to 1 at every timestep
    exp_weight_cols = [f"weight_{c}" for c in ["s1", "s2", "k1", "k2", "t1"]]
    w_sums = res_online[exp_weight_cols].sum(axis=1).to_numpy()
    assert np.allclose(w_sums, 1.0, atol=1e-6)

    # Frozen out-of-sample predict and compact
    res_test = h_opera.predict(df.iloc[100:].reset_index(drop=True))
    assert len(res_test) == 50
    assert np.all(np.isfinite(res_test["prediction_opera"].to_numpy()))
    h_opera.compact()
    assert h_opera.compacted_


def test_two_tier_alias_is_the_same_class():
    """`TwoTierOperaTAM` is another name for `HierarchicalOperaTAM`, exported from the package and from `tam.model`."""
    import tam
    from tam import model

    assert tam.TwoTierOperaTAM is tam.HierarchicalOperaTAM
    assert model.TwoTierOperaTAM is model.HierarchicalOperaTAM
    assert "TwoTierOperaTAM" in tam.__all__


def _single_series(T=150, seed=9):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    y = np.sin(t) + 0.5 * np.cos(2 * t)
    return pd.DataFrame({
        "y": y,
        "s1": y + 0.05 * rng.standard_normal(T),
        "s2": y + 0.60 * rng.standard_normal(T) + 0.5,
        "k1": y + 0.08 * rng.standard_normal(T),
        "k2": y + 0.90 * rng.standard_normal(T) - 0.7,
        "t1": y + 0.15 * rng.standard_normal(T),
    })


FAMILIES = {"Splines": ["s1", "s2"], "KAN": ["k1", "k2"], "Trees": ["t1"]}
EXPERTS = ["s1", "s2", "k1", "k2", "t1"]


def test_frozen_forecast_of_a_single_series_uses_the_learnt_weights():
    """Without `group_col`, `predict` after `fit` applies the last composite weights of the two levels, not a uniform mean of the experts."""
    df = _single_series()
    train, test = df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)
    model = HierarchicalOperaTAM(families=FAMILIES, target_col="y", level1_algorithm="MLPOL", level2_algorithm="BOA").fit(train)
    key = next(iter(model.weights_history_))
    last = model.weights_history_[key][-1]                       # composite weights of the five experts at the end of training
    assert not np.allclose(last, 1.0 / len(EXPERTS), atol=1e-3), "the frozen weights are uniform: the test series does not discriminate the experts"
    out = model.predict(test)
    np.testing.assert_allclose(out["prediction_opera"].to_numpy(), test[EXPERTS].to_numpy() @ last, atol=1e-12)
    np.testing.assert_allclose(out[[f"weight_{c}" for c in EXPERTS]].iloc[0].to_numpy(), last, atol=1e-12)
    assert "__dummy_group__" not in out.columns


def test_single_series_and_constant_group_column_give_the_same_frozen_forecast():
    df = _single_series()
    train, test = df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)
    plain = HierarchicalOperaTAM(families=FAMILIES, target_col="y", level1_algorithm="MLPOL", level2_algorithm="BOA").fit(train).predict(test)
    train_g, test_g = train.assign(g="a"), test.assign(g="a")
    grouped = HierarchicalOperaTAM(families=FAMILIES, target_col="y", level1_algorithm="MLPOL", level2_algorithm="BOA", group_col="g").fit(train_g).predict(test_g)
    np.testing.assert_allclose(plain["prediction_opera"].to_numpy(), grouped["prediction_opera"].to_numpy(), atol=1e-12)


def test_flat_opera_frozen_forecast_of_a_single_series_uses_the_learnt_weights():
    """The flat aggregator has the same contract (it was not affected: its fit and predict already used one key)."""
    df = _single_series()
    train, test = df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)
    model = OperaTAM(target_col="y", expert_cols=EXPERTS, algorithm="MLPOL").fit(train)
    last = next(iter(model.weights_history_.values()))[-1]
    assert not np.allclose(last, 1.0 / len(EXPERTS), atol=1e-3)
    out = model.predict(test)
    np.testing.assert_allclose(out["prediction_opera"].to_numpy(), test[EXPERTS].to_numpy() @ last, atol=1e-12)
