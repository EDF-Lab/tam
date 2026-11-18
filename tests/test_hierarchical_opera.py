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
