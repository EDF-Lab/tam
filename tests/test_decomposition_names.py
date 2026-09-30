# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for the component naming shared by ``decompose_prediction`` and AutoTAM's
``KnowledgeGraph`` (``tam.model._math.decomposition_names``), and for effect labels staying
aligned with their formula terms when a term does not map to exactly one new feature.
"""

import numpy as np
import pandas as pd

import tam
from tam.model._math import decomposition_names
from tam.model.spectrum import OffsetEffect, LinearEffect, SplineEffect


def _linear(name):
    return LinearEffect(name, scaled=1.0, lambda_p=1e-3, extrapolate="continue")


def test_unique_features_keep_their_name_and_collisions_are_prefixed():
    effects = [
        OffsetEffect(lambda_p=1e-6, extrapolate="continue"),
        _linear("x"),
        SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1e-3, extrapolate="continue"),
        _linear("z"),
    ]
    assert decomposition_names(effects) == ["offset", "l_x", "s_x", "z"]


def test_collision_surviving_the_prefix_is_suffixed_not_overwritten():
    assert decomposition_names([_linear("x"), _linear("x")]) == ["l_x", "l_x_2"]


def _interaction_panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "a": rng.uniform(0.0, 1.0, n),
        "b": rng.integers(0, 3, n).astype(float),
        "z": rng.normal(size=n),
    })
    df["y"] = np.sin(2 * np.pi * df["a"]) * (1.0 + df["b"]) + 2.0 * df["z"] + rng.normal(0, 0.05, n)
    return df


def test_tensor_product_does_not_shift_the_labels_of_later_terms():
    # te() introduces two features; labels used to be reassigned by position from the feature
    # list, so l(z) was reported under 'b' and the te() under 'a'.
    df = _interaction_panel()
    model = tam.StaticTAM(formula="y ~ te(s(a, k=8), c(b, n_cat=3, topo='nominal')) + l(z)").fit(df)

    out = model.decompose_prediction(df)
    effect_cols = [c for c in out.columns if c.startswith("effect_")]
    assert effect_cols == ["effect_offset", "effect_te_a_x_b", "effect_z"]

    np.testing.assert_allclose(
        out[effect_cols].sum(axis=1).to_numpy(), model.predict(df)["Estimatedy"].to_numpy(), atol=1e-6)
    slope = np.polyfit(df["z"], out["effect_z"], 1)[0]
    assert abs(slope - 2.0) < 0.1
