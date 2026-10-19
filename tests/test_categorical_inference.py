# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``c(col)`` without ``n_cat``: the levels are the codes of the training rows (any integers, contiguous or not), and a code the model
never saw at fit time has no categorical effect (zero: the prediction falls back on the intercept and the other terms) instead of
taking the effect of an edge level.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.spectrum import CategoricalEffect


def _panel(codes, effects, n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(codes, n_per)
    x = rng.normal(size=g.size)
    lookup = dict(zip(codes, effects))
    y = 2.0 * x + np.array([lookup[c] for c in g]) + 0.05 * rng.normal(size=g.size)
    return pd.DataFrame({"g": g, "x": x, "y": y})


def _categorical(model):
    return next(e for e in model.effects_list_ if isinstance(e, CategoricalEffect))


def test_omitted_n_cat_fits_and_predicts_on_contiguous_codes():
    frame = _panel([0, 1, 2, 3], [0.0, 3.0, -2.0, 5.0])
    model = ta.StaticTAM("y ~ l(x) + c(g, ap=-6)").fit(frame)
    assert _categorical(model).n_categories == 4
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"] - frame["y"]) ** 2)) < 0.2


def test_non_contiguous_codes_get_one_column_per_level():
    frame = _panel([10, 50, 1000], [1.0, -4.0, 6.0])
    model = ta.StaticTAM("y ~ l(x) + c(g, ap=-6)").fit(frame)
    assert _categorical(model).n_categories == 3  # not 1001
    assert model.coefficients_.shape[1] <= 3 + 1 + 1
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"] - frame["y"]) ** 2)) < 0.2


def test_levels_given_in_another_order_or_with_gaps_give_the_same_fit():
    frame = _panel([3, 7, 12, 40], [2.0, -1.0, 4.0, 0.0])
    shuffled = frame.sample(frac=1.0, random_state=1)
    a = ta.StaticTAM("y ~ l(x) + c(g, ap=-6)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    b = ta.StaticTAM("y ~ l(x) + c(g, ap=-6)").fit(shuffled).predict(frame)["Estimatedy"].to_numpy()
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("unseen", [2, 30, 5000, -4])
def test_an_unseen_code_has_no_categorical_effect(unseen):
    frame = _panel([10, 50, 1000], [4.0, -6.0, 1.0])
    model = ta.StaticTAM("y ~ l(x) + c(g, ap=-6)").fit(frame)
    probe = pd.DataFrame({"g": [10, unseen], "x": [0.3, 0.3], "y": 0.0})
    parts = model.decompose_prediction(probe)
    assert abs(parts["effect_g"].iloc[0]) > 1.0  # a level that was seen carries its effect
    assert parts["effect_g"].iloc[1] == 0.0  # an unseen code carries none
    predicted = model.predict(probe)["Estimatedy"].to_numpy()
    assert predicted[1] == pytest.approx(parts["effect_offset"].iloc[1] + parts["effect_x"].iloc[1], abs=1e-9)


def test_an_unseen_code_does_not_take_the_effect_of_an_edge_level():
    frame = _panel([0, 1, 2], [0.0, 3.0, 9.0])
    model = ta.StaticTAM("y ~ l(x) + c(g, n_cat=3, ap=-6)").fit(frame)
    edge = model.predict(pd.DataFrame({"g": [2], "x": [0.0], "y": 0.0}))["Estimatedy"].iloc[0]
    beyond = model.predict(pd.DataFrame({"g": [7], "x": [0.0], "y": 0.0}))["Estimatedy"].iloc[0]
    assert abs(edge - beyond) > 3.0


def test_the_prediction_never_raises_on_unseen_or_missing_levels_in_a_tensor_product():
    rng = np.random.default_rng(5)
    n = 400
    frame = pd.DataFrame({"g": rng.choice([3, 8, 21], n), "x": rng.uniform(-1, 1, n)})
    frame["y"] = np.sin(3 * frame["x"]) * frame["g"].map({3: 1.0, 8: -1.0, 21: 2.0}) + 0.05 * rng.normal(size=n)
    model = ta.StaticTAM("y ~ s(x, k=8) + te(s(x, k=6), c(g))").fit(frame)
    probe = pd.DataFrame({"g": [3, 99], "x": [0.3, 0.3], "y": 0.0})
    pred = model.predict(probe)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(pred))
    assert pred[0] != pytest.approx(pred[1], abs=1e-3)


def test_codes_one_to_n_cat_and_other_documented_codings_are_unchanged():
    rng = np.random.default_rng(2)
    n = 300
    frame = pd.DataFrame({"month": rng.integers(1, 13, n), "x": rng.normal(size=n)})
    frame["y"] = frame["x"] + frame["month"] * 0.5
    model = ta.StaticTAM("y ~ l(x) + c(month, n_cat=12, topo='nominal', ap=-6)").fit(frame)
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"] - frame["y"]) ** 2)) < 0.1


def test_a_flag_constant_in_training_keeps_its_single_level_for_every_row():
    rng = np.random.default_rng(8)
    n = 300
    frame = pd.DataFrame({"x": rng.normal(size=n), "crisis": np.zeros(n)})
    frame["y"] = 3.0 + 2.0 * frame["x"] + 0.05 * rng.normal(size=n)
    model = ta.StaticTAM("y ~ l(x) + c(crisis, ap=-8)").fit(frame)
    assert _categorical(model).n_categories == 1
    probe = pd.DataFrame({"x": [0.0, 0.0], "crisis": [0.0, 1.0], "y": 0.0})
    pred = model.predict(probe)["Estimatedy"].to_numpy()
    assert pred[0] == pytest.approx(pred[1], abs=1e-9)  # the flag did not exist in training: no step at test time
