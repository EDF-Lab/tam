# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``te(s(x), c(group), mode='per_level')``: a factor-smooth with one smoothing parameter per level of the factor, and ``l(z, raw=True)``: a
varying-coefficient slope on the raw (unnormalised) covariate, so that the interaction is exactly zero where ``z = 0``.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import SplineEffect, CategoricalEffect, TensorProductEffect


def _te(mode="shared", n_levels=4, lambda_s=1e-3, lambda_c=1e-3, k=6):
    smooth = SplineEffect("x", k, 3, 2, lambda_s, "linear")
    factor = CategoricalEffect("g", n_levels, "nominal", lambda_c, 1, "continue")
    return TensorProductEffect([smooth, factor], 1.0, "linear", mode=mode)


def test_shared_is_the_default_and_unchanged():
    effect = _te("shared")
    assert effect.n_penalty_coordinates == 2
    default = TensorProductEffect([SplineEffect("x", 6, 3, 2, 1e-3, "linear"), CategoricalEffect("g", 4, "nominal", 1e-3, 1, "continue")], 1.0, "linear")
    assert torch.equal(effect.build_penalty_matrix(), default.build_penalty_matrix())


def test_per_level_has_one_coordinate_per_level_and_the_factor_ridge():
    effect = _te("per_level")
    assert effect.n_penalty_coordinates == 4 + 1
    assert effect.penalty_coordinates() == pytest.approx([1e-3] * 4 + [1e-3])
    labels = effect.coordinate_labels()
    assert len(labels) == 5


def test_equal_coordinates_give_the_shared_penalty():
    shared, per_level = _te("shared", lambda_s=2e-2, lambda_c=5e-4), _te("per_level", lambda_s=2e-2, lambda_c=5e-4)
    assert torch.allclose(shared.build_penalty_matrix(), per_level.build_penalty_matrix(), atol=1e-12, rtol=0)


def test_a_level_coordinate_penalises_its_own_block_only():
    effect = _te("per_level", lambda_c=0.0)
    effect.set_penalty_coordinates([1.0, 0.0, 0.0, 0.0, 0.0])
    penalty = effect.build_penalty_matrix()
    k, m = 6 + 3, 4
    blocks = penalty.reshape(k, m, k, m)  # index = smooth * m + level
    assert float(blocks[:, 0, :, 0].abs().sum()) > 0.0
    for level in (1, 2, 3):
        assert float(blocks[:, level, :, level].abs().sum()) == 0.0
    assert float(torch.linalg.eigvalsh(penalty).min()) > -1e-10


def test_per_level_needs_one_smooth_and_one_nominal_factor():
    smooth = SplineEffect("x", 6, 3, 2, 1e-3, "linear")
    with pytest.raises(ValueError, match="per_level"):
        TensorProductEffect([smooth, SplineEffect("z", 6, 3, 2, 1e-3, "linear")], 1.0, "linear", mode="per_level")
    with pytest.raises(ValueError, match="per_level"):
        TensorProductEffect([smooth, CategoricalEffect("g", 4, "ordinal", 1e-3, 1, "continue")], 1.0, "linear", mode="per_level")
    with pytest.raises(ValueError, match="mode"):
        TensorProductEffect([smooth, CategoricalEffect("g", 4, "nominal", 1e-3, 1, "continue")], 1.0, "linear", mode="by")


def _two_groups(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, 2 * n)
    g = np.repeat([0, 1], n)
    truth = np.where(g == 0, 1.0 * x, np.sin(14 * x))
    noise = np.where(g == 0, 0.6, 0.05)
    return pd.DataFrame({"x": x, "g": g, "y": truth + noise * rng.normal(size=2 * n)})


def test_auto_fit_gives_each_level_its_own_smoothness():
    frame = _two_groups()
    model = ta.StaticTAM("y ~ c(g, n_cat=2, ap=-6) + te(s(x, k=12, p=2), c(g, n_cat=2), mode='per_level')").auto_fit(frame, criterion="reml")
    te = next(e for e in model.effects_list_ if isinstance(e, TensorProductEffect))
    lambdas = te.penalty_coordinates()
    assert lambdas[0] > 10.0 * lambdas[1]  # the noisy low-frequency group is smoothed much harder than the clean wiggly one


def test_per_level_fit_beats_a_single_smoothing_parameter_when_the_groups_differ():
    # one data set can go either way (the criterion is a noisy function of the sample): the comparison is on the mean error of several
    errors = {"shared": [], "per_level": []}
    for seed in range(1, 7):
        train, test = _two_groups(seed=seed), _two_groups(seed=seed + 50)
        truth = np.where(test["g"] == 0, test["x"], np.sin(14 * test["x"]))
        for mode in errors:
            model = ta.StaticTAM(f"y ~ c(g, n_cat=2, ap=-6) + te(s(x, k=12, p=2), c(g, n_cat=2), mode='{mode}')").auto_fit(train, criterion="reml")
            errors[mode].append(float(np.sqrt(np.mean((model.predict(test)["Estimatedy"].to_numpy() - truth) ** 2))))
    assert np.mean(errors["per_level"]) < 0.9 * np.mean(errors["shared"])


# --- l(z, raw=True) ---------------------------------------------------------------------------------
def _varying(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 3, n), rng.uniform(3.0, 7.0, n)  # z never near 0
    return pd.DataFrame({"x": x, "z": z, "y": (2.0 + 3.0 * np.sin(x)) * z + 0.1 * rng.normal(size=n)})


def test_a_raw_varying_coefficient_recovers_the_slope_surface():
    train, test = _varying(seed=3), _varying(seed=4)
    model = ta.StaticTAM("y ~ te(s(x, k=10), l(z, raw=True, ap=-9))").fit(train)
    truth = (2.0 + 3.0 * np.sin(test["x"])) * test["z"]
    assert float(np.sqrt(np.mean((model.predict(test)["Estimatedy"].to_numpy() - truth) ** 2))) < 0.3


def test_the_raw_interaction_is_zero_where_the_covariate_is_zero():
    model = ta.StaticTAM("y ~ te(s(x, k=10), l(z, raw=True, ap=-9))").fit(_varying())
    probe = pd.DataFrame({"x": [0.5, 2.0], "z": [0.0, 0.0], "y": 0.0})
    parts = model.decompose_prediction(probe)
    te_column = next(c for c in parts.columns if c.startswith("effect_") and c not in ("effect_offset",))
    assert np.allclose(parts[te_column].to_numpy(), 0.0, atol=1e-9)


def test_a_raw_covariate_may_not_be_read_by_another_term():
    with pytest.raises(ValueError, match="raw"):
        ta.StaticTAM("y ~ l(z, raw=True) + s(z)").fit(_varying())


def test_a_plain_linear_term_is_unchanged():
    frame = _varying()
    a = ta.StaticTAM("y ~ s(x, k=8) + l(z)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    b = ta.StaticTAM("y ~ s(x, k=8) + l(z, raw=False)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    assert np.array_equal(a, b)


def test_a_zero_level_coordinate_raises_no_warning():
    import warnings
    effect = _te("per_level", lambda_c=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        effect.set_penalty_coordinates([1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isfinite(effect.effects[0].lambda_p)
