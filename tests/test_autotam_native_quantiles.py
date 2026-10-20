# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Native quantile experts (opt-in): a distributional StaticTAM per island champion that forecasts its own quantiles, joins the pool as a
static expert, and feeds ``predict_quantiles(aggregation="opera")``.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.native_quantiles import NativeQuantileExpert, scale_formula

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=700, seed=5):
    """A target whose noise grows with x2: a location-scale model sees it, a constant error offset does not."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.uniform(size=n)
    y = 20 + 3 * x1 + 2 * np.sin(6 * x2) + rng.normal(size=n) * (0.2 + 2.5 * x2)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n), "x1": x1, "x2": x2, "y": y})


def test_the_scale_is_linear_in_the_numeric_features_of_the_location():
    assert scale_formula("y", "y ~ l(x1) + s(x2, k=8)") == "y ~ l(x1) + l(x2)"
    assert scale_formula("y", "y ~ c(g, n_cat=3)") is None
    assert scale_formula("y", "y ~ l(x1) + te(s(x1), s(x2))") == "y ~ l(x1)"


def test_the_expert_gives_a_median_column_and_sorted_wider_quantiles_where_the_noise_is_larger():
    frame = _frame()
    expert = NativeQuantileExpert("y", "y ~ l(x1) + s(x2, k=8)", date_col="ds").fit(frame.iloc[:500])
    test = frame.iloc[500:]
    assert list(expert.predict(test).columns) == ["Estimatedy"] and len(expert.predict(test)) == len(test)
    q = expert.predict_quantiles(test, (0.1, 0.5, 0.9))
    assert (q["q0.1"] <= q["q0.5"]).all() and (q["q0.5"] <= q["q0.9"]).all()
    width = (q["q0.9"] - q["q0.1"]).to_numpy()
    x2 = test["x2"].to_numpy()
    assert width[x2 > 0.8].mean() > 1.5 * width[x2 < 0.2].mean()


def test_a_location_without_a_numeric_feature_is_refused():
    with pytest.raises(ValueError, match="numeric feature"):
        NativeQuantileExpert("y", "y ~ c(g, n_cat=3)")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=3, pop_size=6, use_opera=True, seed=2, native_quantiles=True,
                    export_dir=str(tmp_path_factory.mktemp("native")))
    model.fit(frame.iloc[:520], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_the_native_experts_join_the_pool_and_forecast(fitted):
    model, frame = fitted
    names = [e["name"] for e in model.trained_experts]
    assert any(n.startswith("Native_") for n in names), names
    out = model.predict(frame.iloc[520:540], date_col="ds")
    assert np.isfinite(out[model.final_model_column_].to_numpy()).all()


def test_the_aggregation_with_native_quantiles_is_sorted_finite_and_different_from_the_empirical_one(fitted):
    model, frame = fitted
    test = frame.iloc[520:]
    own = model.predict_quantiles(test, taus=(0.1, 0.5, 0.9), date_col="ds", aggregation="opera")
    empirical = model.predict_quantiles(test, taus=(0.1, 0.5, 0.9), date_col="ds")
    assert np.isfinite(own.to_numpy()).all() and (own["q0.1"] <= own["q0.5"]).all() and (own["q0.5"] <= own["q0.9"]).all()
    assert not np.allclose(own.to_numpy(), empirical.to_numpy())


def test_without_the_option_there_are_no_native_experts(tmp_path):
    frame = _frame(300)
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=2, pop_size=4, use_opera=True, seed=2, export_dir=str(tmp_path))
    model.fit(frame.iloc[:240], date_col="ds", expansions=EXPANSIONS)
    assert not any(e["name"].startswith("Native_") for e in model.trained_experts)


def test_the_native_experts_are_quantile_members_but_not_members_of_the_final_model(fitted):
    model, frame = fitted
    natives = {e["name"] for e in model.trained_experts if e["name"].startswith("Native_")}
    assert natives and not (natives & set(model.apex_members_))               # the point forecast of the final model is untouched
    model.predict_quantiles(frame.iloc[520:540], taus=(0.1, 0.9), date_col="ds", aggregation="opera")
    assert natives & set(model.quantile_members_)                            # the quantile layer weights them
