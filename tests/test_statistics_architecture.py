# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Architecture & routing tests for the "One Atom, Many Statistics" design.

Verifies that a single ``StaticTAM`` routes to the three schedules by its inputs, that the statistics-layer
public surface is exposed cleanly (with the schedule loops kept private), that the legacy ``tam.model.*``
import aliases survive the move into ``statistics/``, and that the conformal pipeline wrappers deliver their
finite-sample coverage guarantee (a regression guard for studentized ``predict_intervals``).
"""
import importlib

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam import StaticTAM


def _lognormal(seed, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mu, sigma = 1.0 + 1.5 * x, 0.2 + 0.6 * x
    return pd.DataFrame({"x": x, "y": np.exp(mu + sigma * rng.standard_normal(n))})


# --------------------------------------------------------------------------- #
# 1. Routing: one class, three schedules selected by the inputs
# --------------------------------------------------------------------------- #
def test_string_formula_routes_to_plain_mean_regression():
    model = StaticTAM("y ~ s(x)")
    assert model._mode_ == "plain"
    assert model._mixture_components_ is None
    assert model._reweighting_strategy_ is None  # loss='l2' => single exact solve


def test_dict_formula_routes_to_distributional():
    model = StaticTAM({"mu": "y ~ s(x)", "sigma": "~ s(x)"}, loss={"mu": "l2", "sigma": "gamma"})
    assert model._mode_ == "distributional"
    assert model._scale_loss_ == "gamma"
    # a bare dict in `formula` also triggers distributional mode
    assert StaticTAM({"mu": "y ~ s(x)"})._mode_ == "distributional"


def test_mixture_components_routes_to_mixture():
    model = StaticTAM("y ~ s(x)", mixture_components=3)
    assert model._mode_ == "plain" and model._mixture_components_ == 3


def test_mixture_with_dict_formula_is_rejected():
    with pytest.raises(ValueError):
        StaticTAM({"mu": "y ~ s(x)"}, mixture_components=2)


def test_predict_is_guarded_on_non_plain_models():
    frame = _lognormal(0, 400)
    dist_model = StaticTAM({"mu": "y ~ s(x)", "sigma": "~ s(x)"}, loss={"mu": "l2", "sigma": "gamma"}).fit(frame)
    with pytest.raises(RuntimeError):
        dist_model.predict(frame)
    mix = StaticTAM("y ~ s(x)", mixture_components=2, mixture_kwargs={"seed": 0}).fit(frame)
    with pytest.raises(RuntimeError):
        mix.predict(frame)


# --------------------------------------------------------------------------- #
# 2. Public API surface: user tools exposed, schedule loops kept private
# --------------------------------------------------------------------------- #
def test_statistics_public_surface():
    stats = importlib.import_module("tam.model.statistics")
    for name in ("build_strategy", "GaussianCopulaTAM", "SafetyTAM",
                 "ConformalDistributionalTAM", "GeneralizedParetoTail", "fit_gpd_tail",
                 "adaptive_conformal_scores", "adaptive_conformal_intervals"):
        assert name in stats.__all__


def test_meta_models_exposed_at_model_root():
    # tam.model is the Meta-Model + static-engine namespace.
    from tam.model import StaticTAM as SM, AdaptiveTAM, KalmanTAM, OperaTAM, SafetyTAM as Safe
    for name in ("StaticTAM", "AdaptiveTAM", "KalmanTAM", "OperaTAM", "SafetyTAM"):
        assert name in importlib.import_module("tam.model").__all__
    assert SM is ta.StaticTAM and Safe is ta.SafetyTAM


def test_safety_engine_is_static_and_lives_at_root():
    # SafetyTAM is the root static engine, stripped of ACI...
    from tam.model.safety import SafetyTAM as RootSafety
    assert RootSafety is ta.SafetyTAM
    assert not hasattr(RootSafety, "aci_scores")
    import inspect
    assert "method" not in inspect.signature(RootSafety.predict_intervals).parameters
    # ...and the streaming ACI loop lives in the risk layer instead.
    risk = importlib.import_module("tam.model.statistics.risk")
    assert "adaptive_conformal_intervals" in risk.__all__ and "update_risk_level" in risk.__all__
    assert "SafetyTAM" not in risk.__all__


def test_schedule_loops_are_private():
    estimation = importlib.import_module("tam.model.statistics.estimation")
    stats = importlib.import_module("tam.model.statistics")
    # the IRLS/EM drivers are not part of the public surface
    assert "reweighted_penalized_fit" not in estimation.__all__
    assert "reweighted_penalized_fit" not in stats.__all__
    assert "fit_mixture_em" not in estimation.__all__
    # ...but remain importable by their full private module path (used by the models)
    from tam.model.statistics.estimation._reweighting import reweighted_penalized_fit  # noqa: F401
    from tam.model.statistics.estimation._mixture import fit_mixture_em  # noqa: F401
    from tam.model.statistics.estimation._distributional import select_tail_family  # noqa: F401


def test_legacy_root_aliases_survive_the_move():
    # conformal/copula/extremes live in nested statistics packages, but the flat tam.model.* paths resolve...
    from tam.model import SafetyTAM, ConformalDistributionalTAM, GaussianCopulaTAM, GeneralizedParetoTail
    # ...and are the very same objects as the top-level tam.* exports.
    assert SafetyTAM is ta.SafetyTAM
    assert ConformalDistributionalTAM is ta.ConformalDistributionalTAM
    assert GaussianCopulaTAM is ta.GaussianCopulaTAM
    assert GeneralizedParetoTail is ta.GeneralizedParetoTail


def test_dissolved_classes_are_gone():
    assert not hasattr(ta, "DistributionalTAM")
    assert not hasattr(ta, "GaussianMixtureTAM")


# --------------------------------------------------------------------------- #
# 3. The conformal pipeline wrappers on StaticTAM (coverage regression guard)
# --------------------------------------------------------------------------- #
def _fitted_distributional(seed):
    return StaticTAM(
        {"mu": "y ~ s(x)", "sigma": "~ s(x)"},
        loss={"mu": "l2", "sigma": "gamma"}, dist_kwargs={"tail_family": "normal"},
    ).fit(_lognormal(seed))


def _coverage(intervals, y):
    return float(((y >= intervals["Lower"].to_numpy()) & (y <= intervals["Upper"].to_numpy())).mean())


@pytest.mark.parametrize("studentized", [False, True])
def test_predict_intervals_split_conformal_coverage(studentized):
    model = _fitted_distributional(0)
    model.calibrate_conformal(_lognormal(1), alpha=0.1, studentized=studentized)
    test = _lognormal(2)
    intervals = model.predict_intervals(test, method="static")
    assert abs(_coverage(intervals, test["y"].to_numpy()) - 0.9) < 0.03


def test_predict_intervals_requires_calibration_first():
    with pytest.raises(RuntimeError):
        _fitted_distributional(0).predict_intervals(_lognormal(1))


def test_predict_quantiles_pipeline_is_labelled_and_ordered():
    model = _fitted_distributional(0)
    frame = model.predict_quantiles(_lognormal(3, 500), taus=(0.1, 0.5, 0.9))
    assert list(frame.columns) == ["q0.1", "q0.5", "q0.9"]
    assert (frame["q0.1"] <= frame["q0.9"]).all()
