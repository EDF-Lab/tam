# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The distributional ``StaticTAM`` with ``dist_kwargs={"refit_rounds": r, "scale_correction": ...}``:

* the location is refitted with the inverse-variance weights ``1 / sigma^2`` of the scale model (the efficient estimator under a
  heteroscedastic noise), alternating with the scale until the fitted location stops moving;
* the squared residuals that feed the scale model are corrected for the leverage of the location (``E[r^2] = sigma^2 (1 - h)``);
* the fit records its rounds, its convergence and its log-likelihood.

The defaults (``refit_rounds=0``, ``scale_correction="none"``) are the previous fit, bit for bit.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from tam import StaticTAM
from tam.common.utils import _balance_groups, _ensure_dummies

_LOSS = {"mu": "l2", "sigma": "gamma"}


def _data(n=800, seed=0, location="spline"):
    """z = mu(x) + sigma(x) eps with sigma(x) = 0.3 exp(1.6 x) (ratio 5, log-linear: the scale model is well specified)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    sigma = 0.3 * np.exp(1.6 * x)
    mu = np.sin(6 * x) if location == "spline" else 1.0 + 2.0 * x
    return pd.DataFrame({"x": x, "y": mu + sigma * rng.standard_normal(n)}), sigma


def _model(mu="y ~ s(x, k=25)", sigma="y ~ l(x)", **dist_kwargs):
    kwargs = {"log_target": False, "tail_family": "normal"}
    kwargs.update(dist_kwargs)
    return StaticTAM({"mu": mu, "sigma": sigma}, loss=_LOSS, dist_kwargs=kwargs)


def test_defaults_are_the_previous_fit_bit_for_bit():
    frame, _ = _data()
    a = _model().fit(frame).predict_quantiles(frame, (0.05, 0.5, 0.95))
    b = _model(refit_rounds=0, refit_tol=1e-4, scale_correction="none").fit(frame).predict_quantiles(frame, (0.05, 0.5, 0.95))
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_options_are_validated():
    with pytest.raises(ValueError, match="scale_correction"):
        _model(scale_correction="reml")
    with pytest.raises(ValueError, match="refit_rounds"):
        _model(refit_rounds=-1)
    with pytest.raises(ValueError, match="refit_tol"):
        _model(refit_tol=0.0 - 1.0)


# --- weights plumbing -------------------------------------------------------------------------------
def _linear_frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": 1.0 + 2.0 * x + 0.3 * rng.normal(size=n)})


def test_unit_weights_are_the_unweighted_fit():
    frame = _linear_frame().assign(w=1.0)
    plain = StaticTAM("y ~ l(x)").fit(frame)
    weighted = StaticTAM("y ~ l(x)")
    weighted._weight_col_ = "w"
    weighted.fit(frame)
    assert np.allclose(plain.coefficients_.cpu().numpy(), weighted.coefficients_.cpu().numpy(), atol=1e-10)


def test_a_zero_weight_drops_the_row_from_the_solve():
    frame = _linear_frame().assign(w=1.0)
    frame.loc[frame.index[-1], "w"] = 0.0
    frame.loc[frame.index[-1], "y"] = 1e6  # an outlier that the zero weight must ignore
    weighted = StaticTAM("y ~ l(x)")
    weighted._weight_col_ = "w"
    weighted.fit(frame)
    reference = StaticTAM("y ~ l(x)").fit(frame.iloc[:-1])
    assert np.allclose(weighted.predict(frame.iloc[:5])["Estimatedy"], reference.predict(frame.iloc[:5])["Estimatedy"], atol=1e-4)


def test_the_weighted_solve_is_the_closed_form():
    frame = _linear_frame(n=40)
    rng = np.random.default_rng(3)
    frame["w"] = rng.uniform(0.2, 3.0, len(frame))
    model = StaticTAM("y ~ l(x, ap=-3)")
    model._weight_col_ = "w"
    model.fit(frame)
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    x_t, y_t, _ = model._prepare_data(balanced, target_col="y")
    phi = model._build_design_matrix(x_t)[0].numpy()
    penalty = model._build_penalty_matrix().numpy()
    w = frame["w"].to_numpy()
    n = len(frame)
    a = phi.T @ (w[:, None] * phi) + n * penalty + 1e-6 * n * np.eye(phi.shape[1])
    expected = np.linalg.solve(a, phi.T @ (w * y_t[0, :, 0].numpy()))
    assert np.allclose(model.coefficients_[0, :, 0].cpu().numpy(), expected, atol=1e-8)


@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test_bad_weights_are_refused(bad):
    frame = _linear_frame().assign(w=1.0)
    frame.loc[frame.index[3], "w"] = bad
    model = StaticTAM("y ~ l(x)")
    model._weight_col_ = "w"
    with pytest.raises(ValueError, match="weight"):
        model.fit(frame)


def test_all_zero_weights_are_refused():
    frame = _linear_frame().assign(w=0.0)
    model = StaticTAM("y ~ l(x)")
    model._weight_col_ = "w"
    with pytest.raises(ValueError, match="weight"):
        model.fit(frame)


# --- leverage ---------------------------------------------------------------------------------------
def test_leverage_of_a_linear_model_is_the_textbook_one():
    frame = _linear_frame(n=50)
    model = StaticTAM("y ~ l(x, ap=-9)").fit(frame)
    h = model.leverage(frame)
    x = frame["x"].to_numpy()
    textbook = 1.0 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    assert np.allclose(h, textbook, atol=1e-4)


def test_leverage_sums_to_the_degrees_of_freedom_and_stays_below_one():
    frame, _ = _data(300)
    model = StaticTAM("y ~ s(x, k=25, ap=-3)").fit(frame)
    h = model.leverage(frame)
    assert np.all(h >= 0.0) and np.all(h < 1.0)
    assert 3.0 < h.sum() < 25.0


# --- the loop and its record ------------------------------------------------------------------------
def test_the_fit_records_its_rounds():
    frame, _ = _data()
    model = _model(refit_rounds=3, refit_tol=1e-3).fit(frame)
    assert model.n_refits_ == len(model.refit_history_) >= 1
    assert model.converged_ is True or model.converged_ is False
    assert model.refit_history_[0].keys() >= {"round", "change", "log_likelihood", "clipped_weights", "clamped_leverages"}
    assert np.isfinite(model.log_likelihood_)
    plain = _model().fit(frame)
    assert plain.n_refits_ == 0 and plain.refit_history_ == []


def test_it_converges_within_a_few_rounds():
    frame, _ = _data(1500)
    model = _model(refit_rounds=6, refit_tol=1e-3).fit(frame)
    assert model.converged_ and model.n_refits_ <= 4


def test_the_likelihood_does_not_fall_through_the_rounds():
    frame, _ = _data(1500)
    model = _model(refit_rounds=3, refit_tol=0.0).fit(frame)
    values = [step["log_likelihood"] for step in model.refit_history_]
    assert values[-1] >= values[0] - 1e-6 * abs(values[0])


def test_not_converging_warns_and_keeps_the_last_round():
    frame, _ = _data()
    with pytest.warns(UserWarning, match="refit"):
        model = _model(refit_rounds=1, refit_tol=0.0).fit(frame)
    assert model.converged_ is False and model.n_refits_ == 1


def test_homoscedastic_noise_gives_nearly_equal_weights():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 1500)
    frame = pd.DataFrame({"x": x, "y": np.sin(6 * x) + 0.4 * rng.standard_normal(1500)})
    plain = _model().fit(frame).predict_median(frame)
    refit = _model(refit_rounds=2).fit(frame).predict_median(frame)
    assert np.max(np.abs(plain - refit)) < 0.05


@pytest.mark.parametrize("transform", ["asinh", "log"])
def test_transforms_run_and_keep_the_quantiles_ordered(transform):
    frame, _ = _data(1200)
    if transform == "log":
        frame["y"] = np.exp(0.3 * frame["y"])
    model = StaticTAM({"mu": "y ~ s(x, k=15)", "sigma": "y ~ l(x)"}, loss=_LOSS,
                      dist_kwargs={"target_transform": transform, "tail_family": "normal", "refit_rounds": 2, "scale_correction": "leverage"})
    model.fit(frame)
    q = model.predict_quantiles(frame, (0.1, 0.5, 0.9))
    assert np.all(q["q0.1"].to_numpy() <= q["q0.5"].to_numpy()) and np.all(q["q0.5"].to_numpy() <= q["q0.9"].to_numpy())


def test_several_groups_and_an_unsorted_date_follow_the_target_layout():
    rng = np.random.default_rng(6)
    frames = []
    for group in range(3):
        x = rng.uniform(0, 1, 500)
        frames.append(pd.DataFrame({"g": group, "date": pd.date_range("2020-01-01", periods=500, freq="h"), "x": x,
                                    "y": np.sin(6 * x) + (0.2 + 0.6 * x) * rng.standard_normal(500)}))
    frame = pd.concat(frames).sample(frac=1.0, random_state=1)
    model = StaticTAM({"mu": "y ~ s(x, k=15)", "sigma": "y ~ l(x)"}, group_col="g", date_col="date", loss=_LOSS,
                      dist_kwargs={"log_target": False, "tail_family": "normal", "refit_rounds": 1, "scale_correction": "leverage"})
    model.fit(frame)
    q = model.predict_quantiles(frame, (0.1, 0.9))
    assert len(q) == len(frame) and np.all(q["q0.1"].to_numpy() < q["q0.9"].to_numpy())


# --- the statistics: paired Monte Carlo (small) -----------------------------------------------------
GRID = pd.DataFrame({"x": np.linspace(0.05, 0.95, 40)})


def _location_error(rounds, n, replications, location, mu, seed0=1000):
    """Mean squared error of the fitted location on a grid, the same data for every variant (paired)."""
    truth = np.sin(6 * GRID["x"].to_numpy()) if location == "spline" else 1.0 + 2.0 * GRID["x"].to_numpy()
    errors = []
    for r in range(replications):
        frame, _ = _data(n, seed=seed0 + r, location=location)
        errors.append(np.mean((_model(mu=mu, refit_rounds=rounds).fit(frame).predict_median(GRID) - truth) ** 2))
    return float(np.mean(errors))


def test_the_weighted_refit_lowers_the_error_of_a_parsimonious_location():
    # a linear location under a noise whose spread varies 5 times: the weighted estimator is much more efficient
    plain = _location_error(0, 400, 40, "linear", "y ~ l(x)")
    refit = _location_error(1, 400, 40, "linear", "y ~ l(x)")
    assert refit < 0.8 * plain


def test_the_weighted_refit_changes_little_for_a_flexible_location():
    # with a spline of many knots the variance at a point depends on the noise there only: the weights bring almost nothing
    plain = _location_error(0, 400, 30, "spline", "y ~ s(x, k=25)")
    refit = _location_error(1, 400, 30, "spline", "y ~ s(x, k=25)")
    assert 0.9 < refit / plain <= 1.0


@pytest.mark.parametrize("kind", ["leverage", "edf"])
def test_the_leverage_correction_removes_the_downward_bias_of_the_scale(kind):
    ratios = {"none": [], kind: []}
    for r in range(25):
        frame, sigma = _data(150, seed=300 + r)
        for key in ratios:
            model = _model(scale_correction=key).fit(frame)
            _, sigma_hat = model._mu_sigma(frame)
            ratios[key].append(np.mean(sigma_hat ** 2) / np.mean(sigma ** 2))
    assert np.mean(ratios["none"]) < 0.92
    assert abs(np.mean(ratios[kind]) - 1.0) < 0.05
