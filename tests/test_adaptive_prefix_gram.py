# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
The windows of a linear AdaptiveTAM corrector are solved from prefix sums of outer products (``_simulate_prefix_gram``). The forecasts must be those of the
window-by-window solver, the path must be left to it when a covariate is constant in a window, and the columns of the base decomposition must still be
returned when the adaptive features are already in the frame.
"""
import numpy as np
import pandas as pd

import tam as ta
from tam.model.adaptative import AdaptiveTAM

N, L, W = 160, 48, 8


def _frame(constant_slice=False):
    rng = np.random.default_rng(3)
    t = np.arange(N)
    x = np.sin(t / 9.0) + 0.1 * rng.normal(size=N)
    z = np.cos(t / 13.0) + 0.1 * rng.normal(size=N)
    if constant_slice:
        z[60:130] = 0.5
    y = 2.0 * x - z + 0.3 * (t % 3) + 0.1 * rng.normal(size=N)
    return pd.DataFrame({"date": pd.date_range("2021-01-01", periods=N, freq="D"), "x": x, "z": z, "k": t % 3, "y": y})


def _adaptive(**kwargs):
    return ta.AdaptiveTAM(adaptive_formula="y ~ l(x) + l(z) + c(k, n_cat=3, topo='nominal')", update_interval_periods=W, training_window_periods=L,
                          steps_per_period=1, horizon_steps=1, date_col="date", default_alpha_p=-6.0, **kwargs)


def test_prefix_gram_path_is_taken_and_equals_the_window_solver(monkeypatch):
    df = _frame()
    used = []
    original = AdaptiveTAM._can_use_prefix_gram
    monkeypatch.setattr(AdaptiveTAM, "_can_use_prefix_gram", lambda self, loss, n: used.append(original(self, loss, n)) or used[-1])
    fast = _adaptive().predict_online(df)["Estimatedy"].to_numpy()
    assert used and all(used)
    monkeypatch.setattr(AdaptiveTAM, "_can_use_prefix_gram", lambda self, loss, n: False)
    slow = _adaptive().predict_online(df)["Estimatedy"].to_numpy()
    both = np.isfinite(fast) & np.isfinite(slow)
    assert both.sum() > 0 and np.array_equal(np.isfinite(fast), np.isfinite(slow))
    np.testing.assert_allclose(fast[both], slow[both], rtol=1e-8, atol=1e-8)


def test_a_covariate_constant_in_a_window_leaves_the_path_to_the_window_solver(monkeypatch):
    df = _frame(constant_slice=True)
    used = []
    original = AdaptiveTAM._can_use_prefix_gram
    monkeypatch.setattr(AdaptiveTAM, "_can_use_prefix_gram", lambda self, loss, n: used.append(original(self, loss, n)) or used[-1])
    _adaptive().predict_online(df)
    assert used and not any(used)


def test_effect_columns_are_returned_when_the_adaptive_features_are_in_the_frame():
    df = _frame()
    base = ta.StaticTAM(formula="y ~ l(x) + s(z, k=5)", date_col="date").fit(df.iloc[:80])
    sim = df.copy()
    sim["Residualy"] = sim["y"] - base.predict(sim)["Estimatedy"].to_numpy()
    adaptive = ta.AdaptiveTAM(base_model=base, adaptive_formula="Residualy ~ l(x)", update_interval_periods=W, training_window_periods=L, steps_per_period=1,
                              horizon_steps=1, date_col="date")
    out = adaptive.predict_online(sim)
    assert {"effect_x", "effect_z", "Estimatedy", "AdaptedEstimatedy"} <= set(out.columns)
