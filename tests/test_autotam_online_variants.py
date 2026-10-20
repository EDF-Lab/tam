# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``online_variants``: more configurations of the Kalman and adaptive experts of the best champions (the static forecast as an input of the
adaptive stage, an absolute process-noise grid, a window of 350 periods).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}


def _frame(n=1500):
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    y = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * x1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="D"), "x1": x1, "y": y})


def _names(model, prefix):
    return sorted({row["Model_Name"] for row in model.chronological_test_log if str(row.get("Model_Name", "")).startswith(prefix)})


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    frame = _frame()
    out, warnings = {}, {}
    for label, variants in (("off", False), ("on", True)):
        handler = _Warnings()
        logging.getLogger("tam").addHandler(handler)
        try:
            model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=True, seed=4, dynamic_top=1, online_variants=variants,
                            export_dir=str(tmp_path_factory.mktemp(f"variants_{label}")))
            out[label] = model.fit(frame, date_col="ds", expansions=EXPANSIONS)
        finally:
            logging.getLogger("tam").removeHandler(handler)
        warnings[label] = handler.messages
    return out, frame, warnings


def test_the_default_has_no_forecast_input_and_no_long_window(both):
    models, _, _ = both
    adaptive = _names(models["off"], "Adaptive_")
    assert adaptive and not any("Forecast" in name or "_350p_" in name for name in adaptive)


def test_the_variants_add_forecast_input_formulas_a_noise_grid_and_a_long_window(both):
    models, _, _ = both
    on_adaptive, on_kalman = _names(models["on"], "Adaptive_"), _names(models["on"], "Kalman_")
    assert any(name.startswith("Adaptive_AR_Forecast_") for name in on_adaptive)
    assert any(name.startswith("Adaptive_ECM_Forecast_") for name in on_adaptive)
    assert any("_350p_" in name for name in on_adaptive)
    assert not any("_350p_" in name for name in _names(models["off"], "Adaptive_"))
    assert len(on_kalman) > len(_names(models["off"], "Kalman_"))
    assert any(name.startswith("Kalman_1e_06_") for name in on_kalman)


def test_the_new_experts_are_evaluated_without_a_failure(both):
    models, frame, warnings = both
    assert not [m for m in warnings["on"] if "Forecast" in m or "Kalman_1e" in m or "_350p_" in m]
    model = models["on"]
    out = model.predict(frame.iloc[-30:], date_col="ds")
    assert np.isfinite(out[model.final_model_column_].to_numpy()).all()


def test_grouped_effect_kinds():
    from tam.model.autotam.effect_groups import effect_kind
    assert effect_kind("y_lag_7") == "autoregressive"
    assert effect_kind("te(temp, y_lag_1)") == "autoregressive"
    assert effect_kind("toy") == "calendar" and effect_kind("hour_of_x", ["hour_of_x"]) == "calendar"
    assert effect_kind("temperature") == "exogenous"


def test_the_grouped_decomposition_sums_the_effects_by_kind_and_leaves_the_others_alone():
    import tam as ta
    from tam.model.autotam.effect_groups import GROUP_PREFIX, group_columns, with_effect_groups
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n), "temp": rng.normal(size=n), "toy": np.linspace(0, 6, n) % 1})
    frame["y"] = 3 + frame["temp"] + np.sin(2 * np.pi * frame["toy"]) + rng.normal(scale=0.1, size=n)
    model = ta.StaticTAM("y ~ l(temp) + s(toy, k=6)", date_col="ds").fit(frame)
    grouped = with_effect_groups(model)
    plain, summed = model.decompose_prediction(frame), grouped.decompose_prediction(frame)
    assert set(group_columns(summed)) == {GROUP_PREFIX + "exogenous", GROUP_PREFIX + "calendar"}
    assert [c for c in summed.columns if c.startswith("effect_")] == [c for c in plain.columns if c.startswith("effect_")]
    np.testing.assert_allclose(summed[GROUP_PREFIX + "exogenous"], plain["effect_temp"])
    assert with_effect_groups(grouped) is None   # only a plain StaticTAM is wrapped


@pytest.fixture(scope="module")
def grouped_fit(tmp_path_factory):
    rng = np.random.default_rng(3)
    n = 900
    day = np.arange(n)
    temp = 10 + 8 * np.sin(2 * np.pi * day / 365) + rng.normal(scale=1.0, size=n)
    y = 100 - 2.0 * temp + 5 * np.sin(2 * np.pi * day / 7) + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=n, freq="D"), "temp": temp, "y": y})
    handler = _Warnings()
    logging.getLogger("tam").addHandler(handler)
    try:
        model = AutoTAM("y ~ AutoPipe(temp)", n_experts=2, pop_size=6, use_opera=True, seed=2, dynamic_top=1, online_variants=True,
                        calendar_features=True, export_dir=str(tmp_path_factory.mktemp("grouped_fit")))
        model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    finally:
        logging.getLogger("tam").removeHandler(handler)
    return model, frame, handler.messages


def test_the_grouped_experts_are_built_evaluated_and_forecast(grouped_fit):
    model, frame, messages = grouped_fit
    adaptive = _names(model, "Adaptive_G_") + _names(model, "Adaptive_G_plus_")
    kalman = [n for n in _names(model, "Kalman_") if "_G_" in n]
    assert adaptive and kalman, (adaptive, kalman)
    assert not [m for m in messages if "_G_" in m or "G_plus" in m or "grouped effects" in m]
    out = model.predict(frame.iloc[-20:], date_col="ds")
    assert np.isfinite(out[model.final_model_column_].to_numpy()).all()
