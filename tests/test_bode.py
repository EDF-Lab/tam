# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.bode.ControlDiagnostics``, the Bode-plot suite for
autoregressive PID components. Matplotlib rendering is suppressed.
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.bode import ControlDiagnostics


@pytest.fixture
def pid_data():
    """A single-group autoregressive series with a usable lag feature."""
    rng = np.random.default_rng(0)
    n = 120
    load = np.cumsum(rng.normal(0, 1, n)) + 50.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="h"),
        "grp": "G1",
        "load_lag": np.concatenate([[50.0], load[:-1]]),
        "load": load,
    })


def _fit_pid_model(df):
    model = ta.StaticTAM(
        formula="load ~ pid(load_lag, w=5)",
        group_col="grp", date_col="timestamp",
    )
    model.fit(df)
    return model


def test_unfitted_model_raises():
    model = ta.StaticTAM(formula="load ~ pid(load_lag, w=5)")
    diag = ControlDiagnostics(model)
    with pytest.raises(RuntimeError, match="fitted"):
        diag.plot_bode(pid_feature="load_lag")


def test_plot_bode_static_pid(pid_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    model = _fit_pid_model(pid_data)
    diag = ControlDiagnostics(model)
    # Should resolve the group, extract Kp/Ki/Kd and render without error.
    diag.plot_bode(pid_feature="load_lag", target_group="G1")


def test_plot_bode_defaults_to_first_group(pid_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    model = _fit_pid_model(pid_data)
    diag = ControlDiagnostics(model)
    diag.plot_bode(pid_feature="load_lag")  # target_group=None -> first group


def test_unknown_group_raises(pid_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    model = _fit_pid_model(pid_data)
    diag = ControlDiagnostics(model)
    with pytest.raises(ValueError, match="not found"):
        diag.plot_bode(pid_feature="load_lag", target_group="NONEXISTENT")


def test_missing_pid_effect_raises(pid_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    # Model without a PID effect.
    model = ta.StaticTAM(
        formula="load ~ l(load_lag)",
        group_col="grp", date_col="timestamp",
    )
    model.fit(pid_data)
    diag = ControlDiagnostics(model)
    with pytest.raises(ValueError, match="No standalone PIDEffect"):
        diag.plot_bode(pid_feature="load_lag", target_group="G1")
