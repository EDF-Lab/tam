r"""
Coverage for ``OperaTAM.plot_weights`` and the single-series aggregation path.

Matplotlib runs on the non-interactive Agg backend and ``plt.show`` is patched,
so these tests render figures in-memory without opening windows.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import tam as ta


@pytest.fixture
def grouped_experts():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": list(pd.date_range("2023-01-01", periods=20)) * 2,
        "group": ["A"] * 20 + ["B"] * 20,
        "actual_target": rng.normal(0, 1, 40),
        "expert_1": rng.normal(0, 1, 40),
        "expert_2": rng.normal(0, 1, 40),
    })


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)


def test_plot_weights_before_predict_raises():
    opera = ta.OperaTAM(target_col="y", expert_cols=["e1", "e2"])
    with pytest.raises(ValueError, match="No prediction history"):
        opera.plot_weights()


def test_plot_weights_grouped_with_datetime(grouped_experts):
    opera = ta.OperaTAM(
        target_col="actual_target", expert_cols=["expert_1", "expert_2"],
        algorithm="EWA", group_col="group", date_col="date",
    )
    opera.predict_online(grouped_experts)
    # df provided with a 'date' column -> exercises the datetime axis branch.
    opera.plot_weights(df=grouped_experts)


def test_plot_weights_single_group_integer_axis(grouped_experts):
    opera = ta.OperaTAM(
        target_col="actual_target", expert_cols=["expert_1", "expert_2"],
        algorithm="MLpol", group_col="group", date_col="date",
    )
    opera.predict_online(grouped_experts)
    # No df -> integer step axis; single named group.
    opera.plot_weights(group_name="A")


def test_plot_weights_many_experts_uses_colormap():
    """More than 10 experts forces the tab20 colormap branch."""
    rng = np.random.default_rng(1)
    cols = {f"e{i}": rng.normal(0, 1, 15) for i in range(12)}
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=15), "y": rng.normal(0, 1, 15), **cols})

    opera = ta.OperaTAM(target_col="y", expert_cols=list(cols.keys()), algorithm="EWA", date_col="date")
    opera.predict_online(df)
    opera.plot_weights(df=df)


def test_single_series_no_group_aggregation():
    """A bare target/experts frame (no group column) runs the dummy-group path."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 30),
        "e1": rng.normal(0, 1, 30),
        "e2": rng.normal(0, 1, 30),
    })
    opera = ta.OperaTAM(target_col="y", expert_cols=["e1", "e2"], algorithm="MLpol")
    out = opera.predict_online(df)
    assert "prediction_opera" in out.columns
    assert len(out) == len(df)
    assert "__dummy_group__" not in out.columns
