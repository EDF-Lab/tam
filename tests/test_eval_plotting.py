r"""
Tests for ``tam.evaluation.eval_plotting``, the benchmark dashboard and summary
table. These exercise the plotting code paths to ensure they execute without
errors; figures render in-memory (Agg backend, ``plt.show`` patched).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tam.evaluation.tracker import BenchmarkTracker
from tam.evaluation.eval_plotting import plot_benchmark_dashboard, generate_summary_table


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)


def _build_data_and_trackers(n_per_split=12, n_models=2):
    """Three chronological splits and a couple of evaluated trackers."""
    rng = np.random.default_rng(0)
    splits = {}
    cursor = pd.Timestamp("2023-01-01")
    for name, length in [("fit", n_per_split), ("val", n_per_split), ("test", n_per_split)]:
        dates = pd.date_range(cursor, periods=length, freq="D")
        cursor = dates[-1] + pd.Timedelta(days=1)
        splits[name] = pd.DataFrame({
            "date": dates,
            "value": rng.normal(50, 5, length),
            "month": dates.month,
        })

    total = sum(len(df) for df in splits.values())
    trackers = {}
    for i in range(n_models):
        tracker = BenchmarkTracker(model_name=f"Model_{i}")
        true_full = pd.concat(splits.values())["value"].values
        tracker.y_pred_full = true_full + rng.normal(0, 1 + i, total)
        tracker.slice_and_evaluate(splits, target_col="value")
        trackers[tracker.model_name] = tracker
    return splits, trackers


def test_dashboard_empty_trackers_returns_early():
    plot_benchmark_dashboard({}, {}, target_col="value")


def test_dashboard_timeseries_with_heatmap():
    splits, trackers = _build_data_and_trackers()
    plot_benchmark_dashboard(
        splits, trackers, target_col="value", date_col="date",
        is_timeseries=True, primary_metric="RMSE", heatmap_col="month",
    )


def test_dashboard_timeseries_with_summary_text_and_smoothing():
    splits, trackers = _build_data_and_trackers()
    plot_benchmark_dashboard(
        splits, trackers, target_col="value", date_col="date",
        is_timeseries=True, primary_metric="MAPE",
        summary_text="Example diagnostic summary.", forecast_smoothing=3,
    )


def test_dashboard_cross_sectional():
    splits, trackers = _build_data_and_trackers()
    plot_benchmark_dashboard(
        splits, trackers, target_col="value", date_col="date",
        is_timeseries=False, primary_metric="MAE", summary_text="Cross-sectional run.",
    )


def test_summary_table_is_ranked_dataframe():
    _, trackers = _build_data_and_trackers()
    table = generate_summary_table(trackers, primary_metric="RMSE")
    assert isinstance(table, pd.DataFrame)
    assert "Model" in table.columns
    assert len(table) == len(trackers)
