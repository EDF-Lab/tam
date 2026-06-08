r"""
Unit tests for ``tam.model.autotam.feature_engineer.FeatureEngineer``, automated
augmentation (temporal/cross features), collinearity filtering, and the stateful
train/inference parity that prevents target leakage.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.feature_engineer import FeatureEngineer


def test_string_columns_become_categorical_codes():
    eng = FeatureEngineer()
    df = pd.DataFrame({"city": ["paris", "lyon", "paris"], "load": [1.0, 2.0, 3.0]})
    out = eng._transform_string_into_categorical(df)
    assert out["city"].dtype == np.int32
    assert "city" in eng.do_not_smooth_features


def test_cross_interaction_created_for_two_numeric_features():
    eng = FeatureEngineer()
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    out = eng._create_cross_interactions(df, ["a", "b"])
    assert "a_div_b" in out.columns


def test_should_augment_respects_protected_set():
    eng = FeatureEngineer()
    eng.do_not_smooth_features.add("weekday")
    assert eng._should_augment("weekday") is False
    assert eng._should_augment("temperature") is True


def test_engineer_features_cross_sectional_sets_fitted():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 50),
        "a": rng.normal(0, 1, 50),
        "b": rng.normal(0, 1, 50),
    })
    eng = FeatureEngineer()
    out = eng.engineer_features(
        df, config={"targets": ["y"], "features": ["a", "b"]},
        metadata={"is_time_series": False, "group_col": None},
    )
    assert eng.is_fitted is True
    assert "a_div_b" in out.columns  # cross interaction survived collinearity filter


def test_create_temporal_features_adds_rolling_and_ewma():
    """The temporal generator emits a rolling-mean and an EWMA column per feature."""
    dates = pd.date_range("2022-01-01", periods=80, freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "ds": dates,
        "feat": np.sin(np.arange(80) * 0.4) + rng.normal(0, 0.1, 80),
    }).sort_values("ds")

    eng = FeatureEngineer()
    out = eng._create_temporal_features(df, ["feat"], group_col=None, date_col="ds", max_safe_footprint=12)
    augmented = [c for c in out.columns if "rolling_mean" in c or "ewma" in c]
    assert len(augmented) >= 2  # one rolling + one EWMA
    assert "feat" in eng.learned_temporal_params


def test_engineer_features_timeseries_runs_and_fits():
    dates = pd.date_range("2022-01-01", periods=80, freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "ds": dates,
        "y": rng.normal(0, 1, 80),
        "feat": np.sin(np.arange(80) * 0.4) + rng.normal(0, 0.1, 80),
    })
    eng = FeatureEngineer()
    out = eng.engineer_features(
        df, config={"targets": ["y"], "features": ["feat"], "lags": {}},
        metadata={"is_time_series": True, "group_col": None},
        date_col="ds",
    )
    assert eng.is_fitted is True
    assert len(out) == len(df)


def test_inference_phase_replicates_dropped_features():
    """After fit, a second call must drop exactly the features purged at training."""
    eng = FeatureEngineer()
    eng.is_fitted = True
    eng.dropped_features = ["a_div_b"]
    df = pd.DataFrame({"y": [1.0, 2.0], "a": [1.0, 2.0], "a_div_b": [0.5, 0.5]})
    out = eng.engineer_features(
        df, config={"targets": ["y"]}, metadata={"is_time_series": False, "group_col": None}
    )
    assert "a_div_b" not in out.columns


def test_filter_collinearity_drops_redundant_generated_feature():
    eng = FeatureEngineer(collinearity_threshold=0.95)
    base = np.linspace(0, 10, 50)
    df = pd.DataFrame({"protected": base, "gen": base * 2.0 + 1.0})  # gen is collinear
    eng.generated_features = ["gen"]
    out = eng._filter_collinearity(df, protected_cols=["protected"], target_col=None)
    assert "gen" not in out.columns
    assert "gen" in eng.dropped_features


def test_detect_protected_features_flags_low_cardinality_integers():
    rng = np.random.default_rng(2)
    # 200 rows -> low-cardinality threshold is 10; weekday's 5 values fall below it.
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 200),
        "weekday": np.tile(np.arange(5), 40),  # low-cardinality integer
        "temp": rng.normal(15, 3, 200),
    })
    eng = FeatureEngineer()
    eng._detect_protected_features(df, date_col=None, target_col="y")
    assert "weekday" in eng.do_not_smooth_features
