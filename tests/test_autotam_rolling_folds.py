# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Rolling-origin folds (opt-in): the last k horizons of the training rows, each predicted by a model trained on everything before it.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.data_manager import DataManager, rolling_origin_folds


def _pool(n=120, groups=1):
    parts = [pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "g": g, "y": np.arange(n, dtype=float)}) for g in range(groups)]
    return pd.concat(parts, ignore_index=True).sort_values("ds").reset_index(drop=True)


def test_the_folds_are_consecutive_blocks_at_the_end_with_expanding_training_sets():
    pool = _pool()
    folds = rolling_origin_folds(pool, "ds", k=4, rows=7)
    assert len(folds) == 4
    starts = [fold_val["ds"].min() for _, fold_val in folds]
    assert [(b - a).days for a, b in zip(starts, starts[1:])] == [7, 7, 7]
    assert folds[-1][1]["ds"].max() == pool["ds"].max() and all(len(v) == 7 for _, v in folds)
    sizes = [len(t) for t, _ in folds]
    assert sizes == sorted(sizes) and sizes[1] - sizes[0] == 7


def test_no_training_row_is_dated_on_or_after_its_validation_block():
    for train, val in rolling_origin_folds(_pool(groups=3), "ds", k=3, rows=5):
        assert train["ds"].max() < val["ds"].min()


def test_every_group_is_in_every_block():
    folds = rolling_origin_folds(_pool(groups=3), "ds", k=3, rows=5)
    assert all(len(val) == 15 and set(val["g"]) == {0, 1, 2} for _, val in folds)


def test_a_short_history_gets_fewer_folds_and_a_very_short_one_none():
    assert 1 <= len(rolling_origin_folds(_pool(40), "ds", k=6, rows=7)) < 6
    assert rolling_origin_folds(_pool(12), "ds", k=6, rows=7) == []


def test_without_the_argument_the_folds_are_the_usual_ones_and_a_holdout_is_never_rolled():
    pool = _pool()
    fit, dev = pool.iloc[:90], pool.iloc[90:]
    usual = DataManager("y ~ AutoPipe(x1)").generate_cv_folds(fit, dev, "ds", "auto")
    assert [len(v) for _, v in usual] == [len(v) for _, v in DataManager("y ~ AutoPipe(x1)", rolling_folds=None).generate_cv_folds(fit, dev, "ds", "auto", fold_rows=7)]
    rolled = DataManager("y ~ AutoPipe(x1)", rolling_folds=3).generate_cv_folds(fit, dev, "ds", "auto", fold_rows=7)
    assert len(rolled) == 3 and all(len(v) == 7 for _, v in rolled)
    holdout = DataManager("y ~ AutoPipe(x1)", rolling_folds=3).generate_cv_folds(fit, dev, "ds", "holdout", fold_rows=7)
    assert len(holdout) == 1 and len(holdout[0][1]) == len(dev)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_a_bad_number_of_folds_is_refused(bad):
    with pytest.raises(ValueError, match="rolling_folds"):
        DataManager("y ~ AutoPipe(x1)", rolling_folds=bad)


def test_a_fit_uses_k_folds_of_the_horizon(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": rng.normal(size=n)})
    frame["y"] = 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + frame["x1"] + rng.normal(scale=0.3, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=1, pop_size=4, use_opera=False, seed=1, rolling_folds=3, export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert len(model.ctx.cv_folds) == 3 and all(len(val) == 7 for _, val in model.ctx.cv_folds)
