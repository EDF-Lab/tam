# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Conformal intervals around the final model (``predict_intervals``) and the Student-t family for very heavy tails.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.families import candidate_losses


def test_a_heavy_tail_gets_huber_and_a_very_heavy_one_also_the_student_t():
    base = {"n": 200, "zero_share": 0.0, "negative_share": 0.5, "integer_share": 0.0, "skewness": 0.1, "hints": ["gaussian"]}
    assert candidate_losses({**base, "kurtosis": 8.0}) == ["huber"]
    assert candidate_losses({**base, "kurtosis": 20.0}) == ["huber", "student_t"]
    assert candidate_losses({**base, "kurtosis": 1.0}) == []


def _frame(n=260, seed=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x, "y": 5 + 2 * x + rng.normal(scale=0.5, size=n)})


def test_the_intervals_cover_about_the_nominal_share_and_widen_with_the_level(tmp_path):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=2, export_dir=str(tmp_path))
    model.fit(frame.iloc[:200], date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    test = frame.iloc[200:]
    wide, narrow = model.predict_intervals(test, alpha=0.05, date_col="ds"), model.predict_intervals(test, alpha=0.5, date_col="ds")
    assert list(wide.columns) == ["Predicted", "Lower", "Upper"] and len(wide) == len(test)
    covered = float(((test["y"].to_numpy() >= wide["Lower"].to_numpy()) & (test["y"].to_numpy() <= wide["Upper"].to_numpy())).mean())
    assert covered >= 0.85
    assert (wide["Upper"] - wide["Lower"]).mean() > (narrow["Upper"] - narrow["Lower"]).mean()
    with pytest.raises(ValueError, match="alpha"):
        model.predict_intervals(test, alpha=1.5)
