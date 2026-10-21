# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Small samples without a date column (opt-in ``small_sample``): fewer folds, a corrected-AIC factor on the error of a candidate, at most ``n // 8``
terms, and a regularised linear baseline the final model cannot be worse than.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.pipeline.data_manager import DataManager

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


class _Coefficients:
    def __init__(self, k):
        self.shape = (1, k)
        self._k = k

    def __array__(self, dtype=None, copy=None):
        return np.zeros((1, self._k))


def test_the_corrected_aic_factor_grows_with_the_coefficients_and_is_one_when_off():
    assert DragTAM("y")._small_sample_factor(object()) == 1.0
    engine = DragTAM("y", aicc_rows=25)
    model = lambda k: type("M", (), {"coefficients_": _Coefficients(k)})()
    assert engine._small_sample_factor(model(3)) == pytest.approx(1.0 + 2 * 3 / (25 - 3 - 1))
    assert engine._small_sample_factor(model(5)) > engine._small_sample_factor(model(3)) > 1.0
    assert engine._small_sample_factor(model(24)) == 10.0                      # more coefficients than the rows allow


def test_a_tiny_sample_has_three_folds_and_a_small_one_five():
    def folds(n, small):
        frame = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
        return len(DataManager("y ~ AutoPipe(x)", small_sample=small).generate_cv_folds(frame, frame.iloc[:0], None, "auto"))
    assert folds(20, True) == 3 and folds(20, False) == 5 and folds(60, True) == 5


def test_a_formula_over_the_cap_is_cut_to_its_first_terms_and_the_mandatory_ones_stay():
    engine = DragTAM("y", mandatory_terms=["l(z)"])
    assert engine._cap_terms("l(a) + s(b, k=3) + w(c, n_scales=2) + l(d)", {"max_terms": 2}) == "l(a) + s(b, k=3)"
    assert engine._cap_terms("l(a) + s(b, k=3) + l(z)", {"max_terms": 2}) == "l(z) + l(a)"
    assert engine._cap_terms("l(a) + l(b)", {}) == "l(a) + l(b)"


def _small(n=30, seed=4):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": 2 + 1.5 * x1 + 0.5 * x2 + rng.normal(scale=0.8, size=n)})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1, x2, x3)", n_experts=2, pop_size=6, use_opera=True, seed=3, export_dir=str(directory), **options)


def test_the_option_is_refused_when_it_is_not_a_boolean_or_auto(tmp_path):
    with pytest.raises(ValueError, match="small_sample"):
        _model(tmp_path, small_sample="yes")


def test_a_small_sample_fit_caps_the_terms_and_carries_the_linear_floor(tmp_path):
    model = _model(tmp_path, small_sample=True).fit(_small(), expansions=EXPANSIONS)
    assert model.is_small_sample_ and model.ctx.small_sample_rows == 30 and model.ctx.search_space["max_terms"] == 3
    seeds = model.architectures_.get("Seed", [])
    assert any(f.startswith("y ~ l(x1, ap=0.0)") for f in seeds) and model.selector.seed_floor is True
    terms = [len(f.split(" + ")) for island, group in model.architectures_.items() if island != "Seed" for f in group]
    assert terms and max(terms) <= 3                                               # n // 8 terms at most


def test_auto_turns_it_on_without_a_date_and_below_150_rows_only(tmp_path):
    assert _model(tmp_path / "a", small_sample="auto").fit(_small(60), expansions=EXPANSIONS).is_small_sample_
    assert not _model(tmp_path / "b", small_sample="auto").fit(_small(200), expansions=EXPANSIONS).is_small_sample_
    assert not _model(tmp_path / "c").fit(_small(30), expansions=EXPANSIONS).is_small_sample_
