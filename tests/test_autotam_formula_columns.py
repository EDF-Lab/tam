# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The columns a formula uses are read from its parsed terms, as ``StaticTAM`` reads them: a column taken by keyword (``others``, ``slope``) or inside a
tensor product is found, and a function or keyword word (``s``, ``l``, ``deg``) is not a column even when the frame has one of that name.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.tokens import formula_columns


def test_the_columns_taken_by_keyword_are_found():
    columns = ["y", "x1", "x2", "x3", "x_part", "x_slope", "other"]
    assert formula_columns("y ~ rbf(x1, others='x2|x3') + lt(x_part, slope='x_slope')", columns) == ["y", "x1", "x2", "x3", "x_part", "x_slope"]


def test_the_margins_of_a_tensor_product_are_found():
    assert formula_columns("y ~ te(s(x1, k=5), p(x2, deg=3)) + l(x3)", ["y", "x1", "x2", "x3", "x4"]) == ["y", "x1", "x2", "x3"]


def test_a_function_or_keyword_word_is_not_a_column():
    columns = ["y", "x1", "x2", "s", "l", "deg", "ap", "nominal", "k"]
    assert formula_columns("y ~ s(x1, k=6, deg=3, ap=-3.0) + c(x2, topo='nominal', n_cat=3)", columns) == ["y", "x1", "x2"]


def test_a_dotted_name_and_a_formula_that_cannot_be_parsed():
    assert formula_columns("y ~ s(day.of.year, k=8)", ["y", "day.of.year", "day"]) == ["y", "day.of.year"]
    assert formula_columns("y ~ ((", ["y", "x"]) == ["y"]


def _fitted(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    frame["y"] = 3 + frame["x1"] + 0.5 * frame["x2"] + rng.normal(scale=0.2, size=n)
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=1, pop_size=4, use_opera=False, seed=1, export_dir=str(tmp_path))
    return model.fit(frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})


def test_a_seed_formula_whose_keyword_column_is_missing_is_refused(tmp_path):
    model = _fitted(tmp_path)
    with pytest.raises(ValueError, match="not in the data"):
        model._seed_champions(["y ~ rbf(x1, others='absent')"])
    assert model._seed_champions(["y ~ rbf(x1, others='x2')"]) == ["y ~ rbf(x1, others='x2')"]


def test_a_kept_architecture_whose_keyword_column_is_missing_is_dropped(tmp_path):
    model = _fitted(tmp_path)
    kept = model._kept_architectures({"Isl": ["y ~ rbf(x1, others='absent')", "y ~ rbf(x1, others='x2')"]})
    assert kept == {"Isl": ["y ~ rbf(x1, others='x2')"]}
