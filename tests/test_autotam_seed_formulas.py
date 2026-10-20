# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Seed formulas: formulas the user already trusts, given to the search as experts.

``AutoTAM.fit(..., seed_formulas=["y ~ l(x1) + l(x2)"])`` expands each seed like a champion of the search (prior, Kalman and adaptive experts),
under the island name ``Seed``, so that the seed competes in the leagues and in the Apex pool and is never lost to a worse search.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

TRUE_FORMULA = "y ~ l(x1) + l(x2)"


def _frame(n=150):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "x3": rng.normal(size=n)})
    df["y"] = 3.0 + 2.0 * df["x1"] - 1.0 * df["x2"] + rng.normal(scale=0.2, size=n)
    return df


def _model(**kwargs):
    return AutoTAM("y ~ AutoPipe(x1, x2, x3)", n_experts=1, pop_size=4, use_opera=True, **kwargs)


def test_fit_has_a_defaulted_seed_formulas_argument():
    assert inspect.signature(AutoTAM.fit).parameters["seed_formulas"].default is None


@pytest.mark.parametrize("seeds, fragment", [
    (["y ~ l(nope)"], "nope"),
    (["z ~ l(x1)"], "target"),
    (["l(x1) + l(x2)"], "~"),
    ([42], "string"),
    ("y ~ l(x1)", "list"),
])
def test_a_wrong_seed_is_rejected_before_the_search_with_its_text(seeds, fragment, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as error:
        _model().fit(_frame(), date_col="ds", seed_formulas=seeds)
    assert fragment in str(error.value)


def test_without_seeds_the_search_champions_are_expanded_as_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model()
    seen = {}

    class _Stop(Exception):
        pass

    def spy(island_champions, ctx, test_log, reporter):
        seen["islands"] = set(island_champions)
        raise _Stop()

    monkeypatch.setattr(model.expander, "generate_experts", spy)
    with pytest.raises(_Stop):
        model.fit(_frame(), date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert "Seed" not in seen["islands"]


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_seed"))
    try:
        df = _frame()
        model = _model()
        model.fit(df.iloc[:120].copy(), date_col="ds", seed_formulas=[TRUE_FORMULA],
                  expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": False, "grid": False})
        yield model, df
    finally:
        os.chdir(previous)


def test_the_seed_is_an_expert_labelled_with_its_island(seeded):
    model, _ = seeded
    names = [e["name"] for e in model.trained_experts]
    assert any(n.startswith("Static_Seed_R1") for n in names)
    logged = [e["Model_Name"] for e in model.chronological_test_log if str(e.get("Model_Name", "")).startswith("Static_Seed_R1")]
    assert logged, "the seed is not in the performance board"


def test_the_seed_competes_in_the_apex_pool(seeded):
    model, _ = seeded
    assert any("Seed_R1" in name for name in model.apex_members_)


def test_the_seed_experts_predict(seeded):
    model, df = seeded
    out = model.predict(df.iloc[120:], date_col="ds")
    seed_columns = [c for c in out.columns if "Seed_R1" in c]
    assert seed_columns and out[seed_columns].notna().all().all()
