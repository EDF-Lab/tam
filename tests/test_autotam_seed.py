# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Reproducibility of the search: ``AutoTAM(..., seed=42)``.

With a seed, two fits on the same data give the same experts and the same forecasts, whatever was drawn from the global generators in between and
whatever the hash seed of the interpreter is, and the global generators are left as they were found. Without a seed the search draws from the global
generators as before.
"""

import inspect
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.autotam import seeding
from tam.model.autotam.auto_tam import AutoTAM


def test_the_constructor_has_a_defaulted_seed_argument():
    assert inspect.signature(AutoTAM.__init__).parameters["seed"].default is None


# --------------------------------------------------------------------------- #
# The seeding helper
# --------------------------------------------------------------------------- #

def _draws():
    return random.random(), float(np.random.random()), float(torch.rand(1))


def test_the_same_seed_gives_the_same_draws_and_a_different_seed_other_draws():
    with seeding.seeded(5):
        first = _draws()
    with seeding.seeded(5):
        second = _draws()
    with seeding.seeded(6):
        other = _draws()
    assert first == second and first != other


def test_the_global_generators_are_left_as_they_were_found():
    random.seed(11), np.random.seed(11), torch.manual_seed(11)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    with seeding.seeded(99):
        _draws()
    after = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and torch.equal(before[2], after[2])


def test_without_a_seed_the_generators_are_used_as_before():
    random.seed(11)
    before = random.getstate()
    with seeding.seeded(None):
        random.random()
    assert random.getstate() != before


def test_ties_are_ranked_by_the_score_rounded_then_by_the_label_whatever_the_order():
    candidates = [(1.0000000000001, "b"), (1.0, "a"), (0.5, "z"), (1.0000000000002, "c")]
    expected = ["z", "a", "b", "c"]
    for order in (candidates, candidates[::-1], candidates[2:] + candidates[:2]):
        ranked = sorted(order, key=lambda item: seeding.rank_key(*item))
        assert [label for _, label in ranked] == expected


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=120):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "x3": rng.normal(size=n)})
    df["y"] = 5.0 + 2.0 * df["x1"] - df["x2"] + 0.3 * df["x3"] ** 2 + rng.normal(scale=0.3, size=n)
    return df


def _fit(seed, tmp, df=None):
    df = _frame() if df is None else df
    model = AutoTAM("y ~ AutoPipe(x1, x2, x3)", n_experts=1, pop_size=6, use_opera=False, seed=seed, export_dir=str(tmp))
    model.fit(df.iloc[:100].copy(), date_col="ds", expansions=EXPANSIONS)
    formulas = [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]
    return model, formulas, df


def test_the_same_seed_gives_the_same_experts_and_forecasts_despite_draws_in_between(tmp_path):
    first, formulas_1, df = _fit(7, tmp_path / "a")
    random.random(), np.random.random(), torch.rand(3)     # unrelated draws from the global generators
    second, formulas_2, _ = _fit(7, tmp_path / "b")
    assert formulas_1 == formulas_2
    out_1, out_2 = first.predict(df.iloc[100:], date_col="ds"), second.predict(df.iloc[100:], date_col="ds")
    assert list(out_1.columns) == list(out_2.columns)
    assert np.array_equal(out_1.to_numpy(), out_2.to_numpy())


def test_different_seeds_search_differently(tmp_path):
    _, formulas_1, _ = _fit(7, tmp_path / "a")
    _, formulas_2, _ = _fit(8, tmp_path / "b")
    assert formulas_1 != formulas_2


def test_a_fit_with_a_seed_leaves_the_global_generators_alone(tmp_path):
    random.seed(3), np.random.seed(3), torch.manual_seed(3)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    _fit(7, tmp_path)
    after = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and torch.equal(before[2], after[2])


def test_a_fit_without_a_seed_draws_from_the_global_generators(tmp_path):
    random.seed(3)
    before = random.getstate()
    _fit(None, tmp_path)
    assert random.getstate() != before


def test_the_seed_is_recorded_in_the_export_next_to_the_run(tmp_path):
    _fit(7, tmp_path)
    files = list(Path(tmp_path).glob("AutoTAM_run_info_*.json"))
    assert files and json.loads(files[0].read_text(encoding="utf-8"))["seed"] == 7


SCRIPT = textwrap.dedent('''
    import json, sys, warnings
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd
    from tam.model.autotam.auto_tam import AutoTAM
    rng = np.random.default_rng(4); n = 100
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    df["y"] = 5.0 + 2.0 * df["x1"] - df["x2"] + 0.3 * df["x3"] ** 2 + rng.normal(scale=0.3, size=n)
    m = AutoTAM("y ~ AutoPipe(x1, x2, x3)", n_experts=1, pop_size=6, use_opera=False, seed=7, export_dir=sys.argv[1])
    m.fit(df, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    print("FORMULAS=" + json.dumps([e["Formula"] for e in m.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]))
''')


def test_the_same_seed_gives_the_same_experts_under_different_hash_seeds(tmp_path):
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        run = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / hash_seed)], capture_output=True, text=True, env=env, timeout=600)
        assert run.returncode == 0, run.stderr[-800:]
        outputs.append(next(line for line in run.stdout.splitlines() if line.startswith("FORMULAS=")))
    assert outputs[0] == outputs[1]
