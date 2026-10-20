# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``dynamic_top``: the Kalman and adaptive experts, which take most of the time of a fit, are built only for the best champions of each island.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}


def _frame(n=200):
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * x1 + x2 ** 2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x1, "x2": x2, "y": y})


def _names(model, prefix):
    """The experts of one kind that the search built (the log keeps them whether or not the selection kept them)."""
    return sorted({row["Model_Name"] for row in model.chronological_test_log if str(row.get("Model_Name", "")).startswith(prefix)})


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    frame = _frame()
    out = {}
    for label, top in (("all", None), ("one", 1)):
        model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=5, pop_size=8, use_opera=False, seed=4, dynamic_top=top,
                        export_dir=str(tmp_path_factory.mktemp(f"dyn_{label}")))
        out[label] = model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    return out


def test_the_default_builds_dynamic_experts_for_every_champion_and_one_only_for_the_best_of_each_island(both):
    for prefix in ("Kalman_", "Adaptive_"):
        everything, best_only = _names(both["all"], prefix), _names(both["one"], prefix)
        assert everything and best_only and len(best_only) < len(everything)       # the best champion of each island still gets them


def test_every_champion_keeps_its_static_experts(both):
    assert _names(both["one"], "Static_") == _names(both["all"], "Static_")


def test_the_dynamic_experts_that_remain_belong_to_the_first_champion_of_their_island(both):
    names = _names(both["one"], "Kalman_") + _names(both["one"], "Adaptive_")
    assert names and all(name.endswith("_R1") for name in names)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_value_that_is_not_a_positive_whole_number_is_refused(bad, tmp_path):
    with pytest.raises(ValueError, match="dynamic_top"):
        AutoTAM("y ~ AutoPipe(x1)", dynamic_top=bad, export_dir=str(tmp_path))
