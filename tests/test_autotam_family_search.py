# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Families in AutoTAM: a 0/1 target is binomial (not a count), a proportion gets the binomial loss, the family experts are also tuned by the
working-weight GCV, and with ``family_search`` the evolutionary search itself fits its candidates under the family.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.families import candidate_losses
from tam.model.autotam.target_profile import profile_target

EXPANSIONS = {"prior": True, "autofit": True, "kalman": False, "adaptive": False, "grid": False}
PRIOR = {**EXPANSIONS, "autofit": False}          # the search under the family is the prior phase: no need for the auto-fit experts


def test_a_zero_one_target_is_binomial_and_not_a_count():
    profile = profile_target(pd.Series([0, 1, 0, 1, 1, 0] * 10))
    assert profile["hints"] == ["binomial"] and profile["bounds"] == (0.0, 1.0)
    assert candidate_losses(profile) == ["binomial"]


def test_a_proportion_in_zero_one_gets_the_binomial_loss():
    profile = profile_target(pd.Series(np.random.default_rng(0).uniform(0.05, 0.95, 60)))
    assert profile["bounds"] == (0.0, 1.0) and candidate_losses(profile) == ["binomial"]


def test_counts_with_values_above_one_keep_their_losses():
    counts = pd.Series(np.random.default_rng(1).poisson(4.0, 80))
    assert profile_target(counts)["hints"] == ["count"] and candidate_losses(profile_target(counts)) == ["poisson", "negative_binomial"]
    assert profile_target(pd.Series([0, 1, 2, 0, 1, 3] * 10))["hints"] == ["count"]      # three values: not a Bernoulli outcome


def test_the_search_engine_fits_under_a_family_only_when_asked():
    assert DragTAM("y")._loss_kwargs() == {} and DragTAM("y", loss="poisson")._loss_kwargs() == {"loss": "poisson"}


def _counts(n=150, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    mean = np.exp(0.6 + 1.2 * np.sin(np.pi * x1) + 0.8 * x2)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": rng.poisson(mean).astype(float)})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=1, pop_size=4, use_opera=False, seed=4, family="auto", export_dir=str(directory), **options)


def test_the_family_experts_are_tuned_by_the_working_gcv_too(tmp_path):
    model = _model(tmp_path).fit(_counts(), expansions=EXPANSIONS)
    names = [row["Model_Name"] for row in model.chronological_test_log]          # use_opera=False keeps one expert: the log has them all
    assert any("_Family_poisson" in n for n in names) and any("_FamilyAuto_poisson" in n for n in names)


@pytest.mark.slow
def test_the_search_runs_under_the_family_with_family_search(tmp_path, monkeypatch):
    seen = []
    original = DragTAM.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append(self.loss)

    monkeypatch.setattr(DragTAM, "__init__", record)
    model = _model(tmp_path, family_search=True).fit(_counts(100), expansions=PRIOR)
    assert seen and set(seen) == {"poisson"} and model.ctx.family_search is True
    assert model.trained_experts
    seen.clear()
    _model(tmp_path / "plain").fit(_counts(100), expansions=PRIOR)
    assert set(seen) == {"l2"}
