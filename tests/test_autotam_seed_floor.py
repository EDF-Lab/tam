# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The seed floor (opt-in): the experts of the formulas the caller trusts always join the Apex pool, and the frozen final model is never worse
than the best of them on the validation rows.
"""

import types

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.ensemble_selector import SEED_ISLAND, EnsembleSelector


def _validation(apex_noise, seed_noise, n=60):
    rng = np.random.default_rng(0)
    y = 10 + rng.normal(size=n)
    return pd.DataFrame({
        "y": y,
        "Seed_one": y + rng.normal(scale=seed_noise, size=n),
        "Other": y + rng.normal(scale=2.0, size=n),
        "AutoTAM_Apex_Ensemble": y + rng.normal(scale=apex_noise, size=n),
    })


CTX = types.SimpleNamespace(target="y")
EXPERTS = [{"name": "Seed_one", "island": SEED_ISLAND}, {"name": "Other", "island": "LinearIsland"}]


def test_a_seed_expert_better_than_the_apex_becomes_the_final_model():
    weights = {"Other": 0.6, "Seed_one": 0.4}
    out = EnsembleSelector(seed_floor=True)._apply_seed_floor(weights, EXPERTS, _validation(apex_noise=1.5, seed_noise=0.1), CTX, "rmse")
    assert out == {"Seed_one": 1.0}


def test_an_apex_better_than_every_seed_expert_is_kept():
    weights = {"Other": 0.6, "Seed_one": 0.4}
    out = EnsembleSelector(seed_floor=True)._apply_seed_floor(weights, EXPERTS, _validation(apex_noise=0.1, seed_noise=1.5), CTX, "rmse")
    assert out == weights


def test_without_seed_experts_nothing_changes():
    weights = {"Other": 1.0}
    frame = _validation(apex_noise=1.5, seed_noise=0.1)
    assert EnsembleSelector(seed_floor=True)._apply_seed_floor(weights, [EXPERTS[1]], frame, CTX, "rmse") == weights


def test_the_floor_is_off_by_default(tmp_path):
    assert EnsembleSelector().seed_floor is False
    assert AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path)).selector.seed_floor is False
    assert AutoTAM("y ~ AutoPipe(x1)", seed_floor=True, export_dir=str(tmp_path)).selector.seed_floor is True


def test_the_seed_experts_join_the_apex_pool_whatever_the_quality_ratio(tmp_path):
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    frame["y"] = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * frame["x1"] + rng.normal(scale=0.3, size=n)
    expansions = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=2, seed_floor=True, apex_quality_ratio=1.0001,
                    export_dir=str(tmp_path))
    model.fit(frame.iloc[:200], date_col="ds", expansions=expansions, seed_formulas=["y ~ l(x1)"])
    seeds = {e["name"] for e in model.trained_experts if e.get("island") == SEED_ISLAND}
    assert seeds and seeds <= set(model.apex_members_)
    out = model.predict(frame.iloc[200:205], date_col="ds")
    assert np.isfinite(out[model.final_model_column_]).all()
