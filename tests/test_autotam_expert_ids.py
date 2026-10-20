# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Stable identifiers of the experts: a short hash of the formula and of its expansion, so that an expert can be followed from one fit to the next,
and ``experts_``, the table of the trained experts with their Apex weights.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.expert_ids import expansion_of, experts_table, identifier_of, identifiers, normalise_formula

EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}


def _record(name, kind="static", formula="y ~ l(x1) + l(x2)", params=None, dynamic=None):
    base = SimpleNamespace(formula_=formula)
    model = base if kind == "static" else SimpleNamespace(base_model_=base)
    return {"name": name, "type": kind, "model": model, "params": params or {}, "dynamic_formula": dynamic, "island": "LinearIsland", "val_rmse": 1.0}


def _kalman(rate, observation=0.1):
    return _record("k", "kalman", params={"process_noise_var": rate, "observation_noise_var": observation}, dynamic="y ~ l(e)")


def _adaptive(window, dynamic="r ~ l(e)"):
    return _record("a", "adaptive", params={"training_window_periods": window, "architecture": "AR_Only"}, dynamic=dynamic)


def test_the_order_of_the_terms_and_the_blanks_do_not_change_the_formula():
    assert normalise_formula("y ~ l(x2) + l(x1)") == normalise_formula("y~l(x1)+ l(x2)")
    assert normalise_formula("y ~ s(x1, k=5) + l(x2)") == "y~l(x2)+s(x1,k=5)"      # a comma inside a term is not a separator


def test_the_name_given_by_the_search_does_not_enter_the_identifier():
    one, other = _record("Static_LinearIsland_R1_Prior"), _record("Static_ChebyshevIsland_R3_Prior")
    assert identifier_of(one) == identifier_of(other)
    assert identifier_of(one).startswith("ex_") and len(identifier_of(one)) == 13


def test_another_formula_or_another_expansion_is_another_identifier():
    base = identifier_of(_record("Static_A_R1_Prior"))
    assert identifier_of(_record("Static_A_R1_Auto")) != base
    assert identifier_of(_record("Static_A_R1_Prior", formula="y ~ l(x1)")) != base
    assert identifier_of(_kalman(1e-3)) != identifier_of(_kalman(1e-2))


def test_a_kalman_noise_is_read_relative_to_the_observation_noise():
    assert expansion_of(_kalman(1e-3, 0.1)) == expansion_of(_kalman(1e-2, 1.0))      # the same multiple of the noise of the data
    assert identifier_of(_kalman(1e-3, 0.1)) == identifier_of(_kalman(1e-2, 1.0))


def test_an_adaptive_expert_is_known_by_its_window_and_its_formula():
    assert identifier_of(_adaptive(7)) != identifier_of(_adaptive(30))
    assert identifier_of(_adaptive(7)) != identifier_of(_adaptive(7, "r ~ l(f)"))


def test_two_experts_with_the_same_formula_and_expansion_are_numbered():
    ids = identifiers([_record("Static_A_R1_Prior"), _record("Static_B_R1_Prior"), _record("Static_B_R2_Prior")])
    values = list(ids.values())
    assert len(set(values)) == 3 and values[1] == values[0] + "_2" and values[2] == values[0] + "_3"


def test_the_apex_weights_of_the_table_sum_to_one_and_the_others_are_zero():
    records = [_record("Static_A_R1_Prior"), _record("Static_A_R1_Auto"), _record("Static_A_R2_Prior", formula="y ~ l(x1)")]
    table = experts_table(records, {"Static_A_R1_Prior": 0.3, "Static_A_R2_Prior": 0.2})
    assert table["apex_weight"].tolist() == pytest.approx([0.6, 0.0, 0.4])
    assert list(table.columns[:3]) == ["identifier", "name", "island"] and table["complete"].isna().all()


def test_an_external_expert_is_known_by_its_given_name():
    record = {"name": "rules", "type": "external", "model": None, "island": None, "val_rmse": 1.0}
    assert identifiers([record]) == {"rules": "rules"}


def _frame(n=200, start=0, seed=7):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n + 40), rng.normal(size=n + 40)
    y = 20 + 3 * np.sin(2 * np.pi * np.arange(n + 40) / 7) + 2 * x1 + x2 ** 2 + rng.normal(scale=0.3, size=n + 40)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n + 40, freq="D"), "x1": x1, "x2": x2, "y": y})
    return frame.iloc[start:start + n].reset_index(drop=True)


def _fit(directory, frame, **options):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=5, pop_size=8, seed=4, dynamic_top=1, export_dir=str(directory), **options)
    return model.fit(frame, date_col="ds", expansions=EXPANSIONS, seed_formulas=["y ~ l(x1) + l(x2)"])


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    frame = _frame()
    first = _fit(tmp_path_factory.mktemp("ids_one"), frame)
    second = _fit(tmp_path_factory.mktemp("ids_two"), frame)
    return first, second, frame


def test_two_fits_of_the_same_data_and_seed_give_the_same_identifiers(fitted):
    first, second, _ = fitted
    assert first.experts_.set_index("name")["identifier"].to_dict() == second.experts_.set_index("name")["identifier"].to_dict()


def test_a_refit_with_the_kept_architectures_keeps_the_identifiers_of_the_static_experts(fitted, tmp_path):
    first, _, frame = fitted
    refit = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=5, pop_size=8, seed=4, dynamic_top=1, export_dir=str(tmp_path))
    refit.fit(frame, date_col="ds", expansions=EXPANSIONS, seed_formulas=["y ~ l(x1) + l(x2)"], architectures=first)

    def static(model):
        return set(model.experts_.loc[model.experts_["name"].str.startswith("Static_"), "identifier"])

    assert static(first) and static(first) == static(refit)


def test_on_two_consecutive_windows_an_expert_whose_formula_and_expansion_did_not_change_keeps_its_identifier(tmp_path_factory):
    one = _fit(tmp_path_factory.mktemp("win_a"), _frame(start=0))
    two = _fit(tmp_path_factory.mktemp("win_b"), _frame(start=20))
    pick_one, pick_two = (m.experts_.set_index("name").loc["Static_Seed_R1_Prior"] for m in (one, two))
    assert pick_one["identifier"] == pick_two["identifier"]
    assert pick_one["formula"] == pick_two["formula"] and pick_one["expansion"] == "static:Prior"


def test_the_table_lists_every_expert_and_its_apex_weights_sum_to_one(fitted):
    model = fitted[0]
    table = model.experts_
    assert sorted(table["name"]) == sorted(e["name"] for e in model.trained_experts)
    assert table["identifier"].is_unique
    assert table["apex_weight"].sum() == pytest.approx(1.0)
    assert set(table.loc[table["apex_weight"] > 0, "name"]) <= set(model.weights_top10)


def test_the_default_columns_are_the_names_and_by_identifier_gives_the_identifiers(fitted):
    model, _, frame = fitted
    by_name = model.predict(frame.iloc[150:])
    by_id = model.predict(frame.iloc[150:], by_identifier=True)
    mapping = model.experts_.set_index("name")["identifier"].to_dict()
    assert set(model.experts_["name"]) <= set(by_name.columns)
    assert list(by_id.columns) == [mapping.get(c, c) for c in by_name.columns]
    assert model.final_model_column_ in by_id.columns                          # the final model keeps its name
    np.testing.assert_array_equal(by_name.to_numpy(), by_id.to_numpy())


def test_the_table_says_which_experts_were_complete_on_the_last_prediction(fitted):
    model, _, frame = fitted
    model.predict(frame.iloc[150:])
    assert model.experts_["complete"].notna().all() and model.experts_["complete"].dtype == bool


def test_the_run_info_records_the_experts(tmp_path):
    _fit(tmp_path, _frame())
    recorded = json.loads(next(Path(tmp_path).glob("AutoTAM_run_info_*.json")).read_text(encoding="utf-8"))
    assert recorded["experts"] and {"identifier", "name", "formula", "expansion", "apex_weight"} <= set(recorded["experts"][0])
