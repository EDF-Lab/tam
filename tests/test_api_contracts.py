# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
API contracts: the public signatures that other code (``AutoTAM``, the long-lived feature branches, users) relies on are pinned.

Each pinned callable lists its **existing** parameters in order, with their kind and default. A change that renames, removes, reorders or re-defaults
one of them fails here. Adding a keyword argument with a default after the pinned ones passes: the API is extended, never changed.
Update a pin only on purpose, in the same commit as the deliberate API change.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.common.utils import parse_formula_to_terms

REQUIRED = "<required>"
FIX = ("frozen public API: do not rename, remove, reorder or change the default of an existing parameter; "
       "add a keyword argument with a default after the existing ones instead (see CONTRIBUTING.md).")

# name -> [(parameter, kind, default)] where kind is P (positional or keyword), K (keyword only), * (var-positional), ** (var-keyword)
PINNED = {
    "StaticTAM.__init__": [("self", "P", REQUIRED), ("formula", "P", None), ("group_col", "P", None), ("date_col", "P", None),
                           ("default_alpha_p", "P", -9.0), ("loss", "P", "l2"), ("loss_kwargs", "P", None), ("dist_kwargs", "P", None),
                           ("mixture_components", "P", None), ("mixture_kwargs", "P", None), ("_internal_effects_list", "P", None),
                           ("_internal_features_config", "P", None)],
    "StaticTAM.fit": [("self", "P", REQUIRED), ("data_train", "P", REQUIRED), ("schedule_kwargs", "**", REQUIRED)],
    "StaticTAM.predict": [("self", "P", REQUIRED), ("data", "P", REQUIRED)],
    "AdaptiveTAM.__init__": [("self", "P", REQUIRED), ("adaptive_formula", "P", REQUIRED), ("update_interval_periods", "P", REQUIRED),
                             ("training_window_periods", "P", REQUIRED), ("steps_per_period", "P", REQUIRED), ("base_model", "P", None),
                             ("horizon_steps", "P", 1), ("default_alpha_p", "P", -9.0), ("group_col", "P", None), ("date_col", "P", None),
                             ("add_base_effects", "P", False)],
    "KalmanTAM.__init__": [("self", "P", REQUIRED), ("kalman_formula", "P", REQUIRED), ("base_model", "P", None), ("group_col", "P", None),
                           ("date_col", "P", None), ("use_decomposition", "P", True), ("block_size", "P", None), ("horizon_steps", "P", 1),
                           ("default_alpha_p", "P", -9.0), ("eps", "P", 1e-06), ("offset_boost", "P", 100.0), ("process_noise_var", "P", 0.0001),
                           ("observation_noise_var", "P", 1.0), ("P_init_diag", "P", 1.0), ("add_base_effects", "P", False),
                           ("calibration_steps", "P", None), ("calibration_data", "P", None)],
    "KalmanTAM.fit": [("self", "P", REQUIRED), ("data", "P", REQUIRED), ("kwargs", "**", REQUIRED)],
    "KalmanTAM.predict": [("self", "P", REQUIRED), ("df", "P", REQUIRED)],
    "KalmanTAM.predict_online": [("self", "P", REQUIRED), ("data", "P", REQUIRED), ("kwargs", "**", REQUIRED)],
    "OperaTAM.__init__": [("self", "P", REQUIRED), ("formula", "P", None), ("target_col", "P", None), ("expert_cols", "P", None),
                          ("algorithm", "P", "MLPOL"), ("eta", "P", 1.0), ("loss_type", "P", "square"), ("horizon_steps", "P", 1),
                          ("group_col", "P", None), ("date_col", "P", None)],
    "OperaTAM.fit": [("self", "P", REQUIRED), ("df", "P", REQUIRED)],
    "OperaTAM.predict": [("self", "P", REQUIRED), ("df", "P", REQUIRED)],
    "OperaTAM.predict_online": [("self", "P", REQUIRED), ("df", "P", REQUIRED)],
    "HierarchicalTAM.__init__": [("self", "P", REQUIRED), ("structure", "P", REQUIRED), ("formulas", "P", REQUIRED), ("node_col", "P", REQUIRED),
                                 ("group_col", "P", None), ("date_col", "P", None), ("lambda_p_hier", "P", 1.0)],
    "parse_formula_to_terms": [("formula_str", "P", REQUIRED)],
}

OBJECTS = {
    "StaticTAM.__init__": ta.StaticTAM.__init__, "StaticTAM.fit": ta.StaticTAM.fit, "StaticTAM.predict": ta.StaticTAM.predict,
    "AdaptiveTAM.__init__": ta.AdaptiveTAM.__init__,
    "KalmanTAM.__init__": ta.KalmanTAM.__init__, "KalmanTAM.fit": ta.KalmanTAM.fit, "KalmanTAM.predict": ta.KalmanTAM.predict,
    "KalmanTAM.predict_online": ta.KalmanTAM.predict_online,
    "OperaTAM.__init__": ta.OperaTAM.__init__, "OperaTAM.fit": ta.OperaTAM.fit, "OperaTAM.predict": ta.OperaTAM.predict,
    "OperaTAM.predict_online": ta.OperaTAM.predict_online,
    "HierarchicalTAM.__init__": ta.HierarchicalTAM.__init__,
    "parse_formula_to_terms": parse_formula_to_terms,
}

_KIND = {inspect.Parameter.POSITIONAL_OR_KEYWORD: "P", inspect.Parameter.POSITIONAL_ONLY: "P", inspect.Parameter.KEYWORD_ONLY: "K",
         inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}


def _describe(fn):
    return [(p.name, _KIND[p.kind], REQUIRED if p.default is inspect.Parameter.empty else p.default) for p in inspect.signature(fn).parameters.values()]


def contract_violations(current, pinned):
    """What differs between a signature and its pin; extra parameters after the pinned ones are fine if they have a default."""
    problems = []
    for i, pin in enumerate(pinned):
        if i >= len(current):
            problems.append(f"parameter {pin[0]!r} (position {i}) was removed")
        elif current[i] != pin:
            problems.append(f"position {i}: expected {pin}, found {current[i]}")
    for extra in current[len(pinned):]:
        if extra[1] in ("P", "K") and extra[2] == REQUIRED:
            problems.append(f"new parameter {extra[0]!r} has no default")
    return problems


@pytest.mark.parametrize("name", list(PINNED))
def test_the_signature_is_the_pinned_one_or_extends_it(name):
    problems = contract_violations(_describe(OBJECTS[name]), PINNED[name])
    assert not problems, f"{name}: {'; '.join(problems)}. {FIX}"


# ------------------------------------------------------------------ the checker itself: any change of a pinned parameter fails
def test_the_checker_catches_a_changed_default_a_removed_or_reordered_parameter_and_a_required_new_one():
    def original(self, a, b=1, c="x"):
        pass

    pinned = _describe(original)
    assert contract_violations(pinned, pinned) == []

    def changed_default(self, a, b=2, c="x"):
        pass

    def removed(self, a, b=1):
        pass

    def reordered(self, b=1, a=None, c="x"):
        pass

    def renamed(self, a, bb=1, c="x"):
        pass

    def new_with_default(self, a, b=1, c="x", d=None, *, e=3):
        pass

    def new_required(self, a, b=1, c="x", d=None, *, e):
        pass

    assert contract_violations(_describe(changed_default), pinned)
    assert contract_violations(_describe(removed), pinned)
    assert contract_violations(_describe(reordered), pinned)
    assert contract_violations(_describe(renamed), pinned)
    assert contract_violations(_describe(new_with_default), pinned) == []          # extending the API is allowed
    assert contract_violations(_describe(new_required), pinned)


# ------------------------------------------------------------------ outputs that users and other code read
def test_parse_formula_to_terms_output_schema():
    target, terms = parse_formula_to_terms("y ~ s(x, k=10, ap=-3) + l(z) + c(g, n_cat=3) + te(s(a), s(b), ap=-2)")
    assert target == "y"
    assert [t["type"] for t in terms] == ["s", "l", "c", "te"]
    for term in terms:
        assert set(term) == {"feature", "type", "params"}, FIX
        assert isinstance(term["feature"], str) and isinstance(term["params"], dict)
    assert terms[0] == {"feature": "x", "type": "s", "params": {"k": 10, "ap": -3}}
    assert terms[1]["feature"] == "z" and terms[2]["params"] == {"n_cat": 3}
    assert terms[3]["feature"] == "interaction", FIX


def _small_frame():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n), "x": rng.uniform(-1, 1, n), "k": rng.integers(0, 3, n)})
    df["y"] = df["x"] + 0.3 * df["k"] + rng.normal(0, 0.1, n)
    return df


def test_decompose_prediction_column_naming():
    """One ``effect_<name>`` column per effect: the offset, a feature alone under its name, a shared feature prefixed by its basis, a tensor product as te_<a>_x_<b>."""
    df = _small_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = ta.StaticTAM(formula="y ~ l(x) + c(k, n_cat=3)", date_col="date").fit(df).decompose_prediction(df)
        shared = ta.StaticTAM(formula="y ~ s(x, k=5) + l(x)", date_col="date").fit(df).decompose_prediction(df)
        crossed = ta.StaticTAM(formula="y ~ te(s(x, k=4), c(k, n_cat=3))", date_col="date").fit(df).decompose_prediction(df)
    effects = lambda frame: [c for c in frame.columns if c.startswith("effect_")]
    assert effects(plain) == ["effect_offset", "effect_x", "effect_k"], FIX
    assert effects(shared) == ["effect_offset", "effect_s_x", "effect_l_x"], FIX
    assert effects(crossed) == ["effect_offset", "effect_te_x_x_k"], FIX


def test_summary_columns():
    df = _small_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = ta.StaticTAM(formula="y ~ l(x) + s(x, k=5)", date_col="date").fit(df).summary()
    assert list(summary.columns) == ["Feature", "Type", "Complexity (D)", "Structure / Params", "Reg (log10)"], FIX
