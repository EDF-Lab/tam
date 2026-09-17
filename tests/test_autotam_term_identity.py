# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for how ``tam.model.autotam.knowledge_graph`` identifies formula terms, in particular
tensor products, whose parsed feature is the placeholder 'interaction'.
"""

import numpy as np
import pandas as pd

import tam
from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.knowledge_graph import KnowledgeGraph, term_members, term_signature


def _terms(rhs):
    return parse_formula_to_terms(f"y ~ {rhs}")[1]


def test_term_identity_recovers_tensor_product_members():
    te_term, linear_term = _terms("te(f(toy, m=6, s=2), c(dtw, n_cat=7, topo='nominal')) + l(z)")
    assert term_members(te_term) == [("toy", "f"), ("dtw", "c")]
    assert term_signature(te_term) == "te(f(toy),c(dtw))"
    assert term_members(linear_term) == [("z", "l")]
    assert term_signature(linear_term) == "l(z)"


def test_update_survival_credits_the_features_a_tensor_product_crosses():
    kg = KnowledgeGraph()
    kg.update_survival(_terms("te(s(a), c(b, n_cat=3))"), survived=True)
    assert kg.feature_effect_edges[("a", "te")]["survival_count"] == 1
    assert kg.feature_effect_edges[("b", "te")]["survival_count"] == 1
    assert kg.effects["te"]["survival_count"] == 1
    assert "interaction" not in kg.features


def test_topological_edit_distance_distinguishes_tensor_products():
    kg = KnowledgeGraph()
    first = _terms("te(f(toy), c(dtw, n_cat=7))")
    second = _terms("te(l(lag), c(jf, n_cat=2))")
    assert kg.topological_edit_distance(first, first) == 0.0
    assert kg.topological_edit_distance(first, second) > 0.5


def test_update_and_prune_resolves_tensor_product_columns_on_a_fitted_model():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "a": rng.uniform(0.0, 1.0, n),
        "b": rng.integers(0, 3, n).astype(float),
        "z": rng.normal(size=n),
    })
    df["y"] = np.sin(2 * np.pi * df["a"]) * (1.0 + df["b"]) + 2.0 * df["z"] + rng.normal(0, 0.05, n)
    model = tam.StaticTAM(formula="y ~ te(s(a, k=8), c(b, n_cat=3, topo='nominal')) + l(z)").fit(df)

    kg = KnowledgeGraph()
    kept = kg.update_and_prune(list(model.parsed_terms_), model, df, "y",
                               global_rmse=0.05, target_std=float(df["y"].std()))

    assert [term_signature(t) for t in kept] == ["te(s(a),c(b))", "l(z)"]
    assert kg.feature_effect_edges[("a", "te")]["usage_count"] == 1
    assert kg.feature_effect_edges[("b", "te")]["usage_count"] == 1
    assert kg.interaction_edges[("a", "b")]["usage_count"] == 1
    assert kg.feature_effect_edges[("z", "l")]["usage_count"] == 1
    assert kg.feature_effect_edges[("a", "te")]["avg_variance"] > 0.1
    assert "interaction" not in kg.features


def test_ablation_order_reads_tensor_product_edges_not_a_placeholder():
    kg = KnowledgeGraph()
    te_term, linear_term = _terms("te(s(a), c(b, n_cat=3)) + l(z)")
    kg._register_success(te_term, reward=1.0, penalty=0.0, variance=0.6)
    kg._register_success(linear_term, reward=1.0, penalty=0.0, variance=0.1)
    assert kg.term_avg_variance(te_term) > kg.term_avg_variance(linear_term)
