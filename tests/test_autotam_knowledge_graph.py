# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.knowledge_graph.KnowledgeGraph``, the
Bayesian tracker that scores formula components, prunes redundant/low-variance
terms, and samples effects/parameters for the evolutionary search.
"""

import random

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.knowledge_graph import KnowledgeGraph


class _FakeModel:
    """Minimal stand-in exposing the decompose/predict interface KnowledgeGraph needs."""

    def __init__(self, contributions, estimate):
        self._contributions = contributions
        self._estimate = estimate

    def decompose_prediction(self, df):
        return self._contributions

    def predict(self, df):
        return pd.DataFrame({"Estimatedy": self._estimate})


def test_default_metrics_structure():
    kg = KnowledgeGraph()
    metrics = kg._default_metrics()
    assert metrics["usage_count"] == 0.0
    assert metrics["total_reward"] == 0.0
    assert "params_history" in metrics


def test_update_node_tracks_moving_averages():
    kg = KnowledgeGraph()
    node = kg._default_metrics()
    kg._update_node(node, reward=2.0, penalty=0.5, variance=0.3, params={"k": 10, "name": "x"})
    assert node["usage_count"] == 1
    assert node["total_reward"] == 2.0
    assert node["avg_penalty"] == 0.5
    # Only numeric params are recorded in the history.
    assert node["params_history"]["k"] == [10]
    assert "name" not in node["params_history"]


def test_calculate_score_unused_node_returns_one():
    kg = KnowledgeGraph()
    assert kg._calculate_score(kg._default_metrics()) == 1.0


def test_register_success_populates_nodes_and_interactions():
    kg = KnowledgeGraph()
    term = {"type": "n", "feature": "temp", "params": {"others": "humidity"}}
    kg._register_success(term, reward=1.5, penalty=0.2, variance=0.4)

    assert kg.features["temp"]["usage_count"] == 1
    assert kg.effects["n"]["usage_count"] == 1
    assert kg.feature_effect_edges[("temp", "n")]["usage_count"] == 1
    # The 'others' param wires a feature-feature interaction edge.
    assert kg.interaction_edges[("humidity", "temp")]["usage_count"] == 1


def test_update_survival_increments_only_when_survived():
    kg = KnowledgeGraph()
    terms = [{"type": "s", "feature": "temp", "params": {}}]
    kg.update_survival(terms, survived=False)
    assert kg.features["temp"]["survival_count"] == 0
    kg.update_survival(terms, survived=True)
    assert kg.features["temp"]["survival_count"] == 1


def test_suggest_parameters_returns_median_consensus():
    kg = KnowledgeGraph()
    edge = kg.feature_effect_edges[("temp", "s")]
    edge["params_history"]["k"].extend([5, 9, 11])  # ints -> int median
    consensus = kg.suggest_parameters("temp", "s")
    assert consensus["k"] == 9
    assert isinstance(consensus["k"], int)


def test_suggest_parameters_empty_history_returns_empty():
    kg = KnowledgeGraph()
    assert kg.suggest_parameters("unseen", "s") == {}


def test_suggest_effect_exploration_returns_valid_effect():
    kg = KnowledgeGraph(exploration_rate=1.0)  # always explore
    random.seed(0)
    choice = kg.suggest_effect_for_feature("temp", ["l", "s", "f"])
    assert choice in {"l", "s", "f"}


def test_suggest_effect_exploitation_returns_valid_effect():
    kg = KnowledgeGraph(exploration_rate=0.0)  # always exploit via softmax
    random.seed(1)
    choice = kg.suggest_effect_for_feature("temp", ["l", "s"])
    assert choice in {"l", "s"}


def test_suggest_interaction_handles_empty_candidates():
    kg = KnowledgeGraph(exploration_rate=1.0)
    assert kg.suggest_interaction("temp", []) is None


def test_suggest_interaction_returns_candidate():
    kg = KnowledgeGraph(exploration_rate=0.0)
    random.seed(2)
    choice = kg.suggest_interaction("temp", ["humidity", "wind"])
    assert choice in {"humidity", "wind"}


def test_update_and_prune_drops_redundant_and_low_variance_terms():
    base = np.linspace(0.0, 10.0, 20)
    contributions = {
        "s(x)": base,            # strong, unique -> kept
        "f(x)": 2.0 * base,      # collinear with s(x) -> pruned as redundant
        "l(z)": base * 1e-4,     # negligible variance -> pruned
    }
    model = _FakeModel(contributions, estimate=base)

    parsed_terms = [
        {"type": "s", "feature": "x", "params": {"k": 10}},
        {"type": "f", "feature": "x", "params": {}},
        {"type": "l", "feature": "z", "params": {}},
        {"type": "w", "feature": "q", "params": {}},  # no contribution -> kept
    ]
    kg = KnowledgeGraph()
    df = pd.DataFrame({"dummy": np.arange(20)})

    pruned = kg.update_and_prune(parsed_terms, model, df, "y", global_rmse=5.0, target_std=10.0)
    kept = {(t["type"], t["feature"]) for t in pruned}

    assert ("s", "x") in kept
    assert ("w", "q") in kept  # kept because it had no decomposed contribution
    assert ("f", "x") not in kept
    assert ("l", "z") not in kept
    # The surviving spline registered a success in the graph.
    assert kg.effects["s"]["usage_count"] >= 1


def test_update_and_prune_returns_input_on_decompose_failure():
    class _Broken:
        def decompose_prediction(self, df):
            raise RuntimeError("boom")

        def predict(self, df):
            return pd.DataFrame()

    kg = KnowledgeGraph()
    terms = [{"type": "s", "feature": "x", "params": {}}]
    out = kg.update_and_prune(terms, _Broken(), pd.DataFrame(), "y", 1.0, 1.0)
    assert out == terms
