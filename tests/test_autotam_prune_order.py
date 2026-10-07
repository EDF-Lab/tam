# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Unit tests for the evaluation order of ``KnowledgeGraph.update_and_prune``: near-collinear terms
are resolved strongest-first, and the surviving genome keeps its original order.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.knowledge_graph import KnowledgeGraph


class _FakeModel:
    """Exposes the decompose/predict interface the graph reads (no effects list)."""

    def __init__(self, contributions, estimate):
        self._contributions = contributions
        self._estimate = estimate

    def decompose_prediction(self, df):
        return self._contributions

    def predict(self, df):
        return pd.DataFrame({"Estimatedy": self._estimate})


def _prune(contributions, terms):
    estimate = sum(contributions.values())
    kg = KnowledgeGraph()
    kept = kg.update_and_prune(terms, _FakeModel(contributions, estimate), pd.DataFrame({"i": range(40)}),
                               "y", global_rmse=1.0, target_std=10.0)
    return kg, [(t["type"], t["feature"]) for t in kept]


def test_stronger_of_two_collinear_terms_survives_even_when_listed_second():
    rng = np.random.default_rng(0)
    curve = np.sin(np.linspace(0.0, 6.0, 40))
    contributions = {
        "l(weak)": 0.5 * curve,                    # listed first, collinear, less variance
        "f(strong)": 3.0 * curve,                  # listed second, collinear, more variance
        "s(other)": rng.normal(size=40),           # independent
    }
    terms = [{"type": "l", "feature": "weak", "params": {}},
             {"type": "f", "feature": "strong", "params": {}},
             {"type": "s", "feature": "other", "params": {}}]

    kg, kept = _prune(contributions, terms)

    assert kept == [("f", "strong"), ("s", "other")]
    assert kg.feature_effect_edges[("strong", "f")]["usage_count"] == 1
    assert kg.feature_effect_edges[("weak", "l")]["usage_count"] == 0


def test_surviving_genome_keeps_the_input_order():
    rng = np.random.default_rng(1)
    # Independent terms, all well above the 0.5% variance-share threshold, listed weakest first.
    contributions = {"l(a)": 1.5 * rng.normal(size=40), "l(b)": 5.0 * rng.normal(size=40),
                     "l(c)": 2.5 * rng.normal(size=40)}
    terms = [{"type": "l", "feature": name, "params": {}} for name in ("a", "b", "c")]

    _, kept = _prune(contributions, terms)

    assert kept == [("l", "a"), ("l", "b"), ("l", "c")]


def test_terms_without_a_contribution_are_kept_in_place():
    curve = np.linspace(-1.0, 1.0, 40)
    contributions = {"s(x)": curve}
    terms = [{"type": "w", "feature": "unresolved", "params": {}},
             {"type": "s", "feature": "x", "params": {}}]

    _, kept = _prune(contributions, terms)

    assert kept == [("w", "unresolved"), ("s", "x")]
