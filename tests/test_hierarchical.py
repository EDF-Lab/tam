# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.hierarchical.HierarchicalTAM``, the joint solver
that links several StaticTAM nodes via aggregation constraints.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.hierarchical import HierarchicalTAM


@pytest.fixture
def hierarchy_data():
    """A 'Total = RegionA + RegionB' hierarchy with a shared feature/target schema."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=60)
    frames = []
    for node, base in [("Total", 100.0), ("RegionA", 60.0), ("RegionB", 40.0)]:
        frames.append(pd.DataFrame({
            "timestamp": dates,
            "node": node,
            "temperature": rng.uniform(-5, 30, 60),
            "load": rng.normal(base, 5, 60),
        }))
    return pd.concat(frames, ignore_index=True)


def _structure():
    return {"Total": ["RegionA", "RegionB"]}


def test_hierarchical_init_collects_nodes():
    model = HierarchicalTAM(
        structure=_structure(),
        formulas="load ~ l(temperature)",
        node_col="node",
        date_col="timestamp",
    )
    assert set(model.nodes) == {"Total", "RegionA", "RegionB"}
    assert set(model.sub_models.keys()) == {"Total", "RegionA", "RegionB"}


def test_hierarchical_fit_and_predict(hierarchy_data):
    model = HierarchicalTAM(
        structure=_structure(),
        formulas="load ~ s(temperature, k=5) + l(temperature)",
        node_col="node",
        date_col="timestamp",
        lambda_p_hier=1.0,
    )
    model.fit(hierarchy_data)

    # Every node receives fitted coefficients from the joint solve.
    for node in model.nodes:
        assert model.sub_models[node].coefficients_ is not None

    preds = model.predict(hierarchy_data)
    assert "Estimatedload" in preds.columns
    assert len(preds) == len(hierarchy_data)
    assert not preds["Estimatedload"].isna().all()


def test_hierarchical_per_node_formulas(hierarchy_data):
    formulas = {
        "Total": "load ~ l(temperature)",
        "RegionA": "load ~ l(temperature)",
        "RegionB": "load ~ s(temperature, k=4)",
    }
    model = HierarchicalTAM(
        structure=_structure(),
        formulas=formulas,
        node_col="node",
        date_col="timestamp",
    )
    model.fit(hierarchy_data)
    preds = model.predict(hierarchy_data)
    assert len(preds) == len(hierarchy_data)


def test_hierarchical_grid_search_with_token(hierarchy_data):
    model = HierarchicalTAM(
        structure=_structure(),
        formulas="load ~ s(temperature, k='grid_k')",
        node_col="node",
        date_col="timestamp",
    )
    fitted = model.grid_search_fit(
        hierarchy_data, hierarchy_data,
        grid_config={"grid_k": [4, 6]},
    )
    assert fitted is model
    preds = model.predict(hierarchy_data)
    assert "Estimatedload" in preds.columns


def test_hierarchical_grid_search_without_tokens_falls_back_to_fit(hierarchy_data):
    model = HierarchicalTAM(
        structure=_structure(),
        formulas="load ~ l(temperature)",
        node_col="node",
        date_col="timestamp",
    )
    fitted = model.grid_search_fit(hierarchy_data, hierarchy_data, grid_config={})
    assert fitted is model
    for node in model.nodes:
        assert model.sub_models[node].coefficients_ is not None
