# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Data-dependent state (tree splits and leaf counts, RBF centres) is set from the full training data.

Before v1.3.1 it was set by the first design matrix built, which is the dispatcher's memory probe: one row per group (48 rows
for half-hourly data), or a single row, skipped, then the first chunk.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import build_phi_from_effects


def _panel(n_days=120, n_groups=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for g in range(n_groups):
            x = rng.uniform(-2, 2)
            rows.append({"date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=d), "g": g, "x": x,
                         "y": np.abs(x) + 0.3 * g + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def _tree(model):
    return [e for e in model.effects_list_ if e.effect_type == "tree"][0]


@pytest.mark.parametrize("group_col", ["g", None])
def test_leaf_counts_cover_every_training_row(group_col):
    df = _panel()
    model = ta.StaticTAM(formula="y ~ t(x, n_trees=10, max_depth=3, sp_alpha=1, split_strategy='quantile')",
                         group_col=group_col, date_col="date" if group_col else None)
    if group_col is None:
        df = df.drop(columns="date")
    model.fit(df)
    tree = _tree(model)
    assert int(tree.empirical_counts.sum()) == len(df) * tree.n_trees


def test_quantile_thresholds_come_from_the_training_distribution():
    df = _panel()
    model = ta.StaticTAM(formula="y ~ t(x, n_trees=1, max_leaves=8, split_strategy='quantile')",
                         group_col="g", date_col="date").fit(df)
    thresholds = _tree(model).split_thresholds.flatten().cpu().numpy()
    # n_trees=1 quantile trees split at the evenly spaced quantiles of the (normalised) feature over all rows
    x_stacked, _, _ = model._prepare_data(df, target_col=None)
    expected = np.quantile(x_stacked[..., 0].flatten().cpu().numpy(), np.linspace(0, 1, 9)[1:-1])
    np.testing.assert_allclose(np.sort(thresholds), expected, rtol=1e-6, atol=1e-9)


def test_a_probe_after_fit_changes_nothing():
    df = _panel()
    model = ta.StaticTAM(formula="y ~ t(x, n_trees=5, max_depth=2, sp_alpha=1, split_strategy='quantile') + rbf(x, n_centers=10)",
                         group_col="g", date_col="date").fit(df)
    tree = _tree(model)
    rbf = [e for e in model.effects_list_ if e.effect_type.startswith("rbf")][0]
    before = (tree.split_thresholds.clone(), tree.empirical_counts.clone(), rbf.centers.clone())
    x_stacked, _, _ = model._prepare_data(df, target_col=None)
    build_phi_from_effects(x_stacked[:, 0:1, :], model.effects_list_, feature_columns=model.features_config_["features"])
    assert torch.equal(tree.split_thresholds, before[0])
    assert torch.equal(tree.empirical_counts, before[1])
    assert torch.equal(rbf.centers, before[2])


def test_rbf_centres_are_training_points_of_every_group():
    """With 6 groups, the probe held 6 rows; the centres must be drawn from all 720."""
    df = _panel()
    model = ta.StaticTAM(formula="y ~ rbf(x, n_centers=30)", group_col="g", date_col="date").fit(df)
    rbf = [e for e in model.effects_list_ if e.effect_type.startswith("rbf")][0]
    assert len(torch.unique(rbf.centers)) > 6
