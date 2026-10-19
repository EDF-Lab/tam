# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
A full Newton step of the reweighted loop lowers the penalized objective the step-halving judges, whatever the mean working weight:
about 550 for Poisson counts of mean 500, about 0.24 for binomial. The weights are normalised to unit mean before the solve (so that
``ap`` does not depend on the units of the target); this pins that the objective and the solve stay compatible with that convention.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.common.utils import _ensure_dummies, _balance_groups
from tam.model.statistics.estimation import _reweighting


def _prepared(loss, ap):
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 3000)
    mean = np.exp(0.6 * np.sin(3 * x))
    y = rng.poisson(500 * mean) if loss == "poisson" else rng.binomial(1, mean / (1 + mean))
    model = ta.StaticTAM(f"y ~ s(x, k=25, ap={ap})", loss=loss)
    data = _ensure_dummies(pd.DataFrame({"x": x, "y": y}), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(data, model.group_col_, model.date_col_, method="drop")
    x_t, y_t, _ = model._prepare_data(balanced, target_col=model.target_col_)
    return model, x_t, y_t.to(torch.get_default_dtype())


@pytest.mark.parametrize("loss", ["poisson", "binomial"])
@pytest.mark.parametrize("ap", [-6, -2, 0, 2])
def test_full_step_lowers_the_penalized_objective(loss, ap):
    model, x, y = _prepared(loss, ap)
    strategy = model._reweighting_strategy_
    theta = _reweighting.reweighted_penalized_fit(model, x, y, strategy)
    start = theta + 0.05 * torch.randn(theta.shape, dtype=theta.dtype, generator=torch.Generator().manual_seed(1))
    penalty = model._build_penalty_matrix().to(theta.dtype)
    eta = _reweighting._linear_predictor(model, x, start)
    z, w = strategy.working_response_and_weights(y, eta)
    step = model._solve_pwls_step(x, z, weights=w / w.mean())
    eta_step = _reweighting._linear_predictor(model, x, step)

    def objective(t, e):
        return (strategy.mean_objective(y, e) + (t.mT @ penalty @ t).mean()).item()

    assert objective(step, eta_step) <= objective(start, eta)
