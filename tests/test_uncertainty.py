# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for the epistemic (Bayesian-GAM posterior) prediction interval."""
import numpy as np
import pandas as pd

from tam import StaticTAM
from tam.model.statistics.risk.uncertainty import posterior_prediction


def test_epistemic_std_widens_where_data_is_sparse():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 2000)
    train = pd.DataFrame({"x": x, "y": np.sin(3.0 * x) + 0.1 * rng.standard_normal(2000)})
    model = StaticTAM("y ~ s(x)").fit(train)
    grid = pd.DataFrame({"x": [0.5, 0.95, 1.4]})
    result = posterior_prediction(model, train, grid)
    standard_error = result["epistemic_std"].to_numpy()
    assert standard_error[0] < standard_error[1] < standard_error[2]
    assert np.all(result["lower"].to_numpy() <= result["prediction"].to_numpy())
    assert np.all(result["prediction"].to_numpy() <= result["upper"].to_numpy())
