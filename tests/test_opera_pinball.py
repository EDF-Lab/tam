# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Pinball loss for ``OperaTAM`` (``loss_type='pinball', tau=...``) and ``aggregate_quantiles`` (one aggregation per level).

Aggregating quantile forecasts needs the loss of each level: the square and absolute losses aggregate means and medians.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.opera import aggregate_quantiles


def _two_biased_experts(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    return pd.DataFrame({"t": np.arange(n), "y": y, "high": np.full(n, 1.28), "low": np.full(n, -1.28)})


def _final_weights(frame: pd.DataFrame, **kwargs) -> np.ndarray:
    model = ta.OperaTAM(target_col="y", expert_cols=["high", "low"], date_col="t", **kwargs).fit(frame)
    return next(iter(model.weights_history_.values()))[-1]


def test_pinball_at_the_median_gives_the_absolute_loss_weights():
    # The pinball subgradient at tau = 0.5 is half the absolute one; MLpol's adaptive rates cancel a constant factor, up to
    # the fixed initial learning rate (2**-20), which does not rescale with the loss: equal to about 1e-4.
    frame = _two_biased_experts()
    frame["high"] = frame["y"] + 0.3 + 0.5 * np.random.default_rng(1).standard_normal(len(frame))
    absolute = _final_weights(frame, algorithm="MLPOL", loss_type="absolute")
    pinball = _final_weights(frame, algorithm="MLPOL", loss_type="pinball", tau=0.5)
    np.testing.assert_allclose(pinball, absolute, atol=1e-3)


@pytest.mark.parametrize("algorithm", ["MLPOL", "EWA"])
def test_high_levels_favour_the_high_expert(algorithm):
    # y ~ N(0, 1): the 0.9 quantile is +1.28 (the "high" expert), the 0.1 quantile is -1.28 (the "low" expert).
    frame = _two_biased_experts()
    upper = _final_weights(frame, algorithm=algorithm, loss_type="pinball", tau=0.9, eta=0.5)
    lower = _final_weights(frame, algorithm=algorithm, loss_type="pinball", tau=0.1, eta=0.5)
    assert upper[0] > 0.8 and lower[1] > 0.8


def test_aggregate_quantiles_columns_and_order():
    frame = _two_biased_experts(500)
    experts = {0.1: ["low", "high"], 0.5: ["low", "high"], 0.9: ["low", "high"]}
    table = aggregate_quantiles(frame, target_col="y", expert_cols_by_level=experts, date_col="t")
    assert list(table.columns) == ["q0.1", "q0.5", "q0.9"]
    assert len(table) == len(frame)
    assert (np.diff(table.to_numpy(), axis=1) >= 0).all()


def test_invalid_pinball_level_and_loss_are_rejected():
    with pytest.raises(ValueError, match="tau"):
        ta.OperaTAM(target_col="y", expert_cols=["a"], loss_type="pinball", tau=1.0)
    with pytest.raises(ValueError, match="loss_type"):
        ta.OperaTAM(target_col="y", expert_cols=["a"], loss_type="huber")
