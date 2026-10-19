# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The IRLS loop of the non-Gaussian losses builds the design matrix once per fit, not once per pass and per step-halving.
The coefficients are those of the loop that rebuilt it.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model import _dispatcher, additive
from tam.model.statistics.estimation import _reweighting


def _poisson_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return pd.DataFrame({"x": x, "y": rng.poisson(np.exp(1.0 + np.sin(2.5 * x)))})


@pytest.mark.parametrize("loss, bound", [("poisson", 2), ("gamma", 2), ("negative_binomial", 12), ("tweedie", 12)])
def test_design_matrix_is_built_once_per_fit(loss, bound, monkeypatch):
    frame = _poisson_frame()
    if loss == "gamma":
        frame["y"] = frame["y"] + 0.5
    calls = []
    for module in (additive, _dispatcher):
        original = module.build_phi_from_effects

        def spy(x, *args, _original=original, **kwargs):
            if x.shape[1] > 1:  # the one-row probe of the dispatcher is not a build of the design matrix
                calls.append(1)
            return _original(x, *args, **kwargs)

        monkeypatch.setattr(module, "build_phi_from_effects", spy)
    ta.StaticTAM("y ~ s(x, k=10)", loss=loss).fit(frame)
    assert 1 <= len(calls) <= bound, len(calls)  # one per fit and per dispersion refit, never one per pass or step-halving


def test_cached_loop_matches_rebuilding_loop(monkeypatch):
    frame = _poisson_frame()
    cached = ta.StaticTAM("y ~ s(x, k=10)", loss="poisson").fit(frame)
    monkeypatch.setattr(_reweighting, "_DENSE_CACHE", False)
    rebuilt = ta.StaticTAM("y ~ s(x, k=10)", loss="poisson").fit(frame)
    assert torch.allclose(cached.coefficients_.cpu(), rebuilt.coefficients_.cpu(), atol=1e-10, rtol=0)


def test_gaussian_path_is_untouched():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    frame = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=400)})
    model = ta.StaticTAM("y ~ s(x, k=8)").fit(frame)
    assert getattr(model, "_reweighting_strategy_", None) is None
