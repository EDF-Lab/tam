# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
GCV and the solver use the same ridge floor (``tam.model._math._ridge_floor``).

``auto_fit`` returns the coefficients of the system GCV scored; refitting the model with the selected penalties must give the
same coefficients, for every basis family.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model._math import _ridge_floor, compute_gcv_score, solve_linear_system


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    z = rng.uniform(0, 1, n)
    c = rng.integers(0, 4, n)
    y = np.sin(x) + 2 * z ** 2 + 0.5 * c + x * z + rng.normal(0, 0.2, n)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x": x, "z": z, "c": c, "y": y})


@pytest.mark.parametrize("rhs", [
    "s(x, k=10) + l(z)",
    "f(x, m=4) + s(z, k=8)",
    "c(c, n_cat=4, topo='nominal') + s(x, k=10)",
    "te(s(x, k=6), s(z, k=6)) + l(x)",
])
def test_auto_fit_then_fit_gives_the_same_coefficients(rhs):
    df = _data()
    model = ta.StaticTAM(formula=f"y ~ {rhs}", date_col="date")
    model.auto_fit(df, number_of_steps=6)
    selected = model.coefficients_.detach().clone()
    model.fit(df)
    np.testing.assert_allclose(model.coefficients_.detach().cpu().numpy(), selected.cpu().numpy(), rtol=1e-8, atol=1e-10)


def test_the_ridge_floor_scales_with_the_sample_count():
    floor = _ridge_floor(200, 3, torch.float64, "cpu")
    np.testing.assert_allclose(floor.numpy(), 200e-6 * np.eye(3))


def test_gcv_scores_the_solver_coefficients():
    """The RSS behind the GCV score is the one of the coefficients solve_linear_system returns."""
    rng = np.random.default_rng(1)
    n, k = 300, 5
    phi = torch.tensor(rng.normal(size=(1, n, k)))
    y = phi @ torch.tensor(rng.normal(size=(k, 1))) + 0.1 * torch.tensor(rng.normal(size=(1, n, 1)))
    cov_x, cov_xy, y_sq = phi.mT @ phi, phi.mT @ y, (y ** 2).sum(dim=(1, 2))
    penalty = torch.eye(k, dtype=torch.float64) * 1e-3
    beta = solve_linear_system(cov_x, cov_xy, penalty, n)
    rss = float(((y - phi @ beta.cpu()) ** 2).sum())
    trace = float(torch.einsum("bii->b", cov_x @ torch.linalg.inv(cov_x + n * penalty + _ridge_floor(n, k, torch.float64, "cpu"))))
    expected = (rss / n) / (1 - trace / n) ** 2
    score = compute_gcv_score(cov_x, cov_xy, y_sq, penalty, lambda_p=1.0, n_samples=n, gamma=1.0)
    assert float(score) == pytest.approx(expected, rel=1e-9)
