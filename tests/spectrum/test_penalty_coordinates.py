# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Several smoothing parameters per effect: the coordinates of the warped calendar, the thermal inertia and the temporal operators."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam import StaticTAM
from tam.model.spectrum._inertia import InertiaEffect, WarpedCalendarEffect
from tam.model.spectrum._temporal_operator import TemporalOperatorEffect

CASES = [
    pytest.param(lambda: WarpedCalendarEffect("tod", regime_feature="h", period=48.0, n_harmonics=3, lambda_p=0.3), ["base", "warp", "amp"], id="wcal_regime"),
    pytest.param(lambda: WarpedCalendarEffect("tod", period=48.0, n_harmonics=3, lambda_p=0.3), ["base"], id="wcal_plain"),
    pytest.param(lambda: InertiaEffect("t", alphas=(0.05, 0.01, 0.002), k=5, lambda_p=0.3), ["alpha=0.05", "alpha=0.01", "alpha=0.002"], id="inertia"),
    pytest.param(lambda: TemporalOperatorEffect("x", operator_type="hippo", k_base=5, lambda_p=0.3), ["base", "op"], id="op"),
]


@pytest.mark.parametrize("factory, labels", CASES)
def test_round_trip_and_labels(factory, labels):
    eff = factory()
    assert eff.coordinate_labels() == labels and eff.n_penalty_coordinates == len(labels)
    P0 = eff.build_penalty_matrix()
    eff.set_penalty_coordinates(eff.penalty_coordinates())
    assert torch.equal(eff.build_penalty_matrix(), P0)
    eff.set_penalty_coordinates([0.7])                       # one value: every coordinate times its multiplier
    assert eff.penalty_coordinates() == pytest.approx([0.7] * len(labels))
    with pytest.raises(ValueError, match="penalty coordinates, got"):
        eff.set_penalty_coordinates([1.0] * (len(labels) + 1))
    eff.lambda_p = 0.3                                       # assigning lambda_p resets every coordinate
    assert eff.penalty_coordinates() == pytest.approx([0.3] * len(labels)) and torch.equal(eff.build_penalty_matrix(), P0)


def test_default_penalty_is_the_one_before_the_split():
    """Frozen copies of the single-parameter constructions: the same bits at the defaults."""
    dtype = torch.get_default_dtype()
    lam = 0.3
    # warped calendar
    eff = WarpedCalendarEffect("tod", regime_feature="h", period=48.0, n_harmonics=3, lambda_p=lam, gamma_warp=1.0, gamma_amp=1.0)
    k_vec = torch.arange(1, 4, dtype=dtype)
    p_base = torch.cat([k_vec ** 2.0, k_vec ** 2.0])
    old = torch.diag(torch.cat([p_base, 1.0 * p_base, 1.0 * p_base]) * lam)
    assert torch.equal(eff.build_penalty_matrix(), old)
    # inertia
    inertia = InertiaEffect("t", alphas=(0.05, 0.01), k=5, lambda_p=lam)
    D = torch.diff(torch.eye(inertia._n_spline_coeffs, dtype=dtype), n=2, dim=0)
    block = lam * (D.T @ D)
    assert torch.equal(inertia.build_penalty_matrix(), torch.block_diag(block, block))
    # temporal operator
    op = TemporalOperatorEffect("x", operator_type="attn", k_base=5, lambda_p=lam)
    P_inner = op.inner_effect.build_penalty_matrix()
    K = op.k_inner
    expected = torch.zeros(2 * K, 2 * K, dtype=P_inner.dtype)
    expected[:K, :K] = P_inner
    expected[K:, K:] = P_inner + lam * torch.eye(K, dtype=P_inner.dtype)
    assert torch.equal(op.build_penalty_matrix(), expected)


def test_one_coordinate_moves_only_its_block():
    eff = WarpedCalendarEffect("tod", regime_feature="h", period=48.0, n_harmonics=3, lambda_p=0.3)
    P0 = eff.build_penalty_matrix()
    eff.set_penalty_coordinates([0.3, 3.0, 0.3])
    d = torch.diagonal(eff.build_penalty_matrix() - P0)
    assert d[:6].abs().sum() == 0 and d[6:12].abs().sum() > 0 and d[12:].abs().sum() == 0
    inertia = InertiaEffect("t", alphas=(0.05, 0.01, 0.002), k=5, lambda_p=0.3)
    Q0 = inertia.build_penalty_matrix()
    inertia.set_penalty_coordinates([0.3, 5.0, 0.3])
    n = inertia._n_spline_coeffs
    diff = (inertia.build_penalty_matrix() - Q0).abs()
    assert diff[:n].sum() == 0 and diff[n:2 * n, n:2 * n].sum() > 0 and diff[2 * n:].sum() == 0
    op = TemporalOperatorEffect("x", operator_type="ssa", k_base=5, lambda_p=0.3)
    R0 = op.build_penalty_matrix()
    op.set_penalty_coordinates([0.3, 2.0])
    K = op.k_inner
    diff = (op.build_penalty_matrix() - R0).abs()
    assert diff[:K, :K].sum() == 0 and diff[K:, K:].sum() > 0


def test_auto_fit_gives_one_lambda_per_coordinate():
    rng = np.random.default_rng(0)
    n = 480
    tod = np.tile(np.arange(48.0), n // 48)
    hol = np.repeat((np.arange(n // 48) % 3 == 0).astype(float), 48)
    temp = np.cumsum(rng.normal(size=n)) * 0.2
    y = np.sin(2 * np.pi * tod / 48) * (1 + 0.5 * hol) + 0.3 * np.tanh(temp) + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({"tod": tod, "h": hol, "temp": temp, "y": y})
    m = StaticTAM("y ~ wcal(tod, by=h, period=48, k=3) + inertia(temp, alphas='0.2|0.02', k=5)")
    m.auto_fit(df, alpha_p_bounds=(-4.0, 1.0), number_of_steps=3)
    counts = {type(e).__name__: e.n_penalty_coordinates for e in m.effects_list_}
    assert counts["WarpedCalendarEffect"] == 3 and counts["InertiaEffect"] == 2
    assert np.isfinite(m.predict(df)["Estimatedy"].to_numpy()).all()
