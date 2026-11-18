# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import torch

from tam import StaticTAM
from tam.model.spectrum._patch_basis import PatchBasisEffect


def _series(n=80):
    t = torch.linspace(0, 6.0, n, dtype=torch.float64)
    return torch.sin(t), torch.cos(t)


def test_patch_basis_causality_and_state_continuation():
    x1, _ = _series()
    eff = PatchBasisEffect(feature_name="x1", patch_len=8, n_tokens=8, include_levy=False)
    phi_full = eff.build_feature_map(x1)
    x_mod = x1.clone()
    x_mod[40:] -= 3.0
    phi_mod = eff.build_feature_map(x_mod)
    assert torch.max(torch.abs(phi_full[:40] - phi_mod[:40])).item() < 1e-12

    phi_a = eff.build_feature_map(x1[:45])
    state = eff._state_out
    phi_b = eff.build_feature_map(x1[45:], state_in=state)
    assert torch.max(torch.abs(phi_full - torch.cat([phi_a, phi_b], dim=0))).item() < 1e-10


def test_patch_basis_multivariate_levy_tokens_and_penalty():
    x1, x2 = _series()
    eff = PatchBasisEffect(feature_name="x1", additional_features=["x2"], patch_len=8, n_tokens=8, include_levy=True)
    phi = eff.build_feature_map(torch.stack([x1, x2], dim=-1))
    n_coeffs = eff.get_n_coeffs()
    assert phi.shape == (x1.shape[0], n_coeffs)
    penalty = eff.build_penalty_matrix()
    assert penalty.shape == (n_coeffs, n_coeffs)
    assert torch.all(torch.isfinite(phi))
    assert torch.min(torch.linalg.eigvalsh((penalty + penalty.T) / 2)).item() > -1e-10


def test_patch_basis_token_in_statictam():
    rng = np.random.default_rng(123)
    n = 100
    x1 = np.sin(np.linspace(0, 3 * np.pi, n))
    y = 1.5 * x1 + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"x1": x1, "y": y})
    model = StaticTAM("y ~ patch_basis(x1, patch_len=6, n_tokens=4)")
    model.fit(df.iloc[:70].reset_index(drop=True))
    out = model.predict(df.iloc[70:].reset_index(drop=True), continue_state=True)
    assert len(out) == 30
    assert np.all(np.isfinite(out["Estimatedy"].to_numpy()))


def test_patch_basis_series_shorter_than_the_patch_and_groups():
    """A pass shorter than the patch works from replicated history; each group is tokenized alone."""
    eff = PatchBasisEffect(feature_name="x", patch_len=16, n_tokens=6)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 5, dtype=torch.float64, generator=g)
    short = eff.build_feature_map(x)
    assert short.shape == (3, 5, eff.get_n_coeffs()) and torch.isfinite(short).all()
    long_x = torch.randn(3, 100, dtype=torch.float64, generator=g)
    whole = eff.build_feature_map(long_x)
    assert torch.allclose(whole[1], eff.build_feature_map(long_x[1:2])[0], atol=1e-12)
    assert eff._state_out["hist"].shape == (1, 15, 1)


def test_patch_basis_refuses_isolated_point_extrapolation():
    import pytest

    for mode in ("linear", "saturation"):
        with pytest.raises(ValueError, match="temporal effect"):
            PatchBasisEffect(feature_name="x", extrapolate=mode)


def test_patch_basis_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([PatchBasisEffect(feature_name="x", additional_features=["z"])]) == {"x", "z"}


def test_patch_basis_penalty_weights_the_tokens_by_the_roughness_of_their_filter():
    """A smooth patch filter is penalized less than an oscillating one; the other columns keep a ridge."""
    eff = PatchBasisEffect(feature_name="x", patch_len=8, n_tokens=4, include_levy=False, lambda_p=2.0)
    t = torch.linspace(0, 1, 8, dtype=torch.float64)
    eff._w_patch = torch.stack([t, 0.5 * t, torch.cos(3.0 * torch.pi * t), torch.sin(4.0 * torch.pi * t)])   # two smooth, two oscillating
    P = eff.build_penalty_matrix()
    d = torch.diagonal(P)
    assert d[0] < d[2] and d[1] < d[3] and d[1] <= d[0]
    rough = (eff._w_patch[:, 2:] - 2 * eff._w_patch[:, 1:-1] + eff._w_patch[:, :-2]).pow(2).sum(1)
    assert torch.allclose(d[:4], 2.0 * (1.0 + rough / rough.mean()).to(d.dtype))
    assert torch.allclose(d[4:], torch.full_like(d[4:], 2.0))                   # the two RevIN statistics
    assert float(torch.linalg.eigvalsh(P).min()) > 0 and torch.allclose(P, P.T)
    short = PatchBasisEffect(feature_name="x", patch_len=2, n_tokens=3, include_levy=False)
    assert torch.allclose(torch.diagonal(short.build_penalty_matrix()), torch.full((short.n_coeffs,), short.lambda_p, dtype=P.dtype))
