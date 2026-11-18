# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import StaticTAM
from tam.model.spectrum._foundation import (
    FoundationEffect, MultiRateFilterbankEffect, PFNRidgeEffect, QuantileHistEffect, StructuralPriorEffect, _build_d2_penalty,
)

def test_foundation_basis_univariate_and_multivariate():
    torch.manual_seed(7)
    T = 80
    x1 = torch.sin(torch.linspace(0, 6.0, T, dtype=torch.float64))
    x2 = torch.cos(torch.linspace(0, 6.0, T, dtype=torch.float64))
    x_multi = torch.stack([x1, x2], dim=-1)  # (T, 2)

    eff_uni = FoundationEffect(
        feature_name="x1",
        patch_len=8,
        n_tokens=8,
        n_bins=8,
        scales=(1, 4, 16),
        n_pfn=8,
        include_levy=True,
    )
    # Causality check
    phi_u1 = eff_uni.build_feature_map(x1)
    x1_mod = x1.clone()
    x1_mod[40:] -= 3.0
    phi_u2 = eff_uni.build_feature_map(x1_mod)
    assert torch.max(torch.abs(phi_u1[:40] - phi_u2[:40])).item() < 1e-12

    # State continuation check
    phi_p1 = eff_uni.build_feature_map(x1[:45])
    st = eff_uni._state_out
    phi_p2 = eff_uni.build_feature_map(x1[45:], state_in=st)
    assert torch.max(torch.abs(phi_u1 - torch.cat([phi_p1, phi_p2], dim=0))).item() < 1e-10

    # Multivariate with Levy area hysteresis tokens
    eff_multi = FoundationEffect(
        feature_name="x1",
        additional_features=["x2"],
        patch_len=8,
        n_tokens=8,
        n_bins=8,
        scales=(1, 4),
        n_pfn=8,
        include_levy=True,
    )
    phi_m = eff_multi.build_feature_map(x_multi)
    assert phi_m.shape == (T, eff_multi.get_n_coeffs())
    P = eff_multi.build_penalty_matrix()
    assert P.shape == (eff_multi.get_n_coeffs(), eff_multi.get_n_coeffs())
    assert torch.all(torch.isfinite(phi_m))


def test_foundation_token_in_statictam():
    rng = np.random.default_rng(123)
    n = 100
    x1 = np.sin(np.linspace(0, 3 * np.pi, n))
    x2 = np.cos(np.linspace(0, 3 * np.pi, n))
    y = 1.5 * x1 - 0.8 * x2 + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    model = StaticTAM("y ~ foundation(x1, others=x2, patch_len=8, n_tokens=6, n_bins=6, n_pfn=8) + patch_basis(x1, patch_len=6, n_tokens=4)")
    model.fit(df.iloc[:70].reset_index(drop=True))
    out = model.predict(df.iloc[70:].reset_index(drop=True), continue_state=True)
    assert len(out) == 30
    assert np.all(np.isfinite(out["Estimatedy"].to_numpy()))


def test_foundation_components_are_not_formula_tokens():
    df = pd.DataFrame({"x1": np.sin(np.linspace(0, 6, 60)), "y": np.cos(np.linspace(0, 6, 60))})
    for token in ("quant_hist(x1)", "mrfb(x1)", "pfn_ridge(x1)"):
        with pytest.raises(ValueError, match="Unknown effect type identifier"):
            StaticTAM("y ~ " + token).fit(df)


def test_n_pfn_is_an_alias_of_n_struct():
    kw = dict(patch_len=8, n_tokens=6, n_bins=6, scales=(1, 4))
    x = torch.sin(torch.linspace(0, 6.0, 60, dtype=torch.float64))
    a = FoundationEffect("x", n_pfn=8, **kw)
    b = FoundationEffect("x", n_struct=8, **kw)
    assert a.n_struct == b.n_struct == 8
    assert torch.equal(a.build_feature_map(x), b.build_feature_map(x))
    FoundationEffect("x", n_struct=8, n_pfn=8, **kw)
    with pytest.raises(ValueError, match="alias"):
        FoundationEffect("x", n_struct=8, n_pfn=12, **kw)
    # the same in a formula
    df = pd.DataFrame({"x": np.sin(np.linspace(0, 6, 80)), "y": np.cos(np.linspace(0, 6, 80))})
    pa = StaticTAM("y ~ foundation(x, patch_len=8, n_tokens=6, n_bins=6, n_pfn=8)").fit(df).predict(df)["Estimatedy"].to_numpy()
    pb = StaticTAM("y ~ foundation(x, patch_len=8, n_tokens=6, n_bins=6, n_struct=8)").fit(df).predict(df)["Estimatedy"].to_numpy()
    assert np.array_equal(pa, pb)
    with pytest.raises(ValueError, match="alias"):
        StaticTAM("y ~ foundation(x, n_struct=8, n_pfn=12)").fit(df)


def test_struct_prior_names_and_no_formula_token():
    assert PFNRidgeEffect is StructuralPriorEffect
    assert StructuralPriorEffect("x").effect_type == "struct_prior"
    df = pd.DataFrame({"x1": np.sin(np.linspace(0, 6, 60)), "y": np.cos(np.linspace(0, 6, 60))})
    for token in ("struct_prior(x1)", "pfn_ridge(x1)"):
        with pytest.raises(ValueError, match="Unknown effect type identifier"):
            StaticTAM("y ~ " + token).fit(df)


def test_foundation_source_names_no_product():
    import inspect
    import re

    from tam.model.spectrum import _foundation

    src = inspect.getsource(_foundation)    # the module itself, so the test also runs against an installed wheel
    assert re.search(r"TabPFN|TimesFM|Chronos|MOIRAI|PatchTST|Lag-Llama", src) is None


def _dense_mrfb(scales, x, ema_prev=None, env_prev=None):
    """The dense (T x T) filterbank the recurrences replace (one group)."""
    B, T = x.shape
    S = len(scales)
    alphas = torch.tensor([2.0 / (float(s) + 1.0) for s in scales], dtype=x.dtype)
    gammas = 1.0 - alphas
    ema_prev = x[:, :1].expand(B, S).clone() if ema_prev is None else ema_prev
    env_prev = torch.zeros((B, S), dtype=x.dtype) if env_prev is None else env_prev
    idx = torch.arange(T, dtype=x.dtype)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    g_view, a_view = gammas.view(1, 1, S), alphas.view(1, 1, S)
    d_view = torch.clamp(diff, min=0.0).unsqueeze(-1)
    w_ema = torch.where((diff >= 0).unsqueeze(-1), a_view * (g_view ** d_view), torch.zeros((T, T, S), dtype=x.dtype))
    g_pows = gammas.unsqueeze(0) ** (idx.unsqueeze(1) + 1.0)
    ema_seq = torch.einsum("tsk,bs->btk", w_ema, x) + g_pows.unsqueeze(0) * ema_prev.unsqueeze(1)
    band_seq = x.unsqueeze(-1) - ema_seq
    band_sq = torch.tanh(0.25 * band_seq * band_seq)
    env_seq = torch.einsum("tsk,bsk->btk", w_ema, band_sq) + g_pows.unsqueeze(0) * env_prev.unsqueeze(1)
    return torch.cat([ema_seq, band_seq, env_seq], dim=-1)


def test_filterbank_recurrence_equals_the_dense_construction():
    g = torch.Generator().manual_seed(4)
    x = torch.tanh(torch.cumsum(torch.randn(2, 300, dtype=torch.float64, generator=g), 1) * 0.07)
    eff = MultiRateFilterbankEffect("x", scales=(1, 4, 16, 64))
    got = eff.build_feature_map(x.unsqueeze(-1))
    assert torch.allclose(got, _dense_mrfb((1, 4, 16, 64), x), atol=1e-9)


def test_foundation_fifty_thousand_steps_in_linear_memory():
    import time

    x = torch.tanh(torch.cumsum(torch.randn(1, 50_000, 1, dtype=torch.float64), 1) * 0.01)
    eff = FoundationEffect("x", patch_len=16, n_tokens=8, n_bins=8, scales=(1, 4, 16, 64), n_struct=8)
    t0 = time.time()
    phi = eff.build_feature_map(x)
    assert phi.shape == (1, 50_000, eff.get_n_coeffs()) and torch.isfinite(phi).all()
    assert time.time() - t0 < 20.0


def test_foundation_penalty_coordinates():
    eff = FoundationEffect("x", patch_len=8, n_tokens=6, n_bins=6, scales=(1, 4), n_struct=8, lambda_p=0.02)
    assert eff.n_penalty_coordinates == len(eff.coordinate_labels()) == 4
    assert eff.coordinate_labels() == ["patch", "hist", "mrfb", "struct"]
    P0 = eff.build_penalty_matrix()
    eff.set_penalty_coordinates(eff.penalty_coordinates())
    assert torch.equal(eff.build_penalty_matrix(), P0)
    eff.set_penalty_coordinates([0.5])
    assert eff.penalty_coordinates() == [0.5] * 4 and eff.lambda_p == 0.5
    with pytest.raises(ValueError, match="4 penalty coordinates, got 3"):
        eff.set_penalty_coordinates([1.0, 2.0, 3.0])
    eff.lambda_p = 0.02                                     # assigning lambda_p sets all four blocks
    assert eff.penalty_coordinates() == [0.02] * 4 and torch.equal(eff.build_penalty_matrix(), P0)
    # independence: moving one coordinate moves only its block
    sizes = [e.get_n_coeffs() for e in eff.sub_effects]
    eff.set_penalty_coordinates([0.02, 0.2, 0.02, 0.02])
    P1 = eff.build_penalty_matrix()
    d = (P1 - P0).abs()
    start = sizes[0]
    assert d[:start].sum() == 0 and d[start:start + sizes[1], start:start + sizes[1]].sum() > 0
    assert d[start + sizes[1]:].sum() == 0


def test_foundation_penalty_blocks_at_the_defaults():
    """Histogram and filterbank blocks are the original ones; the patch tokens carry roughness weights, the structural block is second-difference on pulses and heavy tails."""
    lam = 0.03
    eff = FoundationEffect("x", patch_len=8, n_tokens=6, n_bins=8, scales=(1, 4, 16), n_struct=16, lambda_p=lam)
    P = eff.build_penalty_matrix()
    dtype = torch.get_default_dtype()
    cpu = torch.device("cpu")
    kp, kh, km = eff.patch_effect.get_n_coeffs(), eff.hist_effect.get_n_coeffs(), eff.mrfb_effect.get_n_coeffs()
    assert torch.equal(P[:kp, :kp], eff.patch_effect.build_penalty_matrix())          # the patch block carries the roughness weights of its tokens
    assert torch.equal(P[kp:kp + kh, kp:kp + kh], lam * _build_d2_penalty(8, cpu, dtype))
    weights = [m * (float(s) ** 0.5) for m in (1.0, 1.5, 2.0) for s in (1, 4, 16)]
    assert torch.equal(P[kp + kh:kp + kh + km, kp + kh:kp + kh + km], lam * torch.diag(torch.tensor(weights, dtype=dtype)))
    S = P[kp + kh + km:, kp + kh + km:]
    assert S.shape == (16, 16) and float(torch.linalg.eigvalsh(S).min()) > 0
    assert torch.allclose(S, S.T)
    n = 4   # blocks of 4: sigmoids, hinges (diagonal), pulses, heavy tails (second difference, scaled by the block weight)
    assert torch.equal(S[:n, :n], lam * torch.eye(n, dtype=dtype)) and torch.equal(S[n:2 * n, n:2 * n], lam * 1.2 * torch.eye(n, dtype=dtype))
    assert torch.allclose(S[2 * n:3 * n, 2 * n:3 * n], lam * 2.0 * _build_d2_penalty(n, cpu, dtype))
    assert torch.allclose(S[3 * n:, 3 * n:], lam * 1.5 * _build_d2_penalty(n, cpu, dtype))
    assert S[:2 * n, 2 * n:].abs().sum() == 0


def test_foundation_auto_fit_finds_four_lambdas():
    rng = np.random.default_rng(0)
    n = 300
    x = np.sin(np.linspace(0, 12, n)) + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": np.tanh(x) + 0.05 * rng.standard_normal(n)})
    m = StaticTAM("y ~ foundation(x, patch_len=8, n_tokens=6, n_bins=6, scales=(1, 4), n_struct=8)")
    m.auto_fit(df, alpha_p_bounds=(-5.0, 1.0), number_of_steps=3)
    eff = [e for e in m.effects_list_ if isinstance(e, FoundationEffect)][0]
    assert len(eff.penalty_coordinates()) == 4
    assert np.isfinite(m.predict(df)["Estimatedy"].to_numpy()).all()


def test_foundation_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([FoundationEffect("x", additional_features=["z"])]) == {"x", "z"}
    # the internal components are not read as features of their own
    assert extrapolating_features([StructuralPriorEffect("x"), QuantileHistEffect("x"), MultiRateFilterbankEffect("x")]) == set()


def test_second_difference_penalty_carries_a_small_ridge():
    """`_build_d2_penalty(n)` is D2'D2 + 1e-6 I: the ridge keeps it positive definite, so the REML log-determinant is defined."""
    p = _build_d2_penalty(8, torch.device("cpu"), torch.float64)
    assert float(torch.linalg.eigvalsh(p).min()) > 0.5e-6
