# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import StaticTAM
from tam.model.spectrum._temporal_operator import TemporalOperatorEffect

@pytest.mark.parametrize("op_type", ["attn", "hippo", "ssa"])
def test_temporal_operator_causality_and_state_continuation(op_type: str):
    torch.manual_seed(0)
    x = torch.linspace(-0.9, 0.9, 60, dtype=torch.float64) + 0.1 * torch.sin(torch.arange(60, dtype=torch.float64))
    eff = TemporalOperatorEffect(
        feature_name="x",
        operator_type=op_type,
        inner_basis="spline",
        k_base=6,
        decay=0.9,
        window=8,
        lambda_p=1e-2,
    )

    # 1. Causality: mutating future steps t >= 30 must not change features at t < 30
    phi_orig = eff.build_feature_map(x)
    x_mod = x.clone()
    x_mod[30:] += 5.0
    phi_mod = eff.build_feature_map(x_mod)
    assert torch.max(torch.abs(phi_orig[:30] - phi_mod[:30])).item() < 1e-12

    # 2. Exact state continuation across [0, 35) and [35, 60)
    phi_full = eff.build_feature_map(x)
    phi_part1 = eff.build_feature_map(x[:35])
    state_mid = eff._state_out
    phi_part2 = eff.build_feature_map(x[35:], state_in=state_mid)
    phi_cat = torch.cat([phi_part1, phi_part2], dim=0)
    assert torch.max(torch.abs(phi_full - phi_cat)).item() < 1e-10

    # 3. Penalty matrix is symmetric positive semi-definite
    P = eff.build_penalty_matrix()
    assert P.shape == (eff.get_n_coeffs(), eff.get_n_coeffs())
    eigvals = torch.linalg.eigvalsh(0.5 * (P + P.mT))
    assert float(torch.min(eigvals).item()) >= -1e-10



def test_temporal_operator_statictam_integration():
    rng = np.random.default_rng(42)
    n = 120
    t = np.linspace(0, 4 * np.pi, n)
    x = np.sin(t) + 0.1 * rng.standard_normal(n)
    y = 2.0 * x + 0.5 * np.roll(x, 1) + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})

    model = StaticTAM("y ~ op_attn(x, inner='spline', k_base=6, decay=0.85) + op_hippo(x, inner='fourier', k_base=4)")
    model.fit(df.iloc[:80].reset_index(drop=True))
    preds = model.predict(df.iloc[80:].reset_index(drop=True), continue_state=True)
    assert "Estimatedy" in preds.columns
    assert np.all(np.isfinite(preds["Estimatedy"].to_numpy()))


# ----------------------------------------------------------------------------------------------------------------------
# The operators are linear recurrences / windows: they equal the dense (T x T) construction they replace, in O(T) memory.
# ----------------------------------------------------------------------------------------------------------------------
def _dense_reference(eff: TemporalOperatorEffect, x: torch.Tensor) -> torch.Tensor:
    """The dense T x T form of the three operators (the construction the recurrences replace); one group, no carried state."""
    phi_base = eff.inner_effect.transform(x).unsqueeze(0)  # (1, T, K)
    B, T, K = phi_base.shape
    dtype, device = phi_base.dtype, phi_base.device
    idx = torch.arange(T, device=device, dtype=dtype)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    causal_mask = diff >= 0
    if eff.operator_type == "attn":
        M = eff.n_random_features
        proj_w = eff._proj_w.to(device=device, dtype=dtype)
        u_norm = torch.clamp(phi_base, -4.0, 4.0)
        proj = u_norm @ proj_w
        norm_sq = 0.5 * torch.sum(u_norm * u_norm, dim=-1, keepdim=True) / max(float(K), 1.0)
        phi_rf = torch.exp(torch.clamp(proj - norm_sq, min=-10.0, max=6.0))
        kv_outer = torch.einsum("btm,btk->btmk", phi_rf, phi_base)
        weights_mat = torch.where(causal_mask, (1.0 - eff.decay) * (eff.decay ** torch.clamp(diff, min=0.0)), torch.zeros_like(diff))
        S_seq = torch.einsum("ts,bsmk->btmk", weights_mat, kv_outer)
        Z_seq = torch.einsum("ts,bsm->btm", weights_mat, phi_rf)
        num = torch.einsum("btm,btmk->btk", phi_rf, S_seq)
        den = torch.clamp(torch.einsum("btm,btm->bt", phi_rf, Z_seq).unsqueeze(-1), min=1e-8)
        mixed = num / den
    elif eff.operator_type == "hippo":
        gammas = eff._hippo_gammas.to(device=device, dtype=dtype)
        g_view = gammas.view(1, 1, K)
        d_view = torch.clamp(diff, min=0.0).unsqueeze(-1)
        w_hippo = torch.where(causal_mask.unsqueeze(-1), (1.0 - g_view) * (g_view ** d_view), torch.zeros((T, T, K), dtype=dtype, device=device))
        g_pows = gammas.unsqueeze(0) ** (idx.unsqueeze(1) + 1.0)
        mixed = torch.einsum("tsk,bsk->btk", w_hippo, phi_base) + g_pows.unsqueeze(0) * phi_base[:, :1, :]
    else:
        W = eff.window
        w_ssa = eff._ssa_weights.to(device=device, dtype=dtype)
        buf = phi_base[:, :1, :].expand(B, W - 1, K)
        windows = torch.cat([buf, phi_base], dim=1).unfold(dimension=1, size=W, step=1)
        mixed = torch.einsum("btkw,w->btk", torch.flip(windows, dims=[-1]), w_ssa)
    return torch.cat([phi_base, mixed], dim=-1)[0]


@pytest.mark.parametrize("op_type", ["attn", "hippo", "ssa"])
def test_recurrences_equal_the_dense_construction(op_type):
    g = torch.Generator().manual_seed(3)
    x = torch.tanh(torch.cumsum(torch.randn(300, dtype=torch.float64, generator=g), 0) * 0.08)
    eff = TemporalOperatorEffect("x", operator_type=op_type, k_base=6, decay=0.93, window=11)
    phi = eff.build_feature_map(x)
    ref = _dense_reference(eff, x)
    # the dense reference starts the hippo state at the first basis row and the attn state at zero, as the recurrences do
    assert torch.allclose(phi, ref, atol=1e-9), float((phi - ref).abs().max())


@pytest.mark.parametrize("op_type", ["attn", "hippo", "ssa"])
def test_fifty_thousand_steps_in_linear_memory(op_type):
    """A dense T x T matrix at T = 50 000 would need 20 GB; the recurrences do it in a few seconds."""
    import time

    x = torch.tanh(torch.cumsum(torch.randn(1, 50_000, 1, dtype=torch.float64), 1) * 0.01)
    eff = TemporalOperatorEffect("x", operator_type=op_type, k_base=6)
    t0 = time.time()
    phi = eff.build_feature_map(x)
    assert phi.shape == (1, 50_000, eff.get_n_coeffs()) and torch.isfinite(phi).all()
    assert time.time() - t0 < 20.0


def test_temporal_operator_groups_and_state_shapes():
    """Three groups, each mixed alone; the state has one row per group; the inner basis is initialized on the primary input."""
    eff = TemporalOperatorEffect("x", operator_type="attn", k_base=5, additional_features=["z"])
    assert eff.multi_input is True and eff.stateful is True
    g = torch.Generator().manual_seed(1)
    x = torch.tanh(torch.randn(3, 90, 2, dtype=torch.float64, generator=g))
    eff.initialize(x)
    whole = eff.build_feature_map(x)
    assert eff._state_out["S"].shape[0] == 3 and eff._state_out["Z"].shape[0] == 3
    assert torch.allclose(whole[2], eff.build_feature_map(x[2:3])[0], atol=1e-12)


def test_temporal_operator_refuses_isolated_point_extrapolation():
    for mode in ("linear", "saturation"):
        with pytest.raises(ValueError, match="temporal effect"):
            TemporalOperatorEffect("x", extrapolate=mode)


def test_temporal_operator_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([TemporalOperatorEffect("x", additional_features=["z"])]) == {"x", "z"}
