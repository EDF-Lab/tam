# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The causal-state contract every stateful temporal spectrum effect follows.

For each effect, on synthetic data (3 groups, 400 steps) after ``initialize``:
  * idempotent: two calls give the same features and the same state;
  * split continuity: features on the second part given the state of the first equal the features of the whole series;
  * group independence: the features of a group do not depend on the other groups;
  * dtype: float64 in, float64 out;
  * a state of another group count raises ``ValueError``;
  * no fingerprint heuristic left on the effect.
"""

import pytest
import torch

from tam.model.spectrum._signature import SignatureEffect
from tam.model.spectrum._foundation import FoundationEffect
from tam.model.spectrum._inertia import InertiaEffect
from tam.model.spectrum._patch_basis import PatchBasisEffect
from tam.model.spectrum._single_index import SingleIndexEffect
from tam.model.spectrum._temporal_operator import TemporalOperatorEffect

B, T, SPLIT = 3, 400, 250

#: (id, factory, number of input columns). One line per stateful effect.
CASES = [
    pytest.param(lambda: SignatureEffect("x", w=6, N=2, lag=2), 1, id="sig"),
    pytest.param(lambda: SignatureEffect("x", w=5, N=3, lag=0, log_sig=True, lead_lag=True, additional_features=["z"]), 2, id="sig_multi"),
    pytest.param(lambda: SingleIndexEffect("x", lag_window=7, k=6), 1, id="si"),
    pytest.param(lambda: InertiaEffect("x", alphas=(0.3, 0.05, 0.002, 1.0), k=5), 1, id="inertia"),
    pytest.param(lambda: TemporalOperatorEffect("x", operator_type="attn", k_base=5, decay=0.9), 1, id="op_attn"),
    pytest.param(lambda: TemporalOperatorEffect("x", operator_type="hippo", inner_basis="fourier", k_base=4), 1, id="op_hippo"),
    pytest.param(lambda: TemporalOperatorEffect("x", operator_type="ssa", k_base=5, window=9), 1, id="op_ssa"),
    pytest.param(lambda: FoundationEffect("x", patch_len=8, n_tokens=6, n_bins=6, scales=(1, 4, 16), n_struct=8), 1, id="foundation"),
    pytest.param(lambda: FoundationEffect("x", additional_features=["z"], patch_len=6, n_tokens=5, n_bins=5, scales=(1, 8), n_struct=8), 2, id="foundation_multi"),
    pytest.param(lambda: PatchBasisEffect("x", patch_len=8, n_tokens=6), 1, id="patch_basis"),
    pytest.param(lambda: PatchBasisEffect("x", additional_features=["z"], patch_len=6, n_tokens=5), 2, id="patch_basis_multi"),
]


def _data(d: int, multi: bool = False, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    base = torch.cumsum(torch.randn(B, T, max(d, 1), dtype=torch.float64, generator=g), dim=1) * 0.05
    x = torch.tanh(base)
    x = x[..., 0] if (d == 1 and not multi) else x   # a one-column effect reads (B, T), or (B, T, 1) when it keeps the feature dimension
    return x.to(dtype)


def _slice(x: torch.Tensor, a, b) -> torch.Tensor:
    return x[:, a:b]


@pytest.mark.parametrize("factory, d", CASES)
def test_idempotent(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    p1 = e.build_feature_map(x)
    s1 = e._state_out
    p2 = e.build_feature_map(x)
    s2 = e._state_out
    assert torch.equal(p1, p2)
    assert set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)


@pytest.mark.parametrize("factory, d", CASES)
def test_split_continuity(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    whole = e.build_feature_map(x)
    e.build_feature_map(_slice(x, 0, SPLIT))
    state = {k: v.clone() for k, v in e._state_out.items()}
    second = e.build_feature_map(_slice(x, SPLIT, T), state_in=state)
    assert torch.allclose(second, whole[:, SPLIT:], atol=1e-8)


@pytest.mark.parametrize("factory, d", CASES)
def test_group_independence(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    whole = e.build_feature_map(x)
    for b in range(B):
        alone = e.build_feature_map(x[b:b + 1])
        assert torch.allclose(alone[0], whole[b], atol=1e-10), b


@pytest.mark.parametrize("factory, d", CASES)
def test_dtype_and_shape(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    phi = e.build_feature_map(x)
    assert phi.dtype == torch.float64 and phi.shape == (B, T, e.get_n_coeffs())
    assert torch.isfinite(phi).all()
    e32 = factory()
    x32 = _data(d, e32.multi_input, torch.float32)
    e32.initialize(x32)
    assert e32.build_feature_map(x32).dtype == torch.float32


@pytest.mark.parametrize("factory, d", CASES)
def test_state_of_another_group_count_raises(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    e.build_feature_map(x[:2])
    state = {k: v.clone() for k, v in e._state_out.items()}
    with pytest.raises(ValueError):
        e.build_feature_map(x, state_in=state)


@pytest.mark.parametrize("factory, d", CASES)
def test_no_fingerprint_heuristic(factory, d):
    e = factory()
    x = _data(d, e.multi_input)
    e.initialize(x)
    e.build_feature_map(x)
    for name in ("_start_fp", "_prev_fp", "last_window", "last_state", "_prev_init_window"):
        assert not hasattr(e, name), name
    assert e.stateful is True
