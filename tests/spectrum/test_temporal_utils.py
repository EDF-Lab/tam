# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the shared helpers of the causal temporal spectrum effects."""

import pytest
import torch

from tam.model.spectrum._temporal_utils import (
    causal_history,
    causal_linear_recurrence,
    causal_window_unfold,
    from_btk,
    tail_state,
    to_btd,
)


@pytest.mark.parametrize("shape, multi, layout, x3_shape", [
    ((3, 20, 4), False, "btd", (3, 20, 4)),
    ((3, 20, 4), True, "btd", (3, 20, 4)),
    ((3, 20), False, "bt", (3, 20, 1)),
    ((15, 4), True, "nd", (1, 15, 4)),
    ((20,), False, "t", (1, 20, 1)),
])
def test_layout_round_trip(shape, multi, layout, x3_shape):
    x = torch.randn(*shape, dtype=torch.float64)
    x3, orig, lay = to_btd(x, multi)
    assert lay == layout and tuple(x3.shape) == x3_shape and tuple(orig) == shape
    k = 5
    phi = torch.randn(x3.shape[0], x3.shape[1], k, dtype=torch.float64)
    out = from_btk(phi, orig, lay)
    expected = {"btd": (3, 20, k), "bt": (3, 20, k), "nd": (15, k), "t": (20, k)}[layout]
    assert tuple(out.shape) == expected


def test_to_btd_rejects_other_dims():
    with pytest.raises(ValueError):
        to_btd(torch.zeros(2, 3, 4, 5), False)


def test_causal_history_padding_and_state():
    x3 = torch.arange(24, dtype=torch.float64).reshape(2, 6, 2)
    h0 = causal_history(x3, 3, None)
    assert h0.shape == (2, 9, 2)
    assert torch.equal(h0[:, :3], x3[:, :1].expand(2, 3, 2))
    assert torch.equal(h0[:, 3:], x3)
    assert causal_history(x3, 0, None) is x3
    state = {"hist": torch.full((2, 3, 2), -1.0, dtype=torch.float64)}
    h1 = causal_history(x3, 3, state)
    assert torch.equal(h1[:, :3], state["hist"])
    with pytest.raises(ValueError, match="continue_state needs the same groups"):
        causal_history(x3, 3, {"hist": torch.zeros(3, 3, 2, dtype=torch.float64)})


def test_tail_state_and_unfold():
    x_hist = torch.arange(2 * 10 * 1, dtype=torch.float64).reshape(2, 10, 1)
    s = tail_state(x_hist, 3)
    assert s.shape == (2, 3, 1) and torch.equal(s, x_hist[:, -3:])
    assert tail_state(x_hist, 0).shape == (2, 0, 1)
    w = causal_window_unfold(x_hist, 4)
    assert w.shape == (2, 7, 4, 1)
    assert torch.equal(w[:, 0, :, 0], x_hist[:, :4, 0])
    w_last = causal_window_unfold(x_hist, 4, t_len=5)
    assert w_last.shape == (2, 5, 4, 1) and torch.equal(w_last, w[:, -5:])
    with pytest.raises(ValueError, match="history"):
        causal_window_unfold(x_hist[:, :3], 4)
    with pytest.raises(ValueError, match="history"):
        causal_window_unfold(x_hist[:, :6], 4, t_len=5)


def _loop(u, a, b, y_prev):
    B, T, C = u.shape
    a = torch.as_tensor(a, dtype=u.dtype).expand(C)
    b = torch.as_tensor(b, dtype=u.dtype).expand(C)
    y = torch.zeros(B, C, dtype=u.dtype) if y_prev is None else y_prev.clone()
    out = []
    for t in range(T):
        y = a * y + b * u[:, t]
        out.append(y)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("T", [63, 64, 1000])
@pytest.mark.parametrize("a", [0.0, 0.5, 0.99, 1.0, -0.5])
@pytest.mark.parametrize("with_prev", [False, True])
def test_recurrence_matches_loop(T, a, with_prev):
    g = torch.Generator().manual_seed(0)
    u = torch.randn(2, T, 3, dtype=torch.float64, generator=g)
    y_prev = torch.randn(2, 3, dtype=torch.float64, generator=g) if with_prev else None
    got = causal_linear_recurrence(u, a, 0.3, y_prev)
    want = _loop(u, a, 0.3, y_prev)
    assert torch.allclose(got, want, atol=1e-10)
    got32 = causal_linear_recurrence(u.float(), a, 0.3, None if y_prev is None else y_prev.float())
    assert got32.dtype == torch.float32
    assert torch.allclose(got32.double(), want, atol=1e-5)


def test_recurrence_per_channel_decay_and_bounds():
    u = torch.randn(1, 200, 4, dtype=torch.float64)
    a = torch.tensor([0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    assert torch.allclose(causal_linear_recurrence(u, a, 1.0 - a.clamp(max=0.99), None),
                          _loop(u, a, 1.0 - a.clamp(max=0.99), None), atol=1e-10)
    with pytest.raises(ValueError):
        causal_linear_recurrence(u, 1.5, 1.0, None)


@pytest.mark.parametrize("T", [300, 5000])
def test_recurrence_fft_path_with_negative_and_alternating_decays(T):
    """Per-channel decays of both signs (negative, alternating, close to -1) give the sequential result on the FFT path."""
    g = torch.Generator().manual_seed(5)
    u = torch.randn(2, T, 4, dtype=torch.float64, generator=g)
    a = torch.tensor([-0.95, -0.3, 0.6, -1.0], dtype=torch.float64)
    y_prev = torch.randn(2, 4, dtype=torch.float64, generator=g)
    got = causal_linear_recurrence(u, a, 0.2, y_prev)
    want = _loop(u, a, 0.2, y_prev)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, want, atol=1e-9)
