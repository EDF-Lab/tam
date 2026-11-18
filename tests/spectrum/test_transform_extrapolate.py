# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the extrapolation router: periodic mode, state guard on the out-of-range probe, temporal effects."""

import pytest
import torch

from tam.model.spectrum._base_effects import BaseEffect


class _Identity(BaseEffect):
    """Feature map = the input itself (one column)."""

    def __init__(self, extrapolate="continue"):
        super().__init__("x", "fake", 1.0, extrapolate)

    def get_n_coeffs(self):
        return 1

    def build_feature_map(self, x_col, state_in=None):
        return x_col.unsqueeze(-1)

    def build_penalty_matrix(self):
        return torch.eye(1)


class _Stateful(_Identity):
    stateful = True

    def __init__(self, extrapolate="continue"):
        super().__init__(extrapolate)
        self._check_temporal_extrapolate()

    def build_feature_map(self, x_col, state_in=None):
        self._state_out = {"y": x_col[..., -1:].detach().clone()}
        return x_col.unsqueeze(-1)


def test_periodic_wraps_into_the_domain():
    e = _Identity("periodic")
    x = torch.tensor([[-1.5, -1.0, 0.0, 0.5, 1.0, 1.25, 3.0]], dtype=torch.float64)
    out = e.transform(x).squeeze(-1)
    wrapped = ((x + 1.0) % 2.0) - 1.0
    assert torch.allclose(out, torch.where(x.abs() > 1.0, wrapped, x))   # only the values outside [-1, 1] move
    inside = torch.tensor([[-0.9, 0.0, 0.9]], dtype=torch.float64)
    assert torch.equal(e.transform(inside).squeeze(-1), inside)


def test_unknown_mode_still_raises_for_every_effect():
    e = _Identity("bogus")
    with pytest.raises(ValueError, match="Unknown extrapolation mode"):
        e.transform(torch.tensor([[2.0]], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["linear", "saturation"])
def test_temporal_effect_refuses_isolated_point_extrapolation(mode):
    with pytest.raises(ValueError, match="temporal effect"):
        _Stateful(mode)
    _Stateful("continue")
    _Stateful("constant")
    _Stateful("periodic")


def test_ood_probe_leaves_state_unchanged():
    e = _Identity("linear")
    e.stateful = False
    e._state_out = {"y": torch.tensor([42.0])}

    def build(x_col, state_in=None):
        e._state_out = {"y": x_col[..., -1:].detach().clone()}
        return x_col.unsqueeze(-1)

    e.build_feature_map = build
    e.transform(torch.tensor([[0.0, 0.5, 2.0]], dtype=torch.float64))
    # the state is the one of the first probe on the clamped input, not of the finite-difference probe
    assert torch.equal(e._state_out["y"], torch.tensor([[1.0]], dtype=torch.float64))


def test_constant_extrapolation_with_state_clamps_the_input():
    seen = {}

    class _E(_Stateful):
        def build_feature_map(self, x_col, state_in=None):
            seen["x"] = x_col.clone()
            seen["state"] = state_in
            return x_col.unsqueeze(-1)

    e = _E("constant")
    x = torch.tensor([[0.0, 3.0, -4.0]], dtype=torch.float64)
    e.transform(x, state_in={"y": 1})
    assert torch.equal(seen["x"], torch.clamp(x, -1, 1)) and seen["state"] == {"y": 1}
    e2 = _E("continue")
    e2.transform(x, state_in={"y": 1})
    assert torch.equal(seen["x"], x)
