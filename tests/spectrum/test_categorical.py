r"""Tests for ``tam.model.spectrum._categorical.CategoricalEffect``."""

import pytest
import torch

import tam
from tam.model.spectrum import CategoricalEffect


def test_nominal_contract(normalized, penalty_shape):
    effect = CategoricalEffect("x", n_categories=5, topology="nominal", lambda_p=1.0, penalty_order=1, extrapolate="continue")
    assert effect.get_n_coeffs() == 5

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 5)
    # One-hot rows sum to exactly one.
    assert torch.allclose(phi.sum(dim=-1), torch.ones(4, 12, dtype=phi.dtype, device=phi.device))

    assert penalty_shape(effect) == (5, 5)


def test_ordinal_contract(normalized, penalty_shape):
    effect = CategoricalEffect("x", n_categories=5, topology="ordinal", lambda_p=1.0, penalty_order=2, extrapolate="continue")
    assert effect.get_n_coeffs() == 5
    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 5)
    assert penalty_shape(effect) == (5, 5)


def test_fourier_topology_contract(normalized, penalty_shape):
    effect = CategoricalEffect("x", n_categories=6, topology="fourier", lambda_p=1.0, penalty_order=1, extrapolate="continue")
    # m = n // 2 + n % 2 = 3 ; get_n_coeffs = 2m = 6
    assert effect.get_n_coeffs() == 6
    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 6)
    assert penalty_shape(effect) == (6, 6)


def test_topology_is_case_insensitive():
    effect = CategoricalEffect("x", n_categories=3, topology="NOMINAL", lambda_p=1.0, penalty_order=1, extrapolate="continue")
    assert effect.topology == "nominal"


def test_invalid_topology_raises():
    with pytest.raises(ValueError, match="invalid"):
        CategoricalEffect("x", n_categories=4, topology="diagonal", lambda_p=1.0, penalty_order=1, extrapolate="continue")
