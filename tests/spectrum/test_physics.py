r"""Tests for ``tam.model.spectrum._physics.UniversalPhysicsEffect`` (PDE-constrained)."""

import pytest
import torch

import tam
from tam.model.spectrum import UniversalPhysicsEffect


@pytest.mark.parametrize("basis,n_coeffs_arg,expected_k", [
    ("spline", 10, 13),   # n_knots=10, default degree 3 -> 13 coefficients
    ("fourier", 10, 10),  # m = 10 // 2 = 5 -> 2m = 10
    ("neural", 12, 12),   # n_neurons = n_coeffs
])
def test_physics_feature_map(basis, n_coeffs_arg, expected_k, normalized):
    effect = UniversalPhysicsEffect(
        "x", basis_type=basis, n_coeffs=n_coeffs_arg,
        diff_weights={"d2": 1.0}, lambda_p=1.0, extrapolate="continue",
    )
    assert effect.get_n_coeffs() == expected_k

    x = normalized(4, 12) if basis != "neural" else normalized(4, 12, 1)
    phi = effect.build_feature_map(x)
    assert phi.shape[-1] == expected_k
    assert torch.isfinite(phi).all()


@pytest.mark.parametrize("basis", ["spline", "fourier"])
def test_physics_penalty_shape(basis, penalty_shape):
    effect = UniversalPhysicsEffect(
        "x", basis_type=basis, n_coeffs=10,
        diff_weights={"d1": 1.0, "d2": 0.5}, lambda_p=1.0, extrapolate="continue",
    )
    k = effect.get_n_coeffs()
    assert penalty_shape(effect) == (k, k)


def test_physics_spline_penalty_with_zeroth_order_term(penalty_shape):
    """A d0 (identity) operator term exercises the order==0 branch of the stiffness build."""
    effect = UniversalPhysicsEffect(
        "x", basis_type="spline", n_coeffs=10,
        diff_weights={"d0": 1.0, "d2": 0.5}, lambda_p=1.0, extrapolate="continue",
    )
    k = effect.get_n_coeffs()
    assert penalty_shape(effect) == (k, k)


def test_physics_neural_penalty_after_feature_map(normalized, penalty_shape):
    """The neural stiffness penalty needs the frozen weights, which init lazily."""
    effect = UniversalPhysicsEffect(
        "x", basis_type="neural", n_coeffs=8,
        diff_weights={"d1": 1.0}, lambda_p=1.0, extrapolate="continue",
    )
    effect.build_feature_map(normalized(4, 12, 1))
    assert penalty_shape(effect) == (8, 8)


def test_physics_invalid_basis_raises():
    with pytest.raises(ValueError, match="Unknown basis type"):
        UniversalPhysicsEffect(
            "x", basis_type="bogus", n_coeffs=5,
            diff_weights={"d1": 1.0}, lambda_p=1.0, extrapolate="continue",
        )
