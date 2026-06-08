r"""Tests for ``tam.model.spectrum._rbf.RBFEffect`` (Gaussian / Matern kernels)."""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import RBFEffect


def _effect(n_centers=5, gamma=1.0, nu=None, additional=None):
    return RBFEffect(
        "x", n_centers=n_centers, gamma=gamma, nu=nu, lambda_p=1.0,
        additional_features=additional, extrapolate="continue",
    )


@pytest.mark.parametrize("nu", [None, 0.5, 1.5, 2.5])
def test_rbf_contract(nu, normalized, penalty_shape):
    effect = _effect(n_centers=5, nu=nu)
    assert effect.get_n_coeffs() == 5

    phi = effect.build_feature_map(normalized(4, 12, 1))
    assert phi.shape == (4, 12, 5)
    assert torch.isfinite(phi).all()
    assert penalty_shape(effect) == (5, 5)


def test_rbf_median_heuristic_sets_gamma(normalized):
    effect = _effect(n_centers=4, gamma=None)
    effect.build_feature_map(normalized(4, 12, 1))
    assert effect.gamma is not None and effect.gamma > 0


def test_rbf_multivariate_inputs(normalized):
    effect = _effect(n_centers=6, gamma=0.5, additional=["lon"])
    assert effect.input_features == ["x", "lon"]
    phi = effect.build_feature_map(normalized(4, 12, 2))
    assert phi.shape == (4, 12, 6)


def test_rbf_matern_arbitrary_nu_uses_scipy_fallback(normalized):
    """Non-half-integer nu routes through the SciPy Bessel implementation."""
    pytest.importorskip("scipy")
    effect = _effect(n_centers=4, gamma=1.0, nu=1.2)
    with pytest.warns(UserWarning, match="CPU synchronization"):
        phi = effect.build_feature_map(normalized(2, 8, 1))
    assert phi.shape == (2, 8, 4)
    assert torch.isfinite(phi).all()


def test_rbf_fewer_samples_than_centers():
    effect = _effect(n_centers=10, gamma=1.0)
    x = torch.rand(1, 3, 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * 2 - 1
    phi = effect.build_feature_map(x)
    assert phi.shape == (1, 3, 10)
    assert effect.centers.shape[0] == 10


def test_rbf_one_dimensional_input_routing():
    """A bare 1D input (OOD wrapper path) is reshaped to a column vector."""
    effect = _effect(n_centers=5, gamma=1.0)
    x = torch.rand(6, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * 2 - 1
    phi = effect.build_feature_map(x)
    assert phi.shape == (6, 5)


def test_rbf_two_dimensional_tensor_product_routing(normalized):
    """A 2D input whose last axis is time (te path) is reshaped to a column vector."""
    effect = _effect(n_centers=5, gamma=1.0)  # univariate: input_features = ['x']
    phi = effect.build_feature_map(normalized(4, 12))  # last axis (12) != feature count (1)
    assert phi.shape == (4, 12, 5)


def test_rbf_memory_probe_shape():
    effect = _effect(n_centers=4, gamma=1.0)
    probe = torch.zeros(1, 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    phi = effect.build_feature_map(probe)
    assert phi.shape[-1] == 4
