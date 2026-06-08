r"""
Tests for ``tam.model.spectrum._base_effects.BaseEffect``, the abstract base
and its universal out-of-distribution (OOD) extrapolation wrapper.

The wrapper has two distinct code paths: a univariate path (scalar features)
and a multivariate path (effects that consume several feature columns, e.g.
RBF, Neural, Tensor-Product, Tree). Both are exercised here for every mode.
"""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import BaseEffect, SplineEffect, RBFEffect, TreeEffect

MODES = ["continue", "constant", "linear", "saturation"]


# --------------------------------------------------------------------------- #
# Univariate extrapolation (scalar feature effects such as splines)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", MODES)
def test_univariate_extrapolation_is_finite(mode, normalized, out_of_distribution):
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate=mode)
    effect.build_feature_map(normalized(4, 12))  # seed cached knots in-distribution

    phi = effect.transform(out_of_distribution(4, 12, value=5.0))
    assert phi.shape[:2] == (4, 12)
    assert torch.isfinite(phi).all()


def test_univariate_no_ood_uses_plain_feature_map(normalized):
    """In-distribution data must bypass the extrapolation maths entirely."""
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="linear")
    x = normalized(3, 10)
    direct = effect.build_feature_map(x)
    via_transform = effect.transform(x)
    assert torch.allclose(direct, via_transform)


def test_univariate_constant_clamps_to_boundary(normalized, out_of_distribution):
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="constant")
    effect.build_feature_map(normalized(4, 12))

    at_boundary = torch.ones(4, 12, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    beyond = out_of_distribution(4, 12, value=9.0)
    assert torch.allclose(effect.transform(at_boundary), effect.transform(beyond))


# --------------------------------------------------------------------------- #
# Multivariate extrapolation (effects consuming several feature columns)
# --------------------------------------------------------------------------- #

def _partial_ood_multivariate(n_rows=10, n_features=2):
    """Half in-distribution rows, half pushed outside the hypercube."""
    g = torch.Generator(device="cpu").manual_seed(7)
    base = (torch.rand(n_rows, n_features, generator=g, dtype=torch.get_default_dtype()) * 1.6 - 0.8)
    base[n_rows // 2:] += 4.0  # push the second half out of [-1, 1]
    return base.to(TORCH_DEVICE)


@pytest.mark.parametrize("mode", ["constant", "linear", "saturation"])
def test_multivariate_extrapolation_is_finite(mode):
    effect = RBFEffect(
        "lat", n_centers=5, gamma=0.5, nu=None, lambda_p=1.0,
        additional_features=["lon"], extrapolate=mode,
    )
    x = _partial_ood_multivariate(n_rows=10, n_features=2)
    phi = effect.transform(x)
    assert phi.shape == (10, 5)
    assert torch.isfinite(phi).all()


def test_tree_multivariate_detection_path():
    """An oblivious binary tree is treated as multivariate by the OOD wrapper."""
    effect = TreeEffect(
        "lat", n_trees=2, max_depth=2, max_leaves=None, lambda_p=1.0,
        additional_features=["lon"], seed=1, extrapolate="constant",
    )
    x = _partial_ood_multivariate(n_rows=8, n_features=2)
    phi = effect.transform(x)
    assert phi.shape == (8, effect.get_n_coeffs())
    assert torch.isfinite(phi).all()


# --------------------------------------------------------------------------- #
# Error handling and the abstract contract
# --------------------------------------------------------------------------- #

def test_invalid_extrapolation_mode_raises(normalized, out_of_distribution):
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    effect.build_feature_map(normalized(2, 6))
    effect.extrapolate = "teleport"  # bypass the constructor's normalization
    with pytest.raises(ValueError, match="Unknown extrapolation mode"):
        effect.transform(out_of_distribution(2, 6, value=3.0))


def test_abstract_methods_raise_not_implemented():
    """The base contract methods raise when invoked via ``super()``."""

    class _PassthroughEffect(BaseEffect):
        def get_n_coeffs(self):
            return super().get_n_coeffs()

        def build_feature_map(self, x_col):
            return super().build_feature_map(x_col)

        def build_penalty_matrix(self):
            return super().build_penalty_matrix()

    effect = _PassthroughEffect("x", "passthrough", lambda_p=1.0, extrapolate="continue")
    with pytest.raises(NotImplementedError):
        effect.get_n_coeffs()
    with pytest.raises(NotImplementedError):
        effect.build_feature_map(torch.zeros(1))
    with pytest.raises(NotImplementedError):
        effect.build_penalty_matrix()


def test_extrapolate_string_is_normalized():
    """The constructor strips quotes/whitespace and lowercases the mode."""
    effect = SplineEffect("x", n_knots=5, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="'LINEAR' ")
    assert effect.extrapolate == "linear"
