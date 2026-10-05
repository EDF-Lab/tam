# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

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
from tam.common.utils import TORCH_DEVICE, parse_formula_to_terms
from tam.model.spectrum import (
    BaseEffect, SplineEffect, RBFEffect, TreeEffect,
    create_effects_from_parsed_terms,
)

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


# --------------------------------------------------------------------------- #
# Penalty homogeneity in lambda_p (the invariant the GCV search rests on)
# --------------------------------------------------------------------------- #

PENALISED_TERMS = [
    "l(x)",
    "c(g, n_cat=3, topo='nominal', p_order=1)",
    "c(g, n_cat=3, topo='ordinal', p_order=1)",
    "s(x, k=10, deg=3, p=2)",
    "f(x, m=6, s=2)",
    "p(x, deg=5, s=2)",
    "w(x, n_scales=4, n_locations=10)",
    "n(x, n_neurons=8, n_hidden_layers=1, act='relu')",
    "rbf(x, n_centers=6)",
    "t(x, n_trees=2, max_depth=2, split_strategy='uniform', sp_alpha=0.0)",
    "lt(x, slope=z, max_depth=2, split_strategy='uniform')",
    "pid(x, w=3, d_pen=10.0)",
    "phys(x, k=10, basis='spline', D2=1.0)",
    "phys(x, n_coeffs=10, basis='fourier', D2=1.0)",
    "te(s(x, k=6, deg=3, p=2), c(g, n_cat=3, topo='nominal', p_order=1))",
]


@pytest.mark.parametrize("term", PENALISED_TERMS)
def test_penalty_is_homogeneous_in_lambda_p(term):
    """Every basis must satisfy P(c * lambda_p) = c * P(lambda_p).

    The GCV solver searches by assigning ``lambda_p = 10**alpha`` to an effect,
    rebuilding its block and reporting that weight back. A basis whose penalty is
    not linear in its own lambda_p would make the reported weight meaningless and
    break the parity between the search and a later `fit()`.
    """
    _, parsed = parse_formula_to_terms(f"y ~ {term}")
    effects = create_effects_from_parsed_terms(
        parsed, token_values={}, default_alpha_p=-3.0, include_offset=False
    )
    assert effects

    for effect in effects:
        effect.lambda_p = 1.0
        unit = effect.build_penalty_matrix()
        unit = (unit.to_dense() if unit.is_sparse else unit).clone()

        effect.lambda_p = 10.0
        scaled = effect.build_penalty_matrix()
        scaled = scaled.to_dense() if scaled.is_sparse else scaled

        assert torch.allclose(scaled, unit * 10.0, rtol=1e-9, atol=0.0), (
            f"{effect.effect_type} penalty is not linear in lambda_p"
        )
