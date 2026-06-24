r"""
Unit tests for the effect factory in ``tam.model.spectrum._factory``.

Covers both halves of the factory:
  * ``create_effects_from_parsed_terms``, instantiating the correct effect
    class per formula token, grid-search token substitution, categorical
    cardinality inference, and error handling.
  * ``build_phi_from_effects`` / ``_infer_feature_columns``, the column
    routing that assembles the global design matrix.
"""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE, parse_formula_to_terms
from tam.model.spectrum import (
    create_effects_from_parsed_terms, build_phi_from_effects,
    OffsetEffect, LinearEffect, FourierEffect, SplineEffect,
    CategoricalEffect, ChebyshevEffect, WaveletEffect, NeuralEffect,
    RBFEffect, UniversalPhysicsEffect, TensorProductEffect, TreeEffect,
    LinearTreeEffect, PIDEffect,
)


def _x(*shape):
    generator = torch.Generator(device="cpu").manual_seed(0)
    return (torch.rand(*shape, generator=generator, dtype=torch.get_default_dtype()) * 2 - 1).to(TORCH_DEVICE)

DEFAULT_AP = -9.0


def _build(formula, token_values=None, data_info=None, include_offset=True):
    _, terms = parse_formula_to_terms(formula)
    return create_effects_from_parsed_terms(
        terms, token_values or {}, DEFAULT_AP,
        include_offset=include_offset, data_info=data_info,
    )


def test_offset_prepended_by_default():
    effects = _build("y ~ l(x)")
    assert isinstance(effects[0], OffsetEffect)
    assert isinstance(effects[1], LinearEffect)


def test_include_offset_false_skips_intercept():
    effects = _build("y ~ l(x)", include_offset=False)
    assert len(effects) == 1
    assert isinstance(effects[0], LinearEffect)


@pytest.mark.parametrize("formula,cls", [
    ("y ~ l(x)", LinearEffect),
    ("y ~ f(x, m=4)", FourierEffect),
    ("y ~ s(x, k=8)", SplineEffect),
    ("y ~ p(x, deg=5)", ChebyshevEffect),
    ("y ~ w(x, n_scales=3, n_locations=4)", WaveletEffect),
    ("y ~ n(x, n_neurons=8)", NeuralEffect),
    ("y ~ rbf(x, n_centers=5)", RBFEffect),
    ("y ~ t(x, n_trees=3, max_depth=2)", TreeEffect),
    ("y ~ pid(x, w=5)", PIDEffect),
    ("y ~ phys(x, basis='spline', D2=1.0)", UniversalPhysicsEffect),
    ("y ~ lt(x, slope=z)", LinearTreeEffect),
])
def test_each_token_builds_expected_effect(formula, cls):
    effects = _build(formula, include_offset=False)
    assert isinstance(effects[0], cls), f"{formula} did not build a {cls.__name__}."


def test_tensor_product_token_builds_tensor_effect():
    effects = _build("y ~ te(s(lat), s(lon))", include_offset=False)
    assert isinstance(effects[0], TensorProductEffect)
    assert len(effects[0].effects) == 2


def test_spline_param_parsing():
    effect = _build("y ~ s(x, k=12, deg=2, p=1)", include_offset=False)[0]
    assert effect.n_knots == 12
    assert effect.spline_degree == 2
    assert effect.penalty_order == 1


def test_fourier_cyclic_flag_parsing():
    effect = _build("y ~ f(x, m=3, cyclic=True)", include_offset=False)[0]
    assert effect.cyclic is True
    assert effect.m == 3


def test_grid_search_token_substitution():
    """A string token in the formula is resolved from token_values at build time."""
    effects = _build("y ~ s(x, k='grid_k')", token_values={"grid_k": 7}, include_offset=False)
    assert effects[0].n_knots == 7


def test_ap_token_sets_regularization():
    # ap = log10(lambda_p); ap=-2 -> lambda_p = 0.01
    effect = _build("y ~ l(x, ap=-2)", include_offset=False)[0]
    assert abs(effect.lambda_p - 1e-2) < 1e-9


def test_ap_offset_token_controls_intercept_penalty():
    effects = _build("y ~ l(x)", token_values={"ap_offset": -3.0})
    assert abs(effects[0].lambda_p - 1e-3) < 1e-9


def test_categorical_infers_n_cat_from_data_info():
    effect = _build("y ~ c(region)", data_info={"region": 4}, include_offset=False)[0]
    assert isinstance(effect, CategoricalEffect)
    assert effect.n_categories == 4


def test_categorical_explicit_n_cat():
    effect = _build("y ~ c(region, n_cat=6, topo='ordinal')", include_offset=False)[0]
    assert effect.n_categories == 6
    assert effect.topology == "ordinal"


def test_categorical_without_context_raises():
    with pytest.raises(ValueError, match="requires 'n_cat'"):
        _build("y ~ c(region)", include_offset=False)


def test_multivariate_others_parsing():
    effect = _build("y ~ rbf(lat, n_centers=5, others='lon')", include_offset=False)[0]
    assert effect.input_features == ["lat", "lon"]


def test_physics_diff_weights_parsing():
    effect = _build("y ~ phys(x, basis='spline', D1=2.0, D2=0.5)", include_offset=False)[0]
    assert effect.diff_weights == {"D1": 2.0, "D2": 0.5}


def test_invalid_ap_raises():
    with pytest.raises(ValueError, match="Invalid value for 'ap'"):
        _build("y ~ l(x, ap='not_a_number')", include_offset=False)


def test_tensor_product_with_single_subterm_raises():
    """A tensor product needs at least two functional sub-terms."""
    with pytest.raises(ValueError, match="Tensor Product"):
        _build("y ~ te(s(x))", include_offset=False)


def test_unknown_token_raises():
    from tam.model.spectrum import create_effects_from_parsed_terms
    bogus = [{"feature": "x", "type": "zzz", "params": {}}]
    with pytest.raises(ValueError, match="Unknown effect type"):
        create_effects_from_parsed_terms(bogus, {}, -9.0, include_offset=False)


def test_linear_tree_forces_single_tree():
    effect = _build("y ~ lt(x, slope=z, max_depth=3)", include_offset=False)[0]
    assert effect.base_tree.n_trees == 1

def test_tree_default_params():
    """Ensures TreeEffect defaults to isotropic uniform splits if omitted."""
    effect = _build("y ~ t(x, max_depth=3)", include_offset=False)[0]
    assert isinstance(effect, TreeEffect)
    assert effect.sparsity_alpha == 0.0
    assert effect.split_strategy == "uniform"
    assert effect.is_oblivious_binary is True

def test_tree_advanced_params():
    """Verifies that the factory correctly parses the new density and topology configurations."""
    effect = _build("y ~ t(x, sp_alpha=0.5, split_strategy='quantile', max_leaves=10)", include_offset=False)[0]
    
    assert effect.sparsity_alpha == 0.5
    assert effect.split_strategy == "quantile"
    
    # max_leaves should override max_depth and trigger the Flat N-ary architecture
    assert effect.leaves_per_tree == 10
    assert effect.is_oblivious_binary is False

def test_linear_tree_advanced_params_inheritance():
    """
    Verifies that sp_alpha and split_strategy defined in a Linear Tree formula
    are correctly routed down to both the base_tree and the slope_tree.
    """
    effect = _build("y ~ lt(x, slope=z, sp_alpha=0.8, split_strategy='quantile')", include_offset=False)[0]
    
    assert isinstance(effect, LinearTreeEffect)
    
    # Check that the base tree inherited the parameters
    assert effect.base_tree.sparsity_alpha == 0.8
    assert effect.base_tree.split_strategy == "quantile"
    
    # Check that the slope tree also inherited the parameters
    assert effect.slope_tree.sparsity_alpha == 0.8
    assert effect.slope_tree.split_strategy == "quantile"

# --------------------------------------------------------------------------- #
# build_phi_from_effects / _infer_feature_columns column routing
# --------------------------------------------------------------------------- #

def test_build_phi_name_based_multivariate_routing():
    """A multivariate effect selects its columns by name from feature_columns."""
    effect = RBFEffect("lat", n_centers=6, gamma=0.5, nu=None, lambda_p=1.0,
                       additional_features=["lon"], extrapolate="continue")
    phi = build_phi_from_effects(_x(2, 10, 2), [effect], feature_columns=["lat", "lon"])
    assert phi.shape == (2, 10, 6)


def test_build_phi_missing_simple_feature_raises():
    effect = LinearEffect("x", scaled=1.0, lambda_p=1.0, extrapolate="continue")
    with pytest.raises(ValueError, match="not found"):
        build_phi_from_effects(_x(2, 10, 1), [effect], feature_columns=["y"])


def test_build_phi_missing_multivariate_subfeature_raises():
    effect = RBFEffect("lat", n_centers=4, gamma=0.5, nu=None, lambda_p=1.0,
                       additional_features=["lon"], extrapolate="continue")
    with pytest.raises(ValueError, match="sub-feature not found"):
        build_phi_from_effects(_x(2, 10, 2), [effect], feature_columns=["lat", "altitude"])


def test_build_phi_infers_columns_when_none_given():
    effects = [
        OffsetEffect(lambda_p=1e-3, extrapolate="continue"),
        LinearEffect("x", scaled=1.0, lambda_p=1.0, extrapolate="continue"),
    ]
    phi = build_phi_from_effects(_x(2, 10, 1), effects)  # feature_columns omitted
    assert phi.shape == (2, 10, sum(e.get_n_coeffs() for e in effects))


def test_build_phi_infers_columns_through_tensor_product():
    sub_a = SplineEffect("a", n_knots=5, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    sub_b = SplineEffect("b", n_knots=4, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    te = TensorProductEffect([sub_a, sub_b], lambda_p=1.0, extrapolate="continue")
    phi = build_phi_from_effects(_x(2, 8, 2), [te])  # columns inferred as ['a', 'b']
    assert phi.shape == (2, 8, sub_a.get_n_coeffs() * sub_b.get_n_coeffs())
