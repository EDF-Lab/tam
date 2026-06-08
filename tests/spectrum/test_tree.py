r"""Tests for ``tam.model.spectrum._tree.TreeEffect`` (Random Binning Features)."""

import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import TreeEffect


def test_oblivious_binary_contract(normalized, penalty_shape):
    effect = TreeEffect("x", n_trees=3, max_depth=2, max_leaves=None, lambda_p=1.0,
                        additional_features=None, seed=42, extrapolate="continue")
    assert effect.get_n_coeffs() == 12  # 3 trees * 2^2 leaves
    assert effect.is_oblivious_binary is True

    phi = effect.build_feature_map(normalized(4, 12, 1))
    assert phi.shape == (4, 12, 12)
    assert penalty_shape(effect) == (12, 12)


def test_nary_histogram_single_tree(normalized, penalty_shape):
    effect = TreeEffect("x", n_trees=1, max_depth=3, max_leaves=5, lambda_p=1.0,
                        additional_features=None, seed=42, extrapolate="continue")
    assert effect.get_n_coeffs() == 5  # 1 tree * 5 leaves
    assert effect.is_oblivious_binary is False

    phi = effect.build_feature_map(normalized(4, 12, 1))
    assert phi.shape == (4, 12, 5)
    assert penalty_shape(effect) == (5, 5)


def test_nary_histogram_ensemble_multiple_trees(normalized):
    """n_trees>1 with max_leaves exercises the random-threshold ensemble path."""
    effect = TreeEffect("x", n_trees=4, max_depth=3, max_leaves=3, lambda_p=1.0,
                        additional_features=None, seed=42, extrapolate="continue")
    assert effect.get_n_coeffs() == 12  # 4 trees * 3 leaves
    phi = effect.build_feature_map(normalized(2, 20, 1))
    assert phi.shape == (2, 20, 12)
    assert torch.isfinite(phi).all()


def test_tree_constant_input_does_not_crash():
    """Degenerate constant data must not produce zero-width split bounds."""
    effect = TreeEffect("x", n_trees=2, max_depth=2, max_leaves=None, lambda_p=1.0,
                        additional_features=None, seed=1, extrapolate="continue")
    x = torch.full((2, 10, 1), 0.3, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    phi = effect.build_feature_map(x)
    assert phi.shape == (2, 10, 8)
    assert torch.isfinite(phi).all()


def test_tree_one_dimensional_input_routing():
    effect = TreeEffect("x", n_trees=2, max_depth=2, max_leaves=None, lambda_p=1.0,
                        additional_features=None, seed=2, extrapolate="continue")
    x = torch.rand(7, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * 2 - 1
    phi = effect.build_feature_map(x)
    assert phi.shape == (7, 8)


def test_tree_two_dimensional_tensor_product_routing(normalized):
    """A 2D input whose last axis is time (te regular pass) is reshaped."""
    effect = TreeEffect("x", n_trees=2, max_depth=2, max_leaves=None, lambda_p=1.0,
                        additional_features=None, seed=4, extrapolate="continue")
    phi = effect.build_feature_map(normalized(4, 12))  # last axis (12) != feature count (1)
    assert phi.shape == (4, 12, 8)


def test_tree_memory_probe_returns_zeros():
    """The (1, 1) VRAM probe returns a zero block without fitting the forest."""
    effect = TreeEffect("x", n_trees=2, max_depth=2, max_leaves=None, lambda_p=1.0,
                        additional_features=None, seed=3, extrapolate="continue")
    probe = torch.zeros(1, 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    phi = effect.build_feature_map(probe)
    assert phi.shape[-1] == effect.get_n_coeffs()
    assert torch.allclose(phi, torch.zeros_like(phi))
    assert effect.split_features is None  # forest geometry left uninitialised
