r"""Tests for ``tam.model.spectrum._linear_tree.LinearTreeEffect`` (varying-coefficient)."""

import torch

import tam
from tam.model.spectrum import LinearTreeEffect


def _effect(n_trees=2, max_depth=2):
    return LinearTreeEffect(
        "x", slope_feature="z", n_trees=n_trees, max_depth=max_depth, max_leaves=None,
        lambda_p=1.0, additional_features=None, seed=42, extrapolate="continue",
    )


def test_linear_tree_contract(normalized, penalty_shape):
    effect = _effect(n_trees=2, max_depth=2)
    # base_tree leaves (2*4=8) + tensor (slope_tree 8 x linear 1 = 8) = 16
    assert effect.get_n_coeffs() == 16
    assert effect.input_features == ["x", "z"]

    phi = effect.build_feature_map(normalized(4, 12, 2))
    assert phi.shape == (4, 12, 16)
    assert penalty_shape(effect) == (16, 16)


def test_linear_tree_feature_map_concatenates_base_and_interaction(normalized):
    effect = _effect()
    phi = effect.build_feature_map(normalized(2, 10, 2))
    base_k = effect.base_tree.get_n_coeffs()
    tensor_k = effect.tensor.get_n_coeffs()
    assert phi.shape[-1] == base_k + tensor_k
    assert torch.isfinite(phi).all()
