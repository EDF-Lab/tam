r"""
Unit tests for ``tam.model.autotam.pipeline.base_discoverer.BaseDiscoverer``.

The heavy DragTAM.optimize() evolutionary loop is not exercised here
(per the NEWTODO.md guardrail against slow orchestration tests).
We focus on the pure formula-classification helper, which is the only
logic in this module that does not depend on a full search run.
"""

import pytest
from tam.model.autotam.pipeline.base_discoverer import BaseDiscoverer


def _d() -> BaseDiscoverer:
    return BaseDiscoverer(pop_size=4)


def test_linear_effects_map_to_linear_island():
    d = _d()
    assert d._get_island_from_formula("y ~ l(x) + c(cat)") == "LinearIsland"


def test_spline_maps_to_spline_island():
    assert _d()._get_island_from_formula("y ~ s(x, k=10)") == "SplineIsland"


def test_fourier_maps_to_fourier_island():
    assert _d()._get_island_from_formula("y ~ f(x, m=4)") == "FourierIsland"


def test_chebyshev_maps_to_chebyshev_island():
    assert _d()._get_island_from_formula("y ~ p(x, deg=8)") == "ChebyshevIsland"


def test_wavelet_maps_to_wavelet_island():
    assert _d()._get_island_from_formula("y ~ w(x)") == "WaveletIsland"


def test_neural_maps_to_neural_island():
    assert _d()._get_island_from_formula("y ~ n(x)") == "NeuralIsland"


def test_rbf_maps_to_rbf_island():
    assert _d()._get_island_from_formula("y ~ rbf(x)") == "RBFIsland"


def test_tree_maps_to_tree_island():
    assert _d()._get_island_from_formula("y ~ t(x, n_trees=10)") == "TreeIsland"


def test_tensor_product_with_few_terms_maps_to_cross_island():
    assert _d()._get_island_from_formula("y ~ te(s(a), s(b))") == "CrossIsland"


def test_multiple_non_linear_effects_maps_to_continent():
    # Three distinct core effects -> Continent
    assert _d()._get_island_from_formula("y ~ s(x) + f(z) + n(q) + w(r) + rbf(p)") == "Continent"


def test_missing_tilde_maps_to_continent():
    assert _d()._get_island_from_formula("no tilde here") == "Continent"
