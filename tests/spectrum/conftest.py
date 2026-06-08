r"""
Shared fixtures for the per-effect spectrum test suite.

Every effect script in ``tam.model.spectrum`` has a dedicated test module in
this package. These fixtures provide normalized inputs (the [-1, 1] domain all
effects expect) and a helper to read a penalty matrix's dense shape.
"""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE


@pytest.fixture
def normalized():
    """Returns a factory producing deterministic tensors in [-1, 1]."""
    def _make(*shape, seed=123):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.rand(*shape, generator=generator, dtype=torch.get_default_dtype())
        return (raw * 2.0 - 1.0).to(TORCH_DEVICE)
    return _make


@pytest.fixture
def penalty_shape():
    """Returns a helper that reads an effect's penalty matrix shape (densifying if sparse)."""
    def _shape(effect):
        matrix = effect.build_penalty_matrix()
        if matrix.is_sparse:
            matrix = matrix.to_dense()
        return tuple(matrix.shape)
    return _shape


@pytest.fixture
def out_of_distribution():
    """Returns a factory producing constant tensors outside the [-1, 1] domain."""
    def _make(*shape, value=5.0):
        return torch.full(shape, value, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    return _make
