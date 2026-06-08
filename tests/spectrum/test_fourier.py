r"""Tests for ``tam.model.spectrum._fourier.FourierEffect``."""

import pytest
import torch

import tam
from tam.model.spectrum import FourierEffect


@pytest.mark.parametrize("cyclic", [False, True])
def test_fourier_contract(cyclic, normalized, penalty_shape):
    effect = FourierEffect("x", m=4, s=2, lambda_p=1.0, cyclic=cyclic, extrapolate="continue")
    assert effect.get_n_coeffs() == 8  # 2 * m

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 8)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (8, 8)


def test_fourier_penalty_is_diagonal_sobolev():
    effect = FourierEffect("x", m=3, s=1, lambda_p=2.0, cyclic=False, extrapolate="continue")
    P = effect.build_penalty_matrix()
    # Diagonal Sobolev penalty: off-diagonal entries are zero.
    off_diag = P - torch.diag(torch.diagonal(P))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag))


def test_fourier_rejects_non_positive_m():
    with pytest.raises(ValueError):
        FourierEffect("x", m=0, s=1, lambda_p=1.0, cyclic=False, extrapolate="continue")


def test_fourier_rejects_negative_s():
    with pytest.raises(ValueError):
        FourierEffect("x", m=2, s=-1, lambda_p=1.0, cyclic=False, extrapolate="continue")
