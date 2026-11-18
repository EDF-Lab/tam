# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Bit-identical v1.5.1 parity test for UniversalPhysicsEffect integer derivative penalties."""

import pytest
import torch
from tam.model.spectrum._physics import UniversalPhysicsEffect

ORDER_SETS = [
    {"D0": 1.0},
    {"D1": 1.0},
    {"D2": 1.0},
    {"D1": 1.0, "D2": 1.0},
    {"D0": 0.5, "D1": 1.0, "D2": 2.0},
    {"D1": 1.0, "D3": 0.7},
    {"D2": 1.0, "D4": 0.3},
]

def _v151_penalty(eff):
    """The penalty of v1.5.1, computed on the host that runs the test (the integer-order code of that version, unchanged), so that the comparison does not depend on
    the rounding of the BLAS or of the math library of the platform where a table of numbers was captured."""
    lam = eff.lambda_p
    weights = {int(k[1:]): w for k, w in eff.diff_weights.items()}
    if eff.basis_type == "spline":
        n = eff.n_coeffs_val
        eye = torch.eye(n, dtype=torch.get_default_dtype())
        valid_rows = n - max(weights)
        L_op = torch.zeros((valid_rows, n), dtype=torch.get_default_dtype())
        for order, weight in weights.items():
            mat = eye if order == 0 else torch.diff(eye, n=order, dim=0)
            if mat.shape[0] > valid_rows:
                top = (mat.shape[0] - valid_rows) // 2
                mat = mat[top: top + valid_rows, :]
            L_op += weight * mat
        return lam * (L_op.T @ L_op)
    if eff.basis_type == "fourier":
        freqs = torch.arange(1, eff.base_effect.m + 1, dtype=torch.get_default_dtype())
        eig = torch.zeros_like(freqs, dtype=torch.complex128)
        for order, weight in weights.items():
            eig += weight * (1j * freqs) ** order
        half = lam * (torch.abs(eig) ** 2)
        return torch.diag(torch.cat([half, half]))
    w_flat = eff.base_effect.weights.flatten()
    diag = torch.zeros(eff.n_coeffs_val, dtype=torch.get_default_dtype())
    for order, weight in weights.items():
        diag += weight * (w_flat ** order)
    return lam * torch.diag(torch.abs(diag) ** 2)


@pytest.mark.parametrize("basis,n", [("spline", 6), ("fourier", 4), ("neural", 6)])
@pytest.mark.parametrize("idx", range(len(ORDER_SETS)))
def test_physics_integer_penalty_bit_identical_to_v151(basis, n, idx):
    torch.manual_seed(42)
    eff = UniversalPhysicsEffect("x", basis, n, ORDER_SETS[idx], 1.0, "continue")
    if basis == "neural":
        eff.initialize(torch.linspace(-1, 1, 20, dtype=torch.get_default_dtype()).unsqueeze(-1))
    actual = eff.build_penalty_matrix().detach().cpu()
    assert torch.equal(actual, _v151_penalty(eff).detach().cpu())
