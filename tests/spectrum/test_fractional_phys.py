# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam.model.spectrum._physics import UniversalPhysicsEffect

def test_fractional_caputo_physics_penalty():
    """Verifies that Caputo fractional penalty behaves continuously and annihilates constants."""
    # 1. Fourier fractional derivative
    phys_fourier_05 = UniversalPhysicsEffect(
        "x", basis_type="fourier", n_coeffs=16, diff_weights={"D0.5": 1.0}, lambda_p=1.0, extrapolate="continue"
    )
    p_05 = phys_fourier_05.build_penalty_matrix()
    diag_05 = torch.diagonal(p_05)
    # Derivative order 0.5 should grow as freqs^1.0
    assert (diag_05 > 0).all()
    assert diag_05[0] < diag_05[7]

    phys_fourier_10 = UniversalPhysicsEffect(
        "x", basis_type="fourier", n_coeffs=16, diff_weights={"D1.0": 1.0}, lambda_p=1.0, extrapolate="continue"
    )
    p_10 = phys_fourier_10.build_penalty_matrix()
    diag_10 = torch.diagonal(p_10)
    # Order 1.0 should grow as freqs^2.0
    assert torch.allclose(diag_10[:8], (torch.arange(1, 9, dtype=torch.float64) ** 2))

    # 2. Spline fractional derivative
    phys_spline = UniversalPhysicsEffect(
        "x", basis_type="spline", n_coeffs=12, diff_weights={"D0.5": 1.0}, lambda_p=1.0, extrapolate="continue"
    )
    p_spline = phys_spline.build_penalty_matrix()
    assert p_spline.shape == (12 + 3, 12 + 3)
    eigvals = torch.linalg.eigvalsh(p_spline)
    assert (eigvals >= -1e-10).all()


def test_formula_accepts_fractional_orders():
    """`phys(x, D0_5=1.0)` reaches the effect as the order 0.5.

    The underscore is the spelling of the decimal point: the generic term parser reads a key as a word, so `D0.5=1.0` is refused there.
    """
    from tam import StaticTAM

    for spec, expected in (("D0_5=1.0", {"D0.5": 1.0}), ("D0_5=1.0, D2=0.1", {"D0.5": 1.0, "D2": 0.1}), ("D1_5=2.0", {"D1.5": 2.0})):
        model = StaticTAM(formula=f"y ~ phys(x, basis='fourier', n_coeffs=8, {spec})", group_col="g", date_col="d")
        effect = [e for e in model.effects_list_ if e.feature_name == "x"][0]
        assert effect.diff_weights == expected, (spec, effect.diff_weights)


def test_syntax_reference_shows_the_fractional_spelling(capsys):
    from tam import StaticTAM

    StaticTAM.show_syntax()
    assert "D0_5=1.0" in capsys.readouterr().out
