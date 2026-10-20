# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``s(x, m_pen=m)``: the roughness penalty of a P-spline split in ``m`` locally weighted parts, ``sum_r lambda_r D' diag(W_r) D`` with
smooth non-negative weights ``W_r`` that add up to one, each part with its own smoothing parameter for ``auto_fit``.
``m_pen=1`` is the ordinary penalty.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import SplineEffect, TensorProductEffect


def _effect(m_pen=1, k=20, p=2, lambda_p=1e-3):
    return SplineEffect("x", k, 3, p, lambda_p, "linear", m_pen=m_pen)


def test_default_is_the_ordinary_penalty():
    plain = SplineEffect("x", 20, 3, 2, 1e-3, "linear")
    assert plain.n_penalty_coordinates == 1
    assert torch.equal(plain.build_penalty_matrix(), _effect(1).build_penalty_matrix())


@pytest.mark.parametrize("m_pen", [2, 3, 5, 8])
def test_equal_coordinates_give_the_ordinary_penalty(m_pen):
    effect = _effect(m_pen)
    assert effect.n_penalty_coordinates == m_pen
    reference = _effect(1).build_penalty_matrix()
    assert torch.allclose(effect.build_penalty_matrix(), reference, atol=1e-12, rtol=0)


def test_a_coordinate_penalises_its_own_part_of_the_curve():
    effect = _effect(4)
    n = effect.get_n_coeffs()
    effect.set_penalty_coordinates([1.0, 1e-6, 1e-6, 1e-6])
    penalty = effect.build_penalty_matrix()
    first, last = torch.zeros(n, dtype=penalty.dtype), torch.zeros(n, dtype=penalty.dtype)
    first[:3] = torch.tensor([1.0, -2.0, 1.0])  # a kink at the left end
    last[-3:] = torch.tensor([1.0, -2.0, 1.0])  # the same kink at the right end
    assert float(first @ penalty @ first) > 100.0 * float(last @ penalty @ last)
    eigenvalues = torch.linalg.eigvalsh(penalty)
    assert float(eigenvalues.min()) > -1e-10  # positive semi-definite


def test_the_coordinates_round_trip():
    effect = _effect(3)
    effect.set_penalty_coordinates([0.1, 0.2, 0.3])
    assert effect.penalty_coordinates() == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        effect.set_penalty_coordinates([0.1, 0.2])


def test_bad_values_are_refused():
    with pytest.raises(ValueError, match="m_pen"):
        _effect(0)
    with pytest.raises(ValueError, match="m_pen"):
        _effect(25, k=20)


def test_the_formula_reads_m_pen():
    model = ta.StaticTAM("y ~ s(x, k=12, m_pen=3, ap=-4)")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 300)})
    frame["y"] = np.sin(6 * frame["x"])
    model.fit(frame)
    spline = next(e for e in model.effects_list_ if isinstance(e, SplineEffect))
    assert spline.m_pen == 3 and spline.n_penalty_coordinates == 3


def test_a_tensor_product_exposes_the_coordinates_of_every_margin():
    effect = TensorProductEffect([_effect(3, k=6), SplineEffect("z", 6, 3, 2, 1e-3, "linear")], 1.0, "linear")
    assert effect.n_penalty_coordinates == 4
    effect.set_penalty_coordinates([1e-2, 1e-3, 1e-4, 1e-5])
    assert effect.effects[0].penalty_coordinates() == [1e-2, 1e-3, 1e-4]
    assert effect.effects[1].lambda_p == pytest.approx(1e-5)


def _spike(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    truth = np.sin(2 * np.pi * x) - 2.0 * np.exp(-(((x - 0.98) / 0.02) ** 2))
    return pd.DataFrame({"x": x, "y": truth + 0.1 * rng.normal(size=n)}), truth


def test_adaptive_penalty_resolves_a_localised_feature_the_uniform_penalty_smooths_out():
    # mean over several data sets: one sample can go either way on another platform
    test, truth = _spike(6000, 99)
    errors = {1: [], 5: []}
    for seed in (1, 2, 3):
        train, _ = _spike(3000, seed)
        for m_pen in errors:
            model = ta.StaticTAM(f"y ~ s(x, k=60, m_pen={m_pen}, ap=-3)").auto_fit(train)
            errors[m_pen].append(float(np.sqrt(np.mean((model.predict(test)["Estimatedy"].to_numpy() - truth) ** 2))))
    assert np.mean(errors[5]) <= 0.95 * np.mean(errors[1])


def test_auto_fit_with_m_pen_in_a_tensor_product_runs():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 800), "z": rng.uniform(0, 1, 800)})
    frame["y"] = np.sin(5 * frame["x"]) * frame["z"] + 0.1 * rng.normal(size=800)
    model = ta.StaticTAM("y ~ te(s(x, k=8, m_pen=2), s(z, k=6))").auto_fit(frame)
    assert np.all(np.isfinite(model.predict(frame)["Estimatedy"].to_numpy()))
