# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
``te()`` has one smoothing parameter per margin (mgcv semantics): the penalty of a tensor product is
``sum_i lambda_i (I x ... x P_i x ... x I)`` and each ``lambda_i`` is its own GCV coordinate.

With fixed lambdas the penalty matrix is the one of v1.3.0, ``lambda_te * sum_i (I x ... x P_i(lambda_i) x ... x I)``: a te-level
``ap`` is folded into the margins at construction.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import TensorProductEffect, create_effects_from_parsed_terms
from tam.common.utils import parse_formula_to_terms


def _effects(formula_rhs, default_alpha_p=-4.0):
    _, terms = parse_formula_to_terms("y ~ " + formula_rhs)
    return create_effects_from_parsed_terms(terms, token_values={}, default_alpha_p=default_alpha_p, data_info={})


def _te(formula_rhs, default_alpha_p=-4.0):
    return next(e for e in _effects(formula_rhs, default_alpha_p) if isinstance(e, TensorProductEffect))


def _v130_penalty(margins, lambda_te):
    """The v1.3.0 penalty: lambda_te * sum_i (I x ... x P_i x ... x I), P_i carrying its own lambda_i."""
    dims = [m.get_n_coeffs() for m in margins]
    total = 0
    for i in range(len(margins)):
        term = None
        for j in range(len(margins)):
            mat = margins[j].build_penalty_matrix() if i == j else torch.eye(dims[j], dtype=torch.get_default_dtype())
            term = mat if term is None else torch.kron(term, mat)
        total = total + term
    return total * lambda_te


# ------------------------------------------------------------------ the effect API
def test_a_plain_effect_has_one_penalty_coordinate():
    spline = _effects("s(x, k=6, ap=-3)")[1]
    assert spline.n_penalty_coordinates == 1 and spline.penalty_coordinates() == [pytest.approx(1e-3)]
    spline.set_penalty_coordinates([1e-5])
    assert spline.lambda_p == pytest.approx(1e-5)


def test_a_tensor_product_has_one_coordinate_per_margin():
    te = _te("te(s(x, k=5, ap=-3), f(y, m=2, ap=-6))", default_alpha_p=-9.0)
    assert te.n_penalty_coordinates == 2
    # no te-level ap: the default weight is folded into each margin, the te weight is 1
    assert te.lambda_p == 1.0
    np.testing.assert_allclose(te.penalty_coordinates(), [1e-3 * 1e-9, 1e-6 * 1e-9])


def test_a_te_level_ap_is_folded_into_the_margins_silently():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        te = _te("te(s(x, k=5, ap=-3), f(y, m=2, ap=-6), ap=-2)")
    assert te.lambda_p == 1.0
    np.testing.assert_allclose(te.penalty_coordinates(), [1e-3 * 1e-2, 1e-6 * 1e-2])


def test_set_penalty_coordinates_moves_the_margins_and_the_matrix():
    te = _te("te(s(x, k=5, ap=-3), s(y, k=5, ap=-3))")
    before = te.build_penalty_matrix()
    te.set_penalty_coordinates([te.penalty_coordinates()[0] * 10.0, te.penalty_coordinates()[1]])
    after = te.build_penalty_matrix()
    assert not torch.allclose(before, after)
    with pytest.raises(ValueError):
        te.set_penalty_coordinates([1.0])


def test_a_nested_tensor_product_has_no_coordinates():
    te = _te("te(s(x, k=4, ap=-3), s(y, k=4, ap=-3))")
    outer = TensorProductEffect([te, _effects("s(z, k=4, ap=-3)")[1]], 1.0, "continue")
    with pytest.raises(NotImplementedError):
        outer.penalty_coordinates()


# ------------------------------------------------------------------ folding a weight into a margin is exact for every margin type
@pytest.mark.parametrize("rhs", [
    "s(x, k=6)", "f(x, m=3)", "p(x, deg=4)", "w(x)", "c(c, n_cat=4, topo='ordinal')", "rbf(x, n_centers=6)", "n(x, n_neurons=6)", "t(x)",
    "l(x)", "lt(x, slope=z)",
])
def test_every_margin_penalty_is_linear_in_its_weight(rhs):
    """The weight of a te() is multiplied into its margins: exact only if each margin's penalty is proportional to its weight."""
    effect = _effects(rhs, -2.0)[1]
    effect.lambda_p = 1e-2
    small = effect.build_penalty_matrix()
    effect.lambda_p = 7e-2
    large = effect.build_penalty_matrix()
    small, large = (m.to_dense() if m.is_sparse else m for m in (small, large))
    torch.testing.assert_close(large, 7.0 * small, rtol=1e-9, atol=1e-14)


# ------------------------------------------------------------------ fixed lambdas: the v1.3.0 matrix
@pytest.mark.parametrize("te_rhs,margins_rhs,lambda_te", [
    ("te(s(x, k=5, ap=-3), f(y, m=2, ap=-6))", "s(x, k=5, ap=-3) + f(y, m=2, ap=-6)", 1e-4),              # no te-level ap: the default weight
    ("te(s(x, k=5, ap=-3), f(y, m=2, ap=-6), ap=-2)", "s(x, k=5, ap=-3) + f(y, m=2, ap=-6)", 1e-2),       # a te-level ap
])
def test_fixed_lambdas_give_the_v130_penalty_matrix(te_rhs, margins_rhs, lambda_te):
    te = _te(te_rhs, default_alpha_p=-4.0)
    margins = _effects(margins_rhs, -4.0)[1:]                       # without the offset
    expected = _v130_penalty(margins, lambda_te)
    np.testing.assert_allclose(te.build_penalty_matrix().numpy(), expected.numpy(), rtol=1e-10, atol=1e-14)


def test_fixed_penalty_predictions_are_unchanged_by_the_folding():
    """A fit with fixed weights does not depend on how the weight is split between the te and its margins."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(0, 1, n)})
    df["y"] = np.sin(2 * df["x"]) * df["z"] + rng.normal(0, 0.1, n)
    a = ta.StaticTAM(formula="y ~ te(s(x, k=6, ap=-3), s(z, k=6, ap=-3), ap=-2)", date_col="date").fit(df)
    b = ta.StaticTAM(formula="y ~ te(s(x, k=6, ap=-2), s(z, k=6, ap=-2), ap=-3)", date_col="date").fit(df)
    np.testing.assert_allclose(a.predict(df)["Estimatedy"].to_numpy(), b.predict(df)["Estimatedy"].to_numpy(), rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------ GCV reaches the smooth limit and learns anisotropy
def _edf_of_the_te_block(model, df):
    from tam.common.utils import _ensure_dummies
    x, _, _ = model._prepare_data(_ensure_dummies(df, model.group_col_, model.date_col_), target_col=model.target_col_)
    phi = model._build_design_matrix(x)[0]
    n = phi.shape[0]
    penalty = model._build_penalty_matrix()
    cov = phi.mT @ phi
    hat = torch.linalg.solve(cov + n * penalty + n * 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype), cov)
    start = sum(e.get_n_coeffs() for e in model.effects_list_[:-1])
    return float(torch.trace(hat[start:, start:]))


def test_gcv_reaches_the_linear_limit_on_a_plane():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = df["x"] + df["z"] + rng.normal(0, 0.3, n)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=20), s(z, k=20))", date_col="date")
    model.auto_fit(df, number_of_steps=10, gamma=1.0)
    assert _edf_of_the_te_block(model, df) < 5.0


def test_gcv_learns_anisotropy():
    """A surface smooth in x and rough in z: the penalty of x must end at least 100 times the penalty of z."""
    rng = np.random.default_rng(2)
    n = 1500
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = 0.5 * df["x"] + np.sin(6 * np.pi * df["z"]) + rng.normal(0, 0.1, n)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=15), s(z, k=15))", date_col="date")
    model.auto_fit(df, number_of_steps=10, gamma=1.0)
    lambda_x, lambda_z = next(e for e in model.effects_list_ if isinstance(e, TensorProductEffect)).penalty_coordinates()
    assert lambda_x >= 100.0 * lambda_z, (lambda_x, lambda_z)


def test_auto_fit_stores_one_lambda_per_margin_and_refit_reproduces_the_model():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = np.sin(3 * df["x"]) * df["z"] + rng.normal(0, 0.1, n)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=8), s(z, k=8)) + l(x)", date_col="date")
    model.auto_fit(df, number_of_steps=6)
    te = next(e for e in model.effects_list_ if isinstance(e, TensorProductEffect))
    assert te.lambda_p == 1.0 and len(te.penalty_coordinates()) == 2
    selected = model.coefficients_.detach().clone()
    model.fit(df)
    np.testing.assert_allclose(model.coefficients_.detach().numpy(), selected.numpy(), rtol=1e-8, atol=1e-10)


def test_the_report_and_the_summary_show_one_value_per_margin(capsys):
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = df["x"] * df["z"] + rng.normal(0, 0.1, n)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=6), s(z, k=6))", date_col="date")
    model.auto_fit(df, number_of_steps=4)
    out = capsys.readouterr().out
    assert "[x]" in out and "[z]" in out
    reg = model.summary().set_index("Type")["Reg (log10)"]
    assert " / " in str(reg["TensorProduct"])


def test_an_explicit_alpha_list_is_tried_on_every_coordinate():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = df["x"] * df["z"] + rng.normal(0, 0.1, n)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=6), s(z, k=6))", date_col="date")
    model.auto_fit(df, alpha_p_list=[-6.0, -3.0, 0.0])
    te = next(e for e in model.effects_list_ if isinstance(e, TensorProductEffect))
    assert all(np.log10(v) in (-6.0, -3.0, 0.0) or abs(np.log10(v) - round(np.log10(v))) < 1e-9 for v in te.penalty_coordinates())


# ------------------------------------------------------------------ chunked path == cached path
def test_the_chunked_path_gives_the_cached_path_result(monkeypatch):
    rng = np.random.default_rng(6)
    n = 300
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = np.sin(3 * df["x"]) * df["z"] + rng.normal(0, 0.1, n)

    def fit():
        m = ta.StaticTAM(formula="y ~ te(s(x, k=6), s(z, k=6)) + l(x)", date_col="date")
        m.auto_fit(df, number_of_steps=6)
        te = next(e for e in m.effects_list_ if isinstance(e, TensorProductEffect))
        return m.coefficients_.detach().numpy().copy(), np.array(te.penalty_coordinates())

    cached_coef, cached_lambdas = fit()
    from tam.common import hardware
    monkeypatch.setattr(hardware.hw, "get_available_memory", lambda *a, **k: 1)       # nothing fits: the chunked branch
    chunked_coef, chunked_lambdas = fit()
    np.testing.assert_allclose(chunked_lambdas, cached_lambdas, rtol=1e-9)
    np.testing.assert_allclose(chunked_coef, cached_coef, rtol=1e-7, atol=1e-9)


# ------------------------------------------------------------------ grid search tokens reach the margins
def test_grid_search_tokens_reach_each_margin():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n)})
    df["y"] = df["x"] * df["z"] + rng.normal(0, 0.1, n)
    val = df.iloc[150:]
    model = ta.StaticTAM(formula="y ~ te(s(x, k=5, ap='ga'), s(z, k=5, ap='gb'))", date_col="date")
    best = model.grid_search_fit(df.iloc[:150], val, {"ga": [-6.0, -2.0], "gb": [-6.0, -2.0]})
    te = next(e for e in best.effects_list_ if isinstance(e, TensorProductEffect))
    assert len(te.penalty_coordinates()) == 2
