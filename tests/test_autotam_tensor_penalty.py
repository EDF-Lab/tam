# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The margins of a tensor product keep the penalty the search wrote (opt-in ``tensor_margin_penalty``): ``ap=0.0`` on the product.
"""

import numpy as np
import pandas as pd

from tam.model.additive import StaticTAM
from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.tensor_penalty import with_tensor_outer_penalty


def test_a_tensor_product_without_an_ap_of_its_own_gets_one():
    assert with_tensor_outer_penalty("y ~ te(s(x1, ap=-3.0), s(x2, ap=-5.0))") == "y ~ te(s(x1, ap=-3.0), s(x2, ap=-5.0), ap=0.0)"
    assert with_tensor_outer_penalty("y ~ l(a) + te(p(x1, 3), s(x2, k=5)) + s(x3, ap=-2)") == "y ~ l(a) + te(p(x1, 3), s(x2, k=5), ap=0.0) + s(x3, ap=-2)"


def test_an_ap_of_the_product_and_formulas_without_a_product_are_left_alone():
    assert with_tensor_outer_penalty("y ~ te(s(x1), s(x2), ap=-2.0)") == "y ~ te(s(x1), s(x2), ap=-2.0)"
    assert with_tensor_outer_penalty("y ~ s(x1, ap=-3.0) + l(x2)") == "y ~ s(x1, ap=-3.0) + l(x2)"


def test_two_products_both_get_one():
    out = with_tensor_outer_penalty("y ~ te(s(a), s(b)) + te(s(c, ap=-1.0), s(d))")
    assert out == "y ~ te(s(a), s(b), ap=0.0) + te(s(c, ap=-1.0), s(d), ap=0.0)"


def _frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(size=n), rng.uniform(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": np.sin(4 * x1) * x2 + rng.normal(scale=0.1, size=n)})


def test_the_engine_gives_the_margins_their_own_penalty_with_the_outer_ap_and_the_product_default_without():
    frame = _frame()
    plain = StaticTAM("y ~ te(s(x1, ap=-3.0), s(x2, ap=-5.0))").fit(frame)
    kept = StaticTAM(with_tensor_outer_penalty("y ~ te(s(x1, ap=-3.0), s(x2, ap=-5.0))")).fit(frame)
    tensor = lambda model: next(e for e in model.effects_list_ if e.__class__.__name__ == "TensorProductEffect")
    np.testing.assert_allclose(tensor(kept).penalty_coordinates(), [1e-3, 1e-5])
    assert max(tensor(plain).penalty_coordinates()) < 1e-9


def test_the_option_reaches_the_search_and_the_champions_carry_the_outer_penalty(tmp_path):
    frame = _frame(300)
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=3, pop_size=12, use_opera=False, seed=4, tensor_margin_penalty=True, export_dir=str(tmp_path))
    assert model.discoverer.tensor_margin_penalty is True
    model.fit(frame.iloc[:240], expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    formulas = [f for group in model.architectures_.values() for f in group if "te(" in f]
    assert all("ap=0.0" in f for f in formulas)
    assert AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path)).discoverer.tensor_margin_penalty is False
