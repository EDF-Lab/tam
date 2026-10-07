# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A formula that mixes four kinds of effects (spline, linear, cyclic Fourier, categorical), its forecast and its decomposition: the components sum to the forecast,
and none of them collapses to zero when mixed with the others (the spread of each component is recorded)."""
import numpy as np

import tam as ta
from common import DATE, GROUP, TARGET

KITCHEN_SINK = "load ~ s(temperature, k=10, p=2, deg=3) + l(load_d1) + f(toy, m=5, s=2, cyclic=True) + c(day_type_week, topo='nominal')"
BIGGER = KITCHEN_SINK + " + te(s(temperature, k=4), s(toy, k=4)) + l(load_d7) + c(month, n_cat=12, topo='ordinal', p_order=2, ap=-1)"


def mixed(res, label, formula):
    train, test = res.train, res.test
    model = ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE).fit(train)
    forecast = model.predict(test)[f"Estimated{TARGET}"].to_numpy()
    parts = model.decompose_prediction(test)
    components = [c for c in parts.columns if c.startswith("effect_")]
    total = parts[components].sum(axis=1).to_numpy()                        # the intercept is the component effect_offset
    gap = float(np.max(np.abs(total - forecast)))
    res.forecast(f"{label}.test", test[TARGET], forecast)
    res.value(f"{label}.test_rmse", float(np.sqrt(np.mean((forecast - test[TARGET].to_numpy()) ** 2))))
    res.value(f"{label}.components", ", ".join(components))
    res.gap(f"{label}.max_gap_between_the_sum_of_the_components_and_the_forecast", gap)
    assert gap <= 1e-6, f"the components do not sum to the forecast (gap {gap})"
    collapsed = []
    for column in components:
        spread = float(parts[column].std())
        res.value(f"{label}.std_comp_{column.removeprefix('effect_')}", spread)
        if column != "effect_offset" and not spread > 0:
            collapsed.append(column)
    res.value(f"{label}.components_that_collapsed_to_zero", ", ".join(collapsed) or "none")
    assert not collapsed, f"components without any variation: {collapsed}"


def run(res):
    res.attempt("kitchen_sink", lambda: mixed(res, "mixed", KITCHEN_SINK))
    res.attempt("bigger", lambda: mixed(res, "bigger", BIGGER))

    def per_group():
        model = ta.StaticTAM(formula=KITCHEN_SINK, group_col=GROUP, date_col=DATE).fit(res.train)
        parts = model.decompose_prediction(res.test)
        for hour in (3, 12, 20):
            rows = parts[res.test[GROUP].to_numpy() == hour]
            res.value(f"mixed.hour{hour}.std_comp_temperature", float(rows["effect_s_temperature" if "effect_s_temperature" in rows else "effect_temperature"].std()))

    res.attempt("per_group", per_group)
