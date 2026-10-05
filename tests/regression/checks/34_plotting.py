# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Component plots: one effect, a tensor product as a surface, a heatmap and curves per level. What is recorded is the drawing's content (axes, curves), not pixels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tam as ta  # noqa: E402
from common import BASE, DATE, GROUP  # noqa: E402

FORMULA = ("load ~ s(temperature, k=8) + te(s(temperature, k=5), s(toy, k=5)) + te(s(temperature, k=5), c(day_type_jf, n_cat=2, topo='nominal')) + "
           + BASE)


def run(res):
    holder = {}

    def prepare():
        holder["model"] = ta.StaticTAM(formula=FORMULA, group_col=GROUP, date_col=DATE).fit(res.train)

    res.attempt("prepare", prepare)

    def draw(name, component, **options):
        def step():
            plt.close("all")
            ta.plot_component(holder["model"], res.test, component, **options)
            figure = plt.gcf()
            axes = figure.get_axes()
            res.value(f"{name}.axes", len(axes))
            res.value(f"{name}.lines", sum(len(a.get_lines()) for a in axes))
            res.value(f"{name}.collections", sum(len(a.collections) for a in axes))
            res.value(f"{name}.xlabel", axes[0].get_xlabel() if axes else "")
            plt.close("all")

        res.attempt(name, step)

    draw("one_feature", "temperature")
    draw("surface", "te_temperature_x_toy")
    draw("heatmap", "te_temperature_x_toy", kind="heatmap")
    draw("curves_per_level", "te_temperature_x_day_type_jf")

    def names():
        from tam.model._math import decomposition_names

        res.value("component_names", ", ".join(decomposition_names(holder["model"].effects_list_)))

    res.attempt("component_names", names)
