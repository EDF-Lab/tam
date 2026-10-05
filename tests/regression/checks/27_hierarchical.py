# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""HierarchicalTAM on the Enedis hierarchy (total = residential + professional + enterprise + industrial): joint fit, coherence of the forecasts, the strength of the constraint."""
EXPERIMENTAL = True
import numpy as np
import pandas as pd

import tam as ta
from common import DATE, frame, split

SEGMENTS = {"residential": "enedis_residential", "professional": "enedis_professional", "entreprise": "enedis_entreprise", "industrial": "enedis_industrial"}
NODES = {"total": "enedis_total", **SEGMENTS}
FORMULA = "Target ~ s(temperature, k=8) + c(hour, n_cat=24, topo='nominal') + c(day_type_week, n_cat=7, topo='nominal') + f(toy, m=4, s=1, cyclic=True)"
CALENDAR = [DATE, "hour", "toy", "day_type_week", "temperature"]


def long(df, nodes):
    return pd.concat([df[CALENDAR].assign(Node=name, Target=df[column].to_numpy()) for name, column in nodes.items()], ignore_index=True)


def run(res):
    train, test = split(frame())

    def hierarchy(name, structure, nodes, **options):
        def step():
            model = ta.HierarchicalTAM(structure=structure, formulas=FORMULA, node_col="Node", date_col=DATE, **options).fit(long(train, nodes))
            pred = model.predict(long(test, nodes))
            by_node = {n: pred[pred["Node"] == n].reset_index(drop=True) for n in nodes}
            for node, column in nodes.items():
                res.forecast(f"{name}.{node}", test[column], by_node[node]["EstimatedTarget"])
            parent = next(iter(structure))
            children = sum(by_node[child]["EstimatedTarget"].to_numpy() for child in structure[parent])
            res.value(f"{name}.max_incoherence", float(np.max(np.abs(by_node[parent]["EstimatedTarget"].to_numpy() - children))))

        res.attempt(name, step)

    hierarchy("one_level", {"total": list(SEGMENTS)}, NODES)
    for strength in (0.01, 100.0):
        hierarchy(f"strength{strength:g}", {"total": list(SEGMENTS)}, NODES, lambda_p_hier=strength)
    two = {"total": ["small", "entreprise", "industrial"], "small": ["residential", "professional"]}
    nodes_two = {"total": "enedis_total", "small": "small", "entreprise": "enedis_entreprise", "industrial": "enedis_industrial",
                 "residential": "enedis_residential", "professional": "enedis_professional"}
    for df in (train, test):
        df["small"] = df["enedis_residential"] + df["enedis_professional"]
    hierarchy("two_levels", two, nodes_two)

    def per_node_formulas():
        formulas = {node: FORMULA for node in NODES}
        formulas["industrial"] = "Target ~ l(temperature) + c(hour, n_cat=24, topo='nominal')"
        model = ta.HierarchicalTAM(structure={"total": list(SEGMENTS)}, formulas=formulas, node_col="Node", date_col=DATE).fit(long(train, NODES))
        pred = model.predict(long(test, NODES))
        res.forecast("own_formula_industrial", test["enedis_industrial"], pred[pred["Node"] == "industrial"]["EstimatedTarget"])

    res.attempt("per_node_formulas", per_node_formulas)
