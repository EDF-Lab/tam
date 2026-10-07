# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""OperaTAM aggregating experts online: the two algorithms, the learning rate, the loss, the horizon, the weights it ends with."""
import numpy as np
import pandas as pd

import tam as ta
from common import BASE, DATE, GROUP, TARGET, base_model, frame, second_half

FORMULA = "load ~ l(E1) + l(E2) + l(E3)"


def experts():
    df = frame()
    first = df[df[DATE] < pd.Timestamp("2023-07-01")]
    out = df[[DATE, GROUP, TARGET]].copy()
    for name, formula in (("E1", "load ~ f(toy, m=6, s=1, cyclic=True) + l(temperature) + l(load_d7) + " + BASE),
                          ("E2", "load ~ f(toy, m=3, s=1, cyclic=True) + l(temperature) + " + BASE),
                          ("E3", "load ~ l(toy) + l(temperature) + l(load_d7) + " + BASE)):
        model = ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE).fit(first)
        out[name] = model.predict(df)[f"Estimated{TARGET}"].to_numpy()
    return out


def run(res):
    holder = {}

    def prepare():
        holder["frame"] = experts()
        for name in ("E1", "E2", "E3"):
            res.forecast(f"expert_{name}", *second_half(holder["frame"], name))

    res.attempt("prepare", prepare)

    def aggregate(name, **options):
        def step():
            model = ta.OperaTAM(FORMULA, group_col=GROUP, date_col=DATE, **options)
            out = model.predict_online(holder["frame"])
            res.forecast(name, *second_half(out, "prediction_opera"))
            weights = [c for c in out.columns if c.startswith("weight_")]
            res.value(f"{name}.weight_columns", ", ".join(weights))
            part = out[out[DATE] >= pd.Timestamp("2023-07-01")]
            for column in weights:
                res.value(f"{name}.mean_{column}", float(part[column].mean()))

        res.attempt(name, step)

    aggregate("mlpol", algorithm="MLPOL")
    aggregate("ewa", algorithm="EWA")
    aggregate("ewa_eta_small", algorithm="EWA", eta=0.01)
    aggregate("ewa_eta_large", algorithm="EWA", eta=5.0)
    aggregate("mlpol_absolute_loss", algorithm="MLPOL", loss_type="absolute")
    aggregate("ewa_absolute_loss", algorithm="EWA", loss_type="absolute")
    aggregate("horizon2", algorithm="MLPOL", horizon_steps=2)

    def final_weights():
        model = ta.OperaTAM(FORMULA, group_col=GROUP, date_col=DATE).fit(holder["frame"])
        out = model.predict(holder["frame"].tail(24 * 7))
        res.forecast("fit_then_predict", holder["frame"][TARGET].tail(24 * 7), out["prediction_opera"])
        res.value("weights_history.groups", len(model.weights_history_))

    res.attempt("fit_then_predict", final_weights)

    def three_experts_one_equal():
        frame_ = holder["frame"].copy()
        frame_["E2"] = frame_["E1"]
        out = ta.OperaTAM(FORMULA, group_col=GROUP, date_col=DATE).predict_online(frame_)
        res.forecast("duplicated_expert", *second_half(out, "prediction_opera"))

    res.attempt("duplicated_expert", three_experts_one_equal)

    def single_series():
        part = holder["frame"][holder["frame"][GROUP] == 12]
        out = ta.OperaTAM(FORMULA, date_col=DATE).predict_online(part)
        res.forecast("one_series", *second_half(out, "prediction_opera"))

    res.attempt("one_series", single_series)
