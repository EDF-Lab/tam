# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spline extrapolation in isolation: fit on winter (January to March, cold), forecast summer (June to August, hot, strictly outside the training range).
The default spline and the three explicit modes are compared; this is where the behaviour of the extrapolation engine is locked down."""
import pandas as pd

import tam as ta
from common import DATE, GROUP, TARGET, frame


def run(res):
    df = frame()
    month = df[DATE].dt.month
    train, test = df[month <= 3], df[month.between(6, 8)]
    res.value("train.max_temperature", float(train["temperature"].max()))
    res.value("test.min_temperature", float(test["temperature"].min()))
    res.value("test.share_beyond_training_range", float((test["temperature"] > train["temperature"].max()).mean()))

    def fit(name, formula):
        def step():
            model = ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE).fit(train)
            res.forecast(name, test[TARGET], model.predict(test)[f"Estimated{TARGET}"])

        res.attempt(name, step)

    fit("default", "load ~ s(temperature, k=8)")                                # model A: the spline as written, no mode given
    fit("continue", "load ~ s(temperature, k=8, extrapolate='continue')")        # model B: the polynomial is continued beyond the range
    fit("constant", "load ~ s(temperature, k=8, extrapolate='constant')")        # held at the value of the edge of the trained range
    fit("linear", "load ~ s(temperature, k=8, extrapolate='linear')")            # the tangent at the edge of the trained range
    fit("linear_effect", "load ~ l(temperature)")                                # what a model without a spline does
