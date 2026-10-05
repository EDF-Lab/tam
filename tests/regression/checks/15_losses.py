# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Losses beyond least squares: the GLM families, expectiles and the robust M-estimators."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

MEAN = "load ~ s(temperature, k=8) + " + BASE


def run(res):
    res.static("l2", MEAN)
    res.static("gamma", MEAN, loss="gamma")
    res.static("poisson", MEAN, loss="poisson")
    for tau in (0.1, 0.5, 0.9):
        res.static(f"expectile_tau{tau:g}", MEAN, loss="expectile", loss_kwargs={"tau": tau})
    res.static("huber", MEAN, loss="huber")
    res.static("huber_delta2", MEAN, loss="huber", loss_kwargs={"delta": 2.0})
    res.static("student_t", MEAN, loss="student_t")
    res.static("student_t_nu10", MEAN, loss="student_t", loss_kwargs={"nu": 10.0})

    def binomial():
        train, test = res.train.copy(), res.test.copy()
        threshold = float(train[TARGET].median())
        for frame in (train, test):
            frame["high"] = (frame[TARGET] > threshold).astype(float)
        model = ta.StaticTAM(formula="high ~ s(temperature, k=8) + l(load_d1) + c(day_type_week, n_cat=7, topo='nominal')", group_col=GROUP, date_col=DATE,
                             loss="binomial").fit(train)
        res.forecast("binomial", test["high"], model.predict(test)["Estimatedhigh"])

    res.attempt("binomial", binomial)

    def unknown_loss():
        try:
            ta.StaticTAM(formula=MEAN, group_col=GROUP, date_col=DATE, loss="no_such_loss")
        except ValueError as error:
            res.value("unknown_loss.message", str(error)[:70])

    res.attempt("unknown_loss", unknown_loss)
