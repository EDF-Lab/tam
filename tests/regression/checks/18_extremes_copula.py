# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Extreme-value tail of the residuals and the Gaussian copula that binds several distributional margins."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

SIGMA = "~ s(temperature, k=6) + c(day_type_week, n_cat=7, topo='nominal')"


def margin(target):
    return ta.StaticTAM(formula={"mu": f"{target} ~ s(temperature, k=8) + c(day_type_week, n_cat=7, topo='nominal') + c(day_type_jf, n_cat=2, topo='nominal')",
                                 "sigma": SIGMA}, group_col=GROUP, date_col=DATE, dist_kwargs={"log_target": False})


def run(res):
    def tail():
        model = margin("load").fit(res.train)
        for quantile in (0.9, 0.95):
            gpd = ta.fit_gpd_tail(model, res.train, threshold_quantile=quantile)
            res.value(f"gpd_q{quantile:g}.shape", getattr(gpd, "shape_", float("nan")))
            res.value(f"gpd_q{quantile:g}.scale", getattr(gpd, "scale_", float("nan")))
            res.value(f"gpd_q{quantile:g}.return_level_1e-3", gpd.return_level(1e-3))
            values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            res.forecast(f"gpd_q{quantile:g}.tail_probability", np.zeros(5), gpd.tail_probability(values))
            res.forecast(f"gpd_q{quantile:g}.anomaly_score", np.zeros(5), gpd.anomaly_score(values))

    res.attempt("tail", tail)

    def copula():
        copula = ta.GaussianCopulaTAM({"load": margin("load"), "residential": margin("enedis_residential")}).fit(res.train)
        res.value("copula.correlation", float(copula.correlation_[0, 1]))
        res.forecast("copula.normal_scores", np.zeros(len(res.test) * 2), copula.normal_scores(res.test))
        res.forecast("copula.log_density", np.zeros(len(res.test)), copula.copula_log_density(res.test))
        score = copula.joint_anomaly_score(res.test)
        for column in score.columns:
            if np.issubdtype(score[column].dtype, np.number):
                res.forecast(f"copula.joint.{column}", np.zeros(len(score)), score[column])

    res.attempt("copula", copula)
