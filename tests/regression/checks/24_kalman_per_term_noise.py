# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""KalmanTAM with one process noise per formula term: a dictionary of variances, the term names, the checks on the values."""
EXPERIMENTAL = True
import tam as ta
from common import DATE, GROUP, TARGET, base_model, frame, second_half, with_residual

FORMULA = "load ~ l(L_Res) + l(load_d7) + l(temperature)"


def kalman(base, noise):
    return ta.KalmanTAM(kalman_formula=FORMULA, base_model=base, group_col=GROUP, date_col=DATE, horizon_steps=1, process_noise_var=noise,
                        observation_noise_var=1.0, use_decomposition=False, block_size=1, calibration_steps=20)


def run(res):
    holder = {}

    def prepare():
        holder["base"] = base_model(res)
        holder["sim"] = with_residual(frame(), holder["base"])

    res.attempt("prepare", prepare)

    def forecast(name, noise):
        def step():
            out = kalman(holder["base"], noise).predict_online(holder["sim"])
            res.forecast(name, *second_half(out, f"KalmanAdapted_{TARGET}"))

        res.attempt(name, step)

    forecast("float", 1e-3)
    forecast("default_only", {"default": 1e-3})
    forecast("same_everywhere", {"default": 1e-3, "L_Res": 1e-3, "load_d7": 1e-3, "temperature": 1e-3})
    forecast("one_term_free", {"default": 1e-4, "L_Res": 1e-1})
    forecast("one_term_frozen", {"default": 1e-3, "temperature": 0.0})
    forecast("offset_without_boost", {"default": 1e-3, "offset": 1e-3})

    def names():
        columns = kalman(holder["base"], 1e-3).term_columns()
        res.value("term_columns", ", ".join(f"{k}:{v}" for k, v in columns.items()))

    res.attempt("term_columns", names)

    def refused(name, noise):
        def step():
            try:
                kalman(holder["base"], noise).predict_online(holder["sim"])
                res.value(f"refused.{name}", "accepted")
            except ValueError as error:
                res.value(f"refused.{name}", str(error)[:80])

        res.attempt(f"refused.{name}", step)

    refused("unknown_term", {"default": 1e-3, "tempratur": 1e-3})
    refused("negative", {"default": 1e-3, "L_Res": -1.0})
    refused("nan", float("nan"))
