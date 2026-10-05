# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""KalmanTAM tracking the residual of a base model: block size, process and observation noise, decomposition or not, calibration of the scaling."""
EXPERIMENTAL = True
import tam as ta
from common import DATE, GROUP, TARGET, base_model, chrono_split, frame, second_half, with_residual


def kalman(base, **options):
    settings = dict(kalman_formula="load ~ l(L_Res)", base_model=base, group_col=GROUP, date_col=DATE, horizon_steps=1, process_noise_var=0.5,
                    observation_noise_var=1.0, use_decomposition=True, block_size=24)
    settings.update(options)
    return ta.KalmanTAM(**settings)


def run(res):
    holder = {}

    def prepare():
        holder["base"] = base_model(res)
        holder["sim"] = with_residual(frame(), holder["base"])

    res.attempt("prepare", prepare)
    base, sim = holder["base"], holder["sim"]

    def simulate(name, **options):
        def step():
            out = kalman(base, **options).predict_online(sim)
            res.forecast(name, *second_half(out, f"KalmanAdapted_{TARGET}"))

        res.attempt(name, step)

    simulate("default")
    for block in (1, 7, 48):
        simulate(f"block{block}", block_size=block)
    for noise in (1e-4, 1e-2, 5.0):
        simulate(f"process_noise{noise:g}", process_noise_var=noise)
    for noise in (0.1, 10.0):
        simulate(f"observation_noise{noise:g}", observation_noise_var=noise)
    simulate("without_decomposition", use_decomposition=False)
    simulate("two_terms", kalman_formula="load ~ l(L_Res) + l(load_d7)")
    simulate("calibration_steps", calibration_steps=24 * 10)

    def calibration_data():
        first, _ = chrono_split(frame())
        out = kalman(base, calibration_data=with_residual(first, base)).predict_online(sim)
        res.forecast("calibration_data", *second_half(out, f"KalmanAdapted_{TARGET}"))

    res.attempt("calibration_data", calibration_data)

    def operational():
        model = kalman(base, block_size=1).fit(sim)
        _, last = chrono_split(sim)
        res.forecast("fit_then_predict", last[TARGET], model.predict(last.drop(columns=[TARGET]))[f"KalmanAdapted_{TARGET}"])

    res.attempt("fit_then_predict", operational)
