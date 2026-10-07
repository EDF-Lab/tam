# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A KalmanTAM simulation must equal the loop an operator runs: refit on the history every day, keep the final state for the next day."""
EXPERIMENTAL = True
import inspect

import numpy as np
import pandas as pd

import tam as ta
from common import DATE, GROUP, TARGET, base_model, chrono_split, frame, with_residual

DAYS = 6


def run(res):
    def equivalence():
        base = base_model(res)
        full = with_residual(frame(), base)
        reference, _ = chrono_split(full)
        online = full[(full[DATE] >= pd.Timestamp("2023-07-01")) & (full[DATE] < pd.Timestamp("2023-07-01") + pd.Timedelta(days=DAYS))].reset_index(drop=True)
        settings = dict(kalman_formula="load ~ l(L_Res)", base_model=base, date_col=DATE, group_col=GROUP, horizon_steps=1, process_noise_var=1e-4,
                        use_decomposition=False, block_size=1)
        if "calibration_data" in inspect.signature(ta.KalmanTAM.__init__).parameters:
            settings["calibration_data"] = reference
        day = (online[DATE].dt.normalize() - online[DATE].min().normalize()).dt.days
        loop = {}
        for k in range(1, DAYS):
            today = online[day == k]
            model = ta.KalmanTAM(**settings).fit(online[day < k])
            pred = model.predict(today.drop(columns=[TARGET]))[f"KalmanAdapted_{TARGET}"].to_numpy()
            loop.update(zip(zip([k] * len(today), today[GROUP]), pred))
        out = ta.KalmanTAM(**settings).predict_online(online)
        simulated = dict(zip(zip(day.tolist(), out[GROUP]), out[f"KalmanAdapted_{TARGET}"]))
        keys = [key for key in loop if key in simulated and np.isfinite(simulated[key])]
        a, b = np.array([loop[k] for k in keys]), np.array([simulated[k] for k in keys])
        res.value("compared_rows", len(keys))
        res.gap("max_gap", np.max(np.abs(a - b)) if len(keys) else float("nan"))
        res.gap("rmse_gap", np.sqrt(np.mean((a - b) ** 2)) if len(keys) else float("nan"))
        res.forecast("loop", a, a)

    res.attempt("equivalence", equivalence)
