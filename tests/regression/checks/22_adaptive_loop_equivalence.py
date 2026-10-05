# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""An AdaptiveTAM simulation must equal the loop an operator runs: refit a StaticTAM on the residual of the last window, every day, forecast the next day."""
import numpy as np
import pandas as pd

import tam as ta
from common import DATE, GROUP, TARGET, base_model, frame, with_residual

WINDOW, DAYS = 28, 6
FORMULA = "Residualload ~ s(L_Res, k=6, ap=-4)"


def day_index(df):
    return (df[DATE].dt.normalize() - df[DATE].min().normalize()).dt.days


def run(res):
    def equivalence():
        base = base_model(res)
        full = with_residual(frame(), base)
        start = full[DATE] >= pd.Timestamp("2023-07-01") - pd.Timedelta(days=WINDOW + 1)
        sim = full[start].reset_index(drop=True)
        sim["Residualload"] = sim[TARGET] - sim["E_base"]
        d = day_index(sim)
        loop = {}
        for k in range(DAYS):
            window, target = sim[(d >= k) & (d < k + WINDOW)], sim[d == k + WINDOW]
            corrector = ta.StaticTAM(formula="Residualload ~ s(L_Res, k=6, ap=-4)", group_col=GROUP, date_col=DATE, default_alpha_p=-9.0).fit(window)
            pred = corrector.predict(target.drop(columns=["Residualload"]))["EstimatedResidualload"].to_numpy()
            loop.update(zip(zip([k] * len(target), target[GROUP]), target["E_base"].to_numpy() + pred))
        adaptive = ta.AdaptiveTAM(base_model=base, adaptive_formula=FORMULA, update_interval_periods=1, training_window_periods=WINDOW, steps_per_period=1,
                                  default_alpha_p=-9.0, group_col=GROUP, date_col=DATE)
        out = adaptive.predict_online(sim.drop(columns=["E_base", "Residualload"]))
        simulated = dict(zip(zip((day_index(out) - WINDOW).tolist(), out[GROUP]), out[f"AdaptedEstimated{TARGET}"]))
        keys = [key for key in loop if key in simulated and np.isfinite(simulated[key])]
        a, b = np.array([loop[k] for k in keys]), np.array([simulated[k] for k in keys])
        res.value("compared_rows", len(keys))
        res.gap("max_gap", np.max(np.abs(a - b)) if len(keys) else float("nan"))
        res.gap("rmse_gap", np.sqrt(np.mean((a - b) ** 2)) if len(keys) else float("nan"))
        res.forecast("loop", a, a)
        res.forecast("simulation", a, b)

    res.attempt("equivalence", equivalence)
