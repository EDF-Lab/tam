# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""AutoTAM on a very small search (a handful of models, one pooled series with the hour as a feature): the pipeline runs end to end and forecasts. Its exports go to a temporary folder."""
EXPERIMENTAL = True
import tempfile

import pandas as pd

import tam as ta
from common import DATE, TARGET, frame

RMSE_CEILING = 6000


def run(res):
    def small_search():
        df = frame()
        fit = df[df[DATE] < pd.Timestamp("2023-06-01")]
        dev = df[(df[DATE] >= pd.Timestamp("2023-06-01")) & (df[DATE] < pd.Timestamp("2023-08-01"))]
        val = df[(df[DATE] >= pd.Timestamp("2023-08-01")) & (df[DATE] < pd.Timestamp("2023-10-01"))]
        test = df[df[DATE] >= pd.Timestamp("2023-10-01")]
        with tempfile.TemporaryDirectory() as folder:
            model = ta.AutoTAM(formula="load ~ temperature + toy + day_type_week + day_type_jf + load_d1 + hour", n_experts=3, pop_size=6, export_dir=folder)
            model.fit(df_fit=fit, df_dev=dev, df_val=val, date_col=DATE, expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
            out = model.predict(test, date_col=DATE)
        # The search is evolutionary: another processor takes another path through it, so the exact models and errors are not recorded. What is: the
        # outputs, no missing value, at least the requested experts, and an error below a ceiling that a working search stays under.
        res.gap("experts_missing", max(0, 3 - len(model.trained_experts)), tolerance=0)
        res.value("output_columns", ", ".join(out.columns[-3:]))
        for column in ("AutoTAM_Apex_Ensemble", "Ensemble_Static"):
            if column in out.columns:
                forecast = out[column].to_numpy()
                error = test[TARGET].to_numpy()[: len(out)] - forecast
                res.value(f"{column}.n", len(forecast))
                res.value(f"{column}.missing", int(pd.isna(forecast).sum()))
                res.gap(f"{column}.rmse_above_{RMSE_CEILING}", max(0.0, float((error ** 2).mean() ** 0.5) - RMSE_CEILING), tolerance=0)

    res.attempt("small_search", small_search)
