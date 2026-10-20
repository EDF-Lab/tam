# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""AutoTAM on a very small search (a handful of models, one pooled series with the hour as a feature): the pipeline runs end to end and forecasts. Its exports go to a temporary folder. The search is seeded, so the experts and the forecasts are recorded exactly."""
EXPERIMENTAL = True
import hashlib
import tempfile

import pandas as pd

import tam as ta
from common import DATE, TARGET, frame

SEED = 0


def run(res):
    def small_search():
        df = frame()
        fit = df[df[DATE] < pd.Timestamp("2023-06-01")]
        dev = df[(df[DATE] >= pd.Timestamp("2023-06-01")) & (df[DATE] < pd.Timestamp("2023-08-01"))]
        val = df[(df[DATE] >= pd.Timestamp("2023-08-01")) & (df[DATE] < pd.Timestamp("2023-10-01"))]
        test = df[df[DATE] >= pd.Timestamp("2023-10-01")]
        with tempfile.TemporaryDirectory() as folder:
            model = ta.AutoTAM(formula="load ~ AutoPipe(temperature, toy, day_type_week, day_type_jf, load_d1, hour)", n_experts=3, pop_size=6,
                               export_dir=folder, seed=SEED)
            model.fit(df_fit=fit, df_dev=dev, df_val=val, date_col=DATE, expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
            out = model.predict(test, date_col=DATE)
        # With a seed the search is reproducible, on this machine and on another one: the experts that were trained (their number and a digest of the
        # formulas of the static experts, in the order they were evaluated) and the forecasts are recorded, so a change of the search shows in the diff.
        formulas = [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]
        res.gap("experts_missing", max(0, 3 - len(model.trained_experts)), tolerance=0)
        res.value("experts", len(model.trained_experts))
        res.value("static_formulas", len(formulas))
        res.value("static_formulas_digest", hashlib.sha256("\n".join(formulas).encode("utf-8")).hexdigest()[:12])
        res.value("output_columns", ", ".join(out.columns[-3:]))
        for column in ("AutoTAM_Apex_Ensemble", "Ensemble_Static"):
            if column in out.columns:
                res.forecast(column, test[TARGET].to_numpy()[: len(out)], out[column].to_numpy())

    res.attempt("small_search", small_search)
