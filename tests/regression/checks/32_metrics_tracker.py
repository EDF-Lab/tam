# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The evaluation helpers: regression metrics and the benchmark tracker that slices a continuous forecast into train / dev / val / test."""
import numpy as np
import pandas as pd

import tam as ta
from common import BASE, DATE, GROUP, TARGET


def run(res):
    def metrics():
        from tam.evaluation.metrics import calculate_regression_metrics

        rng = np.random.default_rng(0)
        y = rng.normal(100, 10, 400)
        for name, noise in (("exact", 0.0), ("noisy", 5.0)):
            for key, value in calculate_regression_metrics(y, y + rng.normal(0, noise, 400) if noise else y).items():
                res.value(f"metrics.{name}.{key}", value)

    res.attempt("metrics", metrics)

    def tracker():
        df = res.train.sort_values([DATE, GROUP]).reset_index(drop=True)
        df["year_part"] = np.where(df.index < len(df) * 0.5, "train", np.where(df.index < len(df) * 0.7, "dev", np.where(df.index < len(df) * 0.85, "val", "test")))
        model = ta.StaticTAM(formula="load ~ s(temperature, k=8) + " + BASE, group_col=GROUP, date_col=DATE).fit(df[df.year_part == "train"])
        data = {name: df[df.year_part == name] for name in ("train", "dev", "val", "test")}
        tracking = ta.BenchmarkTracker("static")
        tracking.y_pred_full = model.predict(df)[f"Estimated{TARGET}"].to_numpy()
        tracking.slice_and_evaluate(data, target_col=TARGET)
        for split in data:
            for metric in ("rmse", "mae"):
                res.value(f"tracker.{split}.{metric}", tracking.get_metric(split, metric))

    res.attempt("tracker", tracker)
