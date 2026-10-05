# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""AdaptiveTAM correcting a base model: the residual of the same hour the day before, the base effects as features, the clip to the trained range."""
import tam as ta
from common import DATE, GROUP, TARGET, base_model, chrono_split, frame, second_half, with_residual

CORRECTION = "Residualload ~ l(L_Res) + c(day_type_week, n_cat=7, topo='nominal')"


def adaptive(base, **options):
    return ta.AdaptiveTAM(base_model=base, adaptive_formula=CORRECTION, update_interval_periods=7, training_window_periods=56, steps_per_period=1,
                          horizon_steps=1, group_col=GROUP, date_col=DATE, **options)


def run(res):
    holder = {}

    def residual_correction():
        base = holder["base"] = base_model(res)
        sim = holder["sim"] = with_residual(frame(), base)
        out = adaptive(base).predict_online(sim)
        res.forecast("base_only", *second_half(out, "E_base"))
        res.forecast("adapted", *second_half(out, f"AdaptedEstimated{TARGET}"))
        res.value("adapted.base_forecast_where_no_correction", bool(out[f"AdaptedEstimated{TARGET}"].notna().all()))

    res.attempt("residual_correction", residual_correction)

    def clipped():
        out = adaptive(holder["base"], clip_to_train_range=True).predict_online(holder["sim"])
        res.forecast("clip_to_train_range", *second_half(out, f"AdaptedEstimated{TARGET}"))

    res.attempt("clip_to_train_range", clipped)

    def base_effects():
        out = adaptive(holder["base"], add_base_effects=True).predict_online(holder["sim"])
        res.forecast("add_base_effects", *second_half(out, f"AdaptedEstimated{TARGET}"))

    res.attempt("add_base_effects", base_effects)

    def decomposed():
        out = adaptive(holder["base"]).predict_online(holder["sim"])
        res.value("output_columns", ", ".join(c for c in out.columns if c.startswith(("effect_", "Adapted", "Estimated"))))

    res.attempt("output_columns", decomposed)

    def operational_prediction():
        model = adaptive(holder["base"])
        model.predict_online(holder["sim"])
        _, last = chrono_split(holder["sim"])
        res.forecast("predict_after_simulation", last[TARGET], model.predict(last)[f"AdaptedEstimated{TARGET}"])

    res.attempt("predict_after_simulation", operational_prediction)

    def two_bases():
        other = base_model(res, "load ~ f(toy, m=3, s=1, cyclic=True) + l(temperature) + l(load_d7) + c(day_type_week, n_cat=7, topo='nominal')")
        out = adaptive(other).predict_online(with_residual(frame(), other))
        res.forecast("other_base", *second_half(out, f"AdaptedEstimated{TARGET}"))

    res.attempt("other_base", two_bases)
