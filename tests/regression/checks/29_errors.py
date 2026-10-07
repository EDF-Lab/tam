# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The errors a user meets: what is raised, and what it says, when a formula, a column, a call or a value is wrong."""
import tam as ta
from common import BASE, DATE, GROUP

GOOD = "load ~ s(temperature, k=8) + " + BASE


def run(res):
    train, test = res.train, res.test
    fitted = {}

    def fit_good():
        fitted["model"] = ta.StaticTAM(formula=GOOD, group_col=GROUP, date_col=DATE).fit(train)

    res.attempt("prepare", fit_good)

    def raises(name, call):
        def step():
            try:
                call()
                res.value(name, "no error")
            except Exception as error:                      # noqa: BLE001 - the message is what is recorded
                res.value(name, f"{type(error).__name__}: {str(error)[:90]}")

        res.attempt(f"{name}.harness", step)

    def build(formula, **options):
        return lambda: ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE, **options).fit(train)

    raises("formula.empty_argument", build("load ~ s(temperature,, k=8)"))
    raises("formula.trailing_comma", build("load ~ s(temperature, k=8,)"))
    raises("formula.unbalanced_parenthesis", build("load ~ s(temperature, k=8"))
    raises("formula.unknown_effect", build("load ~ zz(temperature)"))
    raises("formula.no_target", build("s(temperature, k=8)"))
    raises("formula.no_effect", build("load ~"))
    raises("formula.unknown_column", build("load ~ s(nothing_like_it, k=8)"))
    raises("formula.unknown_topology", build("load ~ c(hour, n_cat=24, topo='circular')"))
    raises("formula.unknown_extrapolation", build("load ~ s(temperature, k=8, extrapolate='explode')"))
    raises("formula.bad_knots", build("load ~ s(temperature, k=1)"))
    raises("model.unknown_loss", lambda: ta.StaticTAM(formula=GOOD, loss="no_such_loss"))
    raises("model.predict_before_fit", lambda: ta.StaticTAM(formula=GOOD, group_col=GROUP, date_col=DATE).predict(test))
    raises("model.missing_column_at_predict", lambda: fitted["model"].predict(test.drop(columns=["temperature"])))
    raises("model.unknown_group_at_predict", lambda: fitted["model"].predict(test.assign(hour=99)))
    raises("model.empty_training_frame", lambda: fitted["model"].fit(train.iloc[0:0]))
    raises("model.quantiles_on_a_point_model", lambda: fitted["model"].predict_quantiles(test))
    raises("model.conformal_before_calibration", lambda: fitted["model"].predict_intervals(test))
    raises("adaptive.unfitted_base_model", lambda: ta.AdaptiveTAM(base_model=ta.StaticTAM(formula=GOOD, group_col=GROUP, date_col=DATE),
                                                                    adaptive_formula="Residualload ~ l(L_Res)", update_interval_periods=1,
                                                                    training_window_periods=7, steps_per_period=1))
    raises("adaptive.window_longer_than_the_data", lambda: ta.AdaptiveTAM(adaptive_formula="load ~ l(load_d1)", update_interval_periods=1,
                                                                            training_window_periods=10000, steps_per_period=1, group_col=GROUP,
                                                                            date_col=DATE).predict_online(train))
    raises("kalman.no_formula_effect", lambda: ta.KalmanTAM(kalman_formula="load ~", group_col=GROUP, date_col=DATE).predict_online(train))
    raises("kalman.negative_noise", lambda: ta.KalmanTAM(kalman_formula="load ~ l(load_d1)", group_col=GROUP, date_col=DATE, process_noise_var=-1.0).predict_online(train))
    raises("opera.no_expert", lambda: ta.OperaTAM("load ~ ", group_col=GROUP, date_col=DATE))
    raises("opera.unknown_algorithm", lambda: ta.OperaTAM("load ~ l(load_d1) + l(load_d7)", algorithm="NO_SUCH", group_col=GROUP, date_col=DATE))
    raises("hierarchical.unknown_node", lambda: ta.HierarchicalTAM(structure={"a": ["b"]}, formulas="Target ~ l(temperature)", node_col="Node",
                                                                  date_col=DATE).fit(train.assign(Node="a", Target=train["load"])))
    raises("copula.single_margin", lambda: ta.GaussianCopulaTAM({"only": ta.StaticTAM(formula={"mu": GOOD, "sigma": "~ l(temperature)"})}))
    raises("conformal.bad_alpha", lambda: ta.SafetyTAM(alpha=1.5).calibrate_scores([1.0, 2.0, 3.0]))
