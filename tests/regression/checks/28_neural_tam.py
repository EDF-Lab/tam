# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeuralTAM (backfitted networks): the seed, the activations, the validation split, the guard that keeps a network only where it beats the closed-form effect, and the decomposition into effects."""
EXPERIMENTAL = True
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET


def neural(**options):
    settings = dict(formula="load ~ n(temperature, n_neurons=8, act='relu') + l(load_d7) + " + BASE, group_col=GROUP, date_col=DATE,
                    epochs=8, lr=0.01, batch_size=256, patience=4, backfit_cycles=1)
    settings.update(options)
    return ta.NeuralTAM(**settings)


def run(res):
    train, test = res.train, res.test

    def fit(name, **options):
        def step():
            model = neural(**options).fit(train)
            res.forecast(name, test[TARGET], model.predict(test)[f"Estimated{TARGET}"])
            kept = [used for group in model.network_used_.values() for used in group.values()]
            res.value(f"{name}.networks_kept", f"{sum(kept)} of {len(kept)}")
            return model

        res.attempt(name, step)

    fit("default")
    fit("seed1", seed=1)
    fit("seed42", seed=42)
    fit("tanh", formula="load ~ n(temperature, n_neurons=8, act='tanh') + l(load_d7) + " + BASE)
    fit("cos", formula="load ~ n(temperature, n_neurons=8, act='cos') + l(load_d7) + " + BASE)
    fit("two_networks", formula="load ~ n(temperature, n_neurons=8, act='relu') + n(toy, n_neurons=8, act='tanh') + " + BASE)
    fit("interaction", formula="load ~ n(temperature, others='toy', n_neurons=8, act='relu') + " + BASE)
    fit("shuffled_split", shuffle_split=True)
    fit("val_split_half", val_split=0.5)
    fit("two_cycles", backfit_cycles=2)
    fit("shared_feature", formula="load ~ l(temperature) + n(temperature, n_neurons=8, act='relu') + " + BASE)       # the feature of the network is shared with a linear term
    fit("guard_off", guard=False)
    fit("shared_feature_guard_off", formula="load ~ l(temperature) + n(temperature, n_neurons=8, act='relu') + " + BASE, guard=False)
    fit("strong_network", epochs=60, lr=0.02, batch_size=64, patience=60)       # long enough for the networks to beat the closed-form effect where they can
    fit("default_training", epochs=500, lr=0.01, batch_size=1024, patience=25)         # the defaults of the class: a flat start of the validation loss must not stop the learning

    def reproducible():
        a = neural(seed=3).fit(train).predict(test)[f"Estimated{TARGET}"].to_numpy()
        np.random.rand(1000)
        b = neural(seed=3).fit(train).predict(test)[f"Estimated{TARGET}"].to_numpy()
        res.value("same_seed_same_forecast", bool(np.array_equal(a, b)))

    res.attempt("reproducible", reproducible)

    def decomposition():
        model = neural().fit(train)
        parts = model.decompose_prediction(test)
        effects = [c for c in parts.columns if c.startswith("effect_")]
        res.value("decompose.effects", ", ".join(effects))
        res.gap("decompose.max_gap_to_forecast", np.max(np.abs(parts[effects].sum(axis=1).to_numpy() - model.predict(test)[f"Estimated{TARGET}"].to_numpy())))

    res.attempt("decomposition", decomposition)

    def as_base_model():
        model = neural().fit(train)
        from common import with_residual
        sim = with_residual(test, model)
        res.forecast("as_base_residual", sim[TARGET], sim["E_base"])

    res.attempt("as_base_model", as_base_model)
