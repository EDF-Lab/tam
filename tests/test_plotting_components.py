# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""Plotting after the decomposition naming fix: resolving a feature to its component, and plot_component views."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.common.plotting import (
    plot_component,
    plot_effect_with_data_decomposed,
    plot_effect_with_model_and_data,
    resolve_component,
)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n),
                       "x8": rng.uniform(0, 1, n), "x5": rng.normal(size=n), "c": rng.integers(0, 4, n).astype(float)})
    df["y"] = df.x1 * np.sin(6 * df.x8) + df.x1 * df.x2 + df.x5 + 0.5 * df.c + rng.normal(scale=0.1, size=n)
    return df


def fit(formula, df):
    return ta.StaticTAM(formula=formula).fit(df)


def test_unique_feature_resolves_to_itself(data):
    m = fit("y ~ l(x5) + s(x2, k=5)", data)
    assert resolve_component(m, "x5") == "x5"
    plot_effect_with_model_and_data(m, data, effect="x5")


def test_feature_inside_two_te_is_resolved_by_color_by(data):
    # The frozen v1.3.0 cheatsheet call: te(l(x1), f(x8)) + te(l(x1), l(x2)), effect='x1', color_by='x8'.
    m = fit("y ~ te(l(x1), f(x8, m=3)) + te(l(x1), l(x2)) + l(x5)", data)
    assert resolve_component(m, "x1", color_by="x8") == "te_x1_x_x8"
    assert resolve_component(m, "x1", color_by="x2") == "te_x1_x_x2"
    plot_effect_with_model_and_data(m, data, effect="x1", color_by="x8")


def test_ambiguous_feature_raises_with_the_candidates(data):
    m = fit("y ~ l(x5) + s(x5, k=5) + l(x2)", data)
    with pytest.raises(ValueError, match=r"l_x5.*s_x5|s_x5.*l_x5"):
        resolve_component(m, "x5")
    assert resolve_component(m, "x5", component="s_x5") == "s_x5"
    plot_effect_with_model_and_data(m, data, effect="x5", component="s_x5")


def test_unknown_feature_or_component_raises(data):
    m = fit("y ~ l(x5)", data)
    with pytest.raises(ValueError, match="x2"):
        resolve_component(m, "x2")
    with pytest.raises(ValueError, match="nope"):
        resolve_component(m, "x5", component="nope")


def test_decomposed_frame_resolution(data):
    m = fit("y ~ te(l(x1), f(x8, m=3)) + l(x5) + s(x5, k=5)", data)
    frame = m.decompose_prediction(data)
    plot_effect_with_data_decomposed(frame, "x1")            # only inside one te()
    with pytest.raises(ValueError, match="l_x5"):
        plot_effect_with_data_decomposed(frame, "x5")        # two components
    plot_effect_with_data_decomposed(frame, "x5", component="s_x5")


def test_plot_component_skips_non_finite_rows(data, monkeypatch):
    surface = fit("y ~ te(l(x1), l(x2))", data)
    original = surface.decompose_prediction

    def with_gaps(frame):
        out = original(frame)
        out.loc[out.index[:5], "effect_te_x1_x_x2"] = np.nan
        return out

    monkeypatch.setattr(surface, "decompose_prediction", with_gaps)
    for kind in ("auto", "heatmap"):
        assert plot_component(surface, data, "te_x1_x_x2", kind=kind) is not None


def test_surface_is_evaluated_on_a_grid_for_one_group(data):
    grouped = data.assign(g=(np.arange(len(data)) % 3).astype(float), date=pd.date_range("2020-01-01", periods=len(data), freq="h"))
    model = ta.StaticTAM(formula="y ~ te(l(x1), l(x2))", group_col="g", date_col="date").fit(grouped)
    fig = plot_component(model, grouped, "te_x1_x_x2", group=1.0, resolution=25)
    fig.canvas.draw()
    surface = fig.axes[0].collections[0]
    assert fig.axes[0].name == "3d" and "g = 1.0" in fig.axes[0].get_title()
    assert len(surface.get_facecolors()) == 24 * 24  # one quad per grid cell
    heat = plot_component(model, grouped, "te_x1_x_x2", kind="heatmap", group=1.0, resolution=25)
    assert heat.axes[0].name != "3d"


def test_plot_component_views(data):
    one = fit("y ~ s(x5, k=5) + l(x2)", data)
    assert len(plot_component(one, data, "x5").axes) >= 1

    by_level = fit("y ~ te(l(x1), c(c, n_cat=4, topo='nominal'))", data)
    fig = plot_component(by_level, data, "te_x1_x_c")
    assert fig.axes[0].name != "3d" and len(fig.axes[0].get_legend().get_texts()) == 4

    surface = fit("y ~ te(l(x1), l(x2))", data)
    assert plot_component(surface, data, "te_x1_x_x2").axes[0].name == "3d"
    assert plot_component(surface, data, "te_x1_x_x2", kind="heatmap").axes[0].name != "3d"

    three = fit("y ~ te(l(x1), l(x2), l(x3))", data)
    assert plot_component(three, data, "te_x1_x_x2_x_x3").axes[0].name == "3d"
