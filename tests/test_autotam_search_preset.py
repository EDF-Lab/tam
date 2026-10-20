# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The ``search="fast"`` preset: a bundle of values of existing settings; ``search="full"`` is the search as it was.
"""

import time

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam import search_preset


def _frame(n=140, freq="D"):
    rng = np.random.default_rng(4)
    t = np.arange(n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 15 + 3 * np.sin(2 * np.pi * t / 7) + 2 * x1 + x2 ** 2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq=freq), "x1": x1, "x2": x2, "y": y})


def test_the_default_is_the_full_search_with_the_same_settings_as_before(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", export_dir=str(tmp_path))
    assert model.search == "full" and model.n_experts == 15 and model.discoverer.pop_size == 64
    assert model.discoverer.islands is None and model.discoverer.n_generations == search_preset.FULL_GENERATIONS
    assert model.seasonal_anchor is False


def test_the_fast_preset_sets_the_bundle(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", search="fast", export_dir=str(tmp_path))
    assert model.discoverer.islands == list(search_preset.FAST_ISLANDS)
    assert model.discoverer.pop_size == search_preset.FAST_POP_SIZE and model.n_experts == search_preset.FAST_N_EXPERTS
    assert model.discoverer.n_generations == search_preset.FAST_GENERATIONS and model.seasonal_anchor is True


def test_an_argument_given_wins_over_the_preset(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", search="fast", pop_size=8, n_experts=3, seasonal_anchor=False, export_dir=str(tmp_path))
    assert model.discoverer.pop_size == 8 and model.n_experts == 3 and model.seasonal_anchor is False


def test_an_unknown_search_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="search must be one of"):
        AutoTAM("y ~ AutoPipe(x1)", search="quick", export_dir=str(tmp_path))


@pytest.fixture(scope="module")
def fast_fit(tmp_path_factory):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", search="fast", use_opera=False, seed=2, export_dir=str(tmp_path_factory.mktemp("fast")))
    started = time.perf_counter()
    model.fit(_frame(), date_col="ds")
    return model, time.perf_counter() - started


def test_a_fast_search_builds_no_neural_rbf_tree_or_wavelet_term(fast_fit):
    model, _ = fast_fit
    formulas = " ".join(f for fs in model.architectures_.values() for f in fs)
    assert not any(f"{effect}(" in formulas for effect in ("n", "rbf", "t", "lt", "w"))
    assert not {"NeuralIsland", "RBFIsland", "TreeIsland", "WaveletIsland", "Continent", "SmallContinent"} & set(model.architectures_)


def test_a_fast_search_has_the_anchor_and_few_dynamic_variants(fast_fit):
    model, _ = fast_fit
    assert any("Anchor" in name for name in (e["name"] for e in model.trained_experts)) or model.ctx.anchor_season
    assert len(model.ctx.candidate_windows) <= search_preset.FAST_ADAPTIVE_WINDOW_COUNT
    assert model.ctx.fast_search is True


def test_a_step_without_a_natural_season_means_no_anchor_in_the_preset_not_an_error(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", search="fast", use_opera=False, seed=2, export_dir=str(tmp_path))
    frame = _frame(140).assign(ds=lambda d: pd.date_range("2023-01-01", periods=len(d), freq="3D"))
    model.fit(frame, date_col="ds")
    assert model.ctx.anchor_season is None and model.trained_experts


def test_a_step_without_a_natural_season_still_raises_when_the_anchor_is_asked_for(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", search="fast", seasonal_anchor=True, export_dir=str(tmp_path))
    frame = _frame(140).assign(ds=lambda d: pd.date_range("2023-01-01", periods=len(d), freq="3D"))
    with pytest.raises(ValueError, match="natural season"):
        model.fit(frame, date_col="ds")
