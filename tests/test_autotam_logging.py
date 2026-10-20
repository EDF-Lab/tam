# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The warnings and the progress messages of the AutoTAM package go through ``logging``: a user can filter, silence or collect them.
"""

import pathlib
import re

import numpy as np
import pandas as pd

import tam.model.autotam as package
from tam.model.autotam.auto_tam import AutoTAM


def _sources():
    return sorted(pathlib.Path(package.__file__).parent.rglob("*.py"))


def test_no_warning_or_progress_message_is_printed_any_more():
    pattern = re.compile(r"""print\(\s*f?["'](Warning|Error|AutoTAM: |BaseDiscoverer: |ExpertExpander: |EnsembleSelector: |Apex|AutoTAM Parser Warning|Skipping the AutoTAM|No evaluation logs)""")
    offenders = [f"{path.name}: {m.group(0)}" for path in _sources() for m in pattern.finditer(path.read_text(encoding="utf-8"))]
    assert not offenders, offenders


def test_a_search_logs_its_progress_and_prints_nothing_but_the_reports(tmp_path, caplog, capsys):
    rng = np.random.default_rng(0)
    n = 90
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    df["y"] = 5 + 2 * df["x1"] + rng.normal(scale=0.2, size=n)
    capsys.readouterr()
    with caplog.at_level("INFO"):
        AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=1, export_dir=str(tmp_path)).fit(
            df, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert "Starting Pipeline" in caplog.text and "Starting Evolutionary Search" in caplog.text and "Extracted" in caplog.text
    assert "Starting Pipeline" not in capsys.readouterr().out
