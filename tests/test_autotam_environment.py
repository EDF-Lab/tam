# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The run info records the environment of the run: the same seed gives the same result in the same environment, and a difference between two
environments can be traced to the versions, the threads or the processor.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.seeding import environment_info


def test_the_environment_names_the_versions_the_threads_and_the_processor():
    info = environment_info()
    assert {"python", "platform", "numpy", "pandas", "scipy", "torch", "torch_threads", "cpu_count", "blas"} <= set(info)
    assert info["numpy"] == np.__version__ and info["pandas"] == pd.__version__
    assert isinstance(info["torch_threads"], int) and info["cpu_count"] >= 1


def test_the_run_info_file_carries_the_seed_and_the_environment(tmp_path):
    rng = np.random.default_rng(0)
    n = 90
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": rng.normal(size=n)})
    frame["y"] = 5 + frame["x1"] + rng.normal(scale=0.2, size=n)
    AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=11, export_dir=str(tmp_path)).fit(
        frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    files = list(Path(tmp_path).glob("AutoTAM_run_info_*.json"))
    recorded = json.loads(files[0].read_text(encoding="utf-8"))
    assert recorded["seed"] == 11 and recorded["environment"]["numpy"] == np.__version__
