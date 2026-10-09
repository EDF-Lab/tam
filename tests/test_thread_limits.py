# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Limiting the CPU threads. The supported way is the environment (``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``, set before Python starts).

Some PyTorch CPU builds (seen with torch 2.14.0+cpu on Windows, MKL 2026.1) make any batched LU factorisation of a matrix of a few hundred
rows (``torch.linalg.solve``, ``lu_factor``, ``inv``) print ``Intel oneMKL ERROR: Parameter 6 was incorrect on entry to DLASWP`` and
hang once ``torch.set_num_threads(n)`` has been called with ``n >= 2``, whatever ``n`` and without importing ``tam``. The test runs a
grouped fit large enough to need such a factorisation in a fresh process with the threads limited through the environment: it finishes,
uses the requested number of threads, prints no MKL error and takes about as long as without the limit.
"""
import os
import subprocess
import sys
import time

import pytest

_SCRIPT = r"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import tam as ta
rng = np.random.default_rng(0)
days, groups = 400, 24
n = days * groups
df = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-01", periods=days, freq="D"), groups), "h": np.tile(np.arange(groups), days),
                   "x": rng.normal(size=n), "z": rng.normal(size=n)})
df["y"] = np.sin(df.x) + 0.5 * df.z + 0.1 * rng.normal(size=n)
start = time.time()
ta.StaticTAM("y ~ s(x, k=30) + s(z, k=30) + te(s(x, k=10), s(z, k=10))", group_col="h", date_col="date").fit(df)
print("THREADS", torch.get_num_threads(), "SECONDS", round(time.time() - start, 3))
"""


def _run(threads=None, timeout=240):
    env = {key: value for key, value in os.environ.items() if key not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if threads is not None:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    started = time.time()
    done = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=timeout, env=env)
    return done, time.time() - started


def _field(done, name):
    return float(done.stdout.split(name)[1].split()[0])


def test_the_environment_limits_the_threads_and_the_fit_finishes_without_mkl_errors():
    free, free_seconds = _run()
    assert free.returncode == 0, free.stderr[-500:]
    limited, limited_seconds = _run(threads=2)
    assert limited.returncode == 0, limited.stderr[-500:]
    assert _field(limited, "THREADS") == 2
    assert "DLASWP" not in limited.stdout + limited.stderr
    assert limited_seconds < 10 * max(free_seconds, 5.0)
