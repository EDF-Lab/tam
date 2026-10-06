# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""What every check shares: the frozen dataset, the split, and the ``Results`` collector that turns what a check computes into lines of text.

A check is a module ``checks/<number>_<functionality>.py`` with ``run(res)``; it fits a few models on the frozen FORCE extract and records numbers
with ``res``. Only the public API of tam and the standard scientific stack are used, so the same checks run on any version of tam.
"""
from __future__ import annotations

import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tam as ta

DATA = Path(__file__).resolve().parent / "data" / "force_2023.csv"
FIT_UNTIL = pd.Timestamp("2023-07-01")          # the sequential checks (adaptive, Kalman, Opera) fit before July and simulate the second half-year
GROUP, DATE, TARGET = "hour", "date", "load"
SIGNIFICANT = 6                                 # digits kept in the results file


def seed() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def frame() -> pd.DataFrame:
    """The frozen dataset with the lagged loads of the same hour (one day and one week before; the first week is dropped) and the month (0 to 11)."""
    df = pd.read_csv(DATA, parse_dates=[DATE])
    df["load_d1"] = df.groupby(GROUP)[TARGET].shift(1)
    df["load_d7"] = df.groupby(GROUP)[TARGET].shift(7)
    df["month"] = df[DATE].dt.month - 1                     # 0 to 11
    return df.dropna(subset=["load_d7"]).reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Static checks: every fifth day is held out (about 70 test days spread over the year and over the weekdays, so no effect is asked to extrapolate)."""
    test = (df[DATE].dt.dayofyear % 5) == 2
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)


def chrono_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential checks: fit on the first half-year, simulate (and forecast) the second."""
    return df[df[DATE] < FIT_UNTIL].reset_index(drop=True), df[df[DATE] >= FIT_UNTIL].reset_index(drop=True)


def text(value) -> str:
    """One line of text for a value: floats to six significant digits, everything else as it is, never a line break."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{SIGNIFICANT}g}" if value != 0 else "0"
    return " ".join(str(value).split())


class Results:
    """The numbers of one check, as ``<check>.<name> = <value>`` lines."""

    def __init__(self, check: str):
        self.check = check
        self.lines: dict[str, str] = {}

    def value(self, name: str, value) -> None:
        self.lines[f"{self.check}.{name}"] = text(value)

    def gap(self, name: str, value, tolerance: float = 1e-6) -> None:
        """A difference that should be zero (two ways of computing the same thing): below the tolerance it is rounding noise, which differs between
        machines, so only "<= tolerance" is recorded; above it, the value itself."""
        value = float(value)
        self.lines[f"{self.check}.{name}"] = f"<= {tolerance:g}" if value <= tolerance else text(value)

    def forecast(self, name: str, truth, prediction) -> None:
        """What a forecast vector is made of: error against the truth, and its distribution (mean, spread, quantiles, missing values)."""
        y, p = np.asarray(truth, dtype=float).ravel(), np.asarray(prediction, dtype=float).ravel()
        ok = np.isfinite(y) & np.isfinite(p)
        self.value(f"{name}.n", len(p))
        self.value(f"{name}.missing", int((~np.isfinite(p)).sum()))
        if ok.any():
            self.value(f"{name}.rmse", float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))))
            self.value(f"{name}.bias", float(np.mean(p[ok] - y[ok])))
            self.value(f"{name}.mean", float(p[ok].mean()))
            self.value(f"{name}.std", float(p[ok].std()))
            for q, label in ((0, "min"), (0.25, "q25"), (0.5, "q50"), (0.75, "q75"), (1, "max")):
                self.value(f"{name}.{label}", float(np.quantile(p[ok], q)))

    def attempt(self, name: str, step) -> None:
        """Runs one step of the check; an exception becomes an ERROR line, so that the results file changes and the other steps still run."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                step()
            except Exception as error:                      # noqa: BLE001 - recorded, not hidden
                self.lines[f"{self.check}.{name}"] = text(f"ERROR {type(error).__name__}: {str(error)[:90]}")

    def static(self, name: str, formula: str, *, auto: bool = False, fit_args: dict | None = None, **model_args) -> "ta.StaticTAM":
        """Fits ``StaticTAM(formula)`` on the training months, records its forecast of the test months, returns the model."""
        train, test = self._split
        built: dict = {}

        def step():
            model = ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE, **model_args)
            model.auto_fit(train, **(fit_args or {})) if auto else model.fit(train)
            built["model"] = model
            self.forecast(name, test[TARGET], model.predict(test)[f"Estimated{TARGET}"])

        self.attempt(name, step)
        return built.get("model")

    @property
    def _split(self):
        if not hasattr(self, "_cached"):
            self._cached = split(frame())
        return self._cached

    @property
    def train(self) -> pd.DataFrame:
        return self._split[0]

    @property
    def test(self) -> pd.DataFrame:
        return self._split[1]


# the formula every effect check builds on: yesterday's load, the weekday and the bank-holiday indicators
BASE = "l(load_d1, ap=-30) + c(day_type_week, n_cat=7, topo='nominal', ap=-5) + c(day_type_jf, n_cat=2, topo='nominal', ap=-5)"


# ------------------------------------------------------------------ the sequential checks: a base model fitted on the first half-year, a simulation over the whole year
def base_model(res, formula: str | None = None):
    """A StaticTAM fitted on the first half-year (the base of the adaptive, Kalman and Opera checks). Its formula has no effect that extrapolates (a cyclic
    Fourier term for the day of the year, a linear temperature): these checks are about the loops, the extrapolation of splines has its own check."""
    first, _ = chrono_split(frame())
    formula = formula or "load ~ f(toy, m=6, s=1, cyclic=True) + l(temperature) + l(load_d7) + " + BASE
    return ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE).fit(first)


def with_residual(df: pd.DataFrame, model, column: str = "L_Res") -> pd.DataFrame:
    """The frame with the forecast of ``model`` and the residual of the same hour the day before (what an operator knows at forecast time)."""
    out = df.copy()
    out["E_base"] = model.predict(out)[f"Estimated{TARGET}"].to_numpy()
    out[column] = (out[TARGET] - out["E_base"]).groupby(out[GROUP]).shift(1)
    return out.dropna(subset=[column]).reset_index(drop=True)


def second_half(frame_with_forecast: pd.DataFrame, column: str):
    """(truth, forecast) over the second half-year, where the simulation forecasts."""
    part = frame_with_forecast[frame_with_forecast[DATE] >= FIT_UNTIL]
    return part[TARGET], part[column]
