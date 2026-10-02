# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Predictions of a grouped model on a frame whose index is not unique.

Such frames come from ``pd.concat`` without ``ignore_index=True``. Rows are ordered by date with the groups interleaved (one row per group
per date, as with ``group_col="tod"``), so a placement by position in group order would put predictions on the wrong rows. The engine must
either raise a clear error or place every prediction on its own row; it must never return wrong predictions silently.
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta


def _interleaved_panel(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Three groups with very different levels (10, 20, 30), one row per group per date, rows ordered by date."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for k, g in enumerate(["a", "b", "c"]):
            x = rng.normal()
            rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), "g": g, "x": x,
                         "y": (k + 1) * 10 + 3 * x + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def _fitted_model_and_reference():
    df = _interleaved_panel()
    model = ta.StaticTAM(formula="y ~ l(x)", group_col="g", date_col="date").fit(df)
    reference = model.predict(df)["Estimatedy"].to_numpy(dtype=float)
    return df, model, reference


def _with_duplicated_index(df: pd.DataFrame) -> pd.DataFrame:
    dup = df.copy()
    dup.index = np.arange(len(dup)) % 50  # same rows, same order, labels repeated
    return dup


def test_duplicate_index_is_never_silently_wrong():
    """Guard: a clear ValueError is acceptable; predictions different from the unique-index ones are not."""
    df, model, reference = _fitted_model_and_reference()
    assert np.max(np.abs(reference - df["y"].to_numpy())) < 1.0  # the model itself is accurate (~0.35)
    try:
        out = model.predict(_with_duplicated_index(df))["Estimatedy"].to_numpy(dtype=float)
    except ValueError:
        return
    np.testing.assert_allclose(out, reference, rtol=0, atol=1e-9)


@pytest.mark.xfail(raises=ValueError, strict=True,
                   reason="the engine does not accept duplicate index labels yet (MAIN-33); remove this marker when it does")
def test_duplicate_index_gives_the_unique_index_predictions():
    """Target behaviour (MAIN-33): same predictions as with a unique index, and the input index returned unchanged."""
    df, model, reference = _fitted_model_and_reference()
    dup = _with_duplicated_index(df)
    result = model.predict(dup)
    np.testing.assert_allclose(result["Estimatedy"].to_numpy(dtype=float), reference, rtol=0, atol=1e-9)
    assert (result.index.to_numpy() == dup.index.to_numpy()).all()
