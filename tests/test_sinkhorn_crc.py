# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import HierarchicalTAM, SafetyTAM, StaticTAM
from tam.model.hierarchical import compute_sinkhorn_coherence_matrix

def test_sinkhorn_hierarchical_and_conformal_risk_control():
    # 1. Doubly stochastic Sinkhorn kernel
    K_sink = compute_sinkhorn_coherence_matrix(32, window=6, eps=0.08, n_iters=25)
    assert K_sink.shape == (32, 32)
    assert torch.allclose(K_sink.sum(dim=1), torch.ones(32, dtype=K_sink.dtype), atol=1e-3)
    assert torch.allclose(K_sink.sum(dim=0), torch.ones(32, dtype=K_sink.dtype), atol=1e-3)

    # 2. HierarchicalTAM with coherence_loss='sinkhorn'
    rng = np.random.default_rng(5)
    n = 60
    x = np.linspace(-1, 1, n)
    c1 = 2.0 * x + 0.05 * rng.standard_normal(n)
    c2 = -0.5 * x + 1.0 + 0.05 * rng.standard_normal(n)
    p_tot = c1 + c2 + 0.02 * rng.standard_normal(n)
    df_hier = pd.concat([
        pd.DataFrame({"Region": "Total", "x": x, "y": p_tot}),
        pd.DataFrame({"Region": "North", "x": x, "y": c1}),
        pd.DataFrame({"Region": "South", "x": x, "y": c2}),
    ], ignore_index=True)

    htam = HierarchicalTAM(
        structure={"Total": ["North", "South"]},
        formulas="y ~ l(x)",
        node_col="Region",
        lambda_p_hier=2.0,
        coherence_loss="sinkhorn",
        sinkhorn_window=4,
    )
    htam.fit(df_hier)
    preds_h = htam.predict(df_hier)
    assert "Estimatedy" in preds_h.columns

    # 3. Conformal Risk Control (CRC) on SafetyTAM and StaticTAM
    y_cal = np.sin(np.linspace(0, 10, 200)) + 0.2 * rng.standard_normal(200)
    y_hat_cal = np.sin(np.linspace(0, 10, 200))
    safety = SafetyTAM(alpha=0.15).calibrate_crc(y_cal, y_hat_cal, w_under=3.0, w_over=1.0)
    crc_df = safety.predict_crc_intervals(y_hat_cal[:20], y_true_online=y_cal[:20])
    assert "Lower" in crc_df.columns and "Upper" in crc_df.columns
    # Asymmetric under-prediction penalty w_under=3.0 > w_over=1.0 produces wider upper half-width than lower half-width
    upper_hw = crc_df["Upper"].to_numpy() - crc_df["Predicted"].to_numpy()
    lower_hw = crc_df["Predicted"].to_numpy() - crc_df["Lower"].to_numpy()
    assert np.allclose(upper_hw, 3.0 * lower_hw)
