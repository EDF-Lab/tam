r"""
Unit tests for ``tam.model.diagnostics``, statistical inference on a fitted
StaticTAM (effective DoF, noise variance, t-statistics, and bootstrap).

Plotting is suppressed by monkeypatching ``matplotlib.pyplot.show``.
"""

import numpy as np
import torch
import pytest

import tam as ta
import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.diagnostics import (
    _compute_p_matrix,
    _estimate_noise_variance,
    _compute_t_statistics,
    run_diagnostics,
    compute_bootstrap_significance,
)


# ------------------------------ pure tensor math --------------------------- #

def _phi_system(n_groups=2, n_samples=25, n_coeffs=4, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.rand(n_groups, n_samples, n_coeffs, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    beta = torch.rand(n_groups, n_coeffs, 1, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    y = phi @ beta + 0.05 * torch.rand(n_groups, n_samples, 1, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    return phi, beta, y


def test_compute_p_matrix_shape():
    phi, _, _ = _phi_system()
    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    P = _compute_p_matrix(phi, penalty, n_samples=phi.shape[1])
    # Maps Y (n_samples) -> coefficients (n_coeffs): shape (groups, n_coeffs, n_samples).
    assert P.shape == (2, 4, 25)


def test_estimate_noise_variance_positive():
    phi, beta, y = _phi_system()
    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    n = phi.shape[1]
    P = _compute_p_matrix(phi, penalty, n)
    preds = phi @ beta
    sigma2 = _estimate_noise_variance(phi, P, y, preds, n)
    assert sigma2.shape == (2,)
    assert (sigma2 >= 0).all()


def test_compute_t_statistics_shape_and_finiteness():
    phi, beta, y = _phi_system()
    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    n = phi.shape[1]
    P = _compute_p_matrix(phi, penalty, n)
    sigma2 = _estimate_noise_variance(phi, P, y, phi @ beta, n)
    t_stats = _compute_t_statistics(P, sigma2, beta)
    assert t_stats.shape == (2, 4)
    assert torch.isfinite(t_stats).all()


# ------------------------------ end-to-end pipeline ------------------------ #

def test_run_diagnostics_returns_t_stats(dummy_panel_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    model = ta.StaticTAM(
        formula="load ~ s(temperature, k=5) + l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
    )
    model.fit(dummy_panel_data)

    t_stats = run_diagnostics(model, dummy_panel_data)
    total_coeffs = sum(e.get_n_coeffs() for e in model.effects_list_)
    assert t_stats.shape[0] == len(model.unique_groups_)
    assert t_stats.shape[1] == total_coeffs
    assert torch.isfinite(t_stats).all()


def test_run_diagnostics_unfitted_raises():
    model = ta.StaticTAM(formula="load ~ l(temperature)")
    with pytest.raises(RuntimeError, match="fitted"):
        run_diagnostics(model, None)


def test_bootstrap_significance_returns_t_stats(dummy_panel_data):
    # The block bootstrap solver is formulated for a single group, so restrict
    # the data to one meter (phi becomes a single-group batch). A real group
    # column is kept so the solver's _prepare_data call finds it.
    single = dummy_panel_data[dummy_panel_data["smart_meter_id"] == "Meter_A"].reset_index(drop=True)
    model = ta.StaticTAM(
        formula="load ~ l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
    )
    model.fit(single)

    t_robust = compute_bootstrap_significance(model, single, n_boot=10, block_size=20)
    assert np.all(np.isfinite(t_robust))
