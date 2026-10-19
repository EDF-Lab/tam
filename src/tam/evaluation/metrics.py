# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Metrics Calculation Module for AutoTAM.

Provides a robust, purely NumPy-based suite of regression and forecasting 
evaluation metrics. It safely handles `NaN` and `Inf` values across 
predictions to prevent scoring failures during evolutionary model selection.
"""

#: <metrics_imports>
from typing import Optional

import numpy as np
#: </metrics_imports>

_TINY = 1e-12
FAMILIES = ("gaussian", "poisson", "gamma", "binomial", "negative_binomial", "tweedie")

#: <deviance_scores>
def unit_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: str = "gaussian",
    power: float = 1.5,
    dispersion: float = 0.1,
) -> np.ndarray:
    """
    Unit deviance d(y, mu) of an exponential dispersion family, per observation.

    ``d`` is zero at ``mu = y`` and positive elsewhere; for the Gaussian family it is the squared error. The means
    ``y_pred`` are clipped away from the edge of the family's domain, so a prediction outside it (a negative count)
    costs a very large deviance instead of a NaN.

    Args:
        y_true: Observations.
        y_pred: Predicted means (response scale).
        family: ``gaussian``, ``poisson``, ``gamma``, ``binomial`` (targets in [0, 1]), ``negative_binomial`` or ``tweedie``.
        power: Variance power p of the Tweedie family, 1 < p < 2.
        dispersion: Dispersion alpha of the negative binomial family, Var = mu + alpha mu^2.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(y_pred, dtype=float)
    family = str(family).lower()
    if family == "gaussian":
        return (y - mu) ** 2
    if family == "poisson":
        mu = np.clip(mu, _TINY, None)
        return 2.0 * (y * np.log(np.clip(y, _TINY, None) / mu) - (y - mu))
    if family == "gamma":
        mu = np.clip(mu, _TINY, None)
        return 2.0 * (-np.log(np.clip(y, _TINY, None) / mu) + (y - mu) / mu)
    if family == "binomial":
        mu = np.clip(mu, _TINY, 1.0 - _TINY)
        return 2.0 * (y * np.log(np.clip(y, _TINY, None) / mu) + (1.0 - y) * np.log(np.clip(1.0 - y, _TINY, None) / (1.0 - mu)))
    if family == "negative_binomial":
        if not dispersion > 0:
            raise ValueError(f"dispersion must be > 0; got {dispersion!r}")
        mu = np.clip(mu, _TINY, None)
        return 2.0 * (y * np.log(np.clip(y, _TINY, None) / mu) - (y + 1.0 / dispersion) * np.log((1.0 + dispersion * y) / (1.0 + dispersion * mu)))
    if family == "tweedie":
        if not 1.0 < power < 2.0:
            raise ValueError(f"power must be in (1, 2); got {power!r}")
        mu = np.clip(mu, _TINY, None)
        y_pos = np.clip(y, 0.0, None)
        return 2.0 * (y_pos ** (2.0 - power) / ((1.0 - power) * (2.0 - power)) - y * mu ** (1.0 - power) / (1.0 - power) + mu ** (2.0 - power) / (2.0 - power))
    raise ValueError(f"Unknown family '{family}'. Use one of {FAMILIES}.")


def deviance_explained(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: str = "gaussian",
    y_null: Optional[float] = None,
    power: float = 1.5,
    dispersion: float = 0.1,
) -> float:
    """
    Proportion of deviance explained, ``D2 = 1 - sum d(y, mu) / sum d(y, y_null)``.

    ``y_null`` is the intercept-only prediction (the mean of ``y_true`` when None). For the Gaussian family ``D2`` is
    the ordinary R2; for the other families it is the likelihood-ratio pseudo-R2 that ``mgcv`` reports as the
    deviance explained.
    """
    y = np.asarray(y_true, dtype=float)
    null = float(np.mean(y)) if y_null is None else float(y_null)
    residual = float(np.sum(unit_deviance(y, y_pred, family, power, dispersion)))
    total = float(np.sum(unit_deviance(y, np.full_like(y, null), family, power, dispersion)))
    return 1.0 - residual / total if total > 0 else float("nan")


_DEVIANCE_METRICS = {
    "gaussian_deviance": "gaussian", "poisson_deviance": "poisson", "gamma_deviance": "gamma",
    "binomial_deviance": "binomial", "logloss": "binomial", "log_loss": "binomial",
    "negative_binomial_deviance": "negative_binomial", "tweedie_deviance": "tweedie",
}


def deviance_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """
    Mean unit deviance for a metric name of the model-selection code (``poisson_deviance``, ``gamma_deviance``,
    ``binomial_deviance`` or ``logloss``, ``negative_binomial_deviance``, ``tweedie_deviance``); None for any other name.
    ``logloss`` is half the binomial deviance, the average negative log-likelihood.
    """
    key = str(metric).lower()
    if key not in _DEVIANCE_METRICS:
        return None
    mean_deviance = float(np.mean(unit_deviance(y_true, y_pred, _DEVIANCE_METRICS[key])))
    return mean_deviance / 2.0 if key in ("logloss", "log_loss") else mean_deviance
#: </deviance_scores>

#: <calculate_regression_metrics>
def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: Optional[str] = None,
    power: float = 1.5,
    dispersion: float = 0.1,
) -> dict:
    """
    Computes a comprehensive suite of regression and forecasting metrics.
    
    Args:
        y_true (np.ndarray): The ground truth target values.
        y_pred (np.ndarray): The predicted values from the model.
        
    Returns:
        dict: A dictionary containing RMSE, MAE, NMAE, SMAPE, R2, and MAPE.
    """
    if y_pred is None or len(y_pred) == 0: 
        return {}
        
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)
    if mask.sum() == 0: 
        return {}
    
    yt, yp = y_true[mask], y_pred[mask]
    
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    smape = np.mean(2.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100.0
    nmae = mae / (np.mean(np.abs(yt)) + 1e-8)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'NMAE': nmae,
        'SMAPE': smape,
        'R2': r2
    }
    
    if not np.any(yt == 0):
        metrics['MAPE'] = np.mean(np.abs((yt - yp) / yt)) * 100.0
    else:
        metrics['MAPE'] = np.nan
        
    if family is not None:
        metrics['Deviance'] = float(np.mean(unit_deviance(yt, yp, family, power, dispersion)))
        metrics['D2'] = deviance_explained(yt, yp, family, power=power, dispersion=dispersion)

    return metrics
#: </calculate_regression_metrics>