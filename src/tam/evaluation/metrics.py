# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Metrics Calculation Module for AutoTAM.

Provides a robust, purely NumPy-based suite of regression and forecasting 
evaluation metrics. It safely handles `NaN` and `Inf` values across 
predictions to prevent scoring failures during evolutionary model selection.
"""

#: <metrics_imports>
import numpy as np
#: </metrics_imports>

#: <calculate_regression_metrics>
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
        
    return metrics
#: </calculate_regression_metrics>