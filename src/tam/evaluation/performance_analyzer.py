# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Performance Analyzer Module for AutoTAM.

Provides statistical utilities to analyze prediction residuals, evaluate 
model bias, and detect temporal performance degradation over time.
"""

#: <performance_analyzer_imports>
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
#: </performance_analyzer_imports>

#: <performance_analyzer_residuals>
def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Analyzes prediction residuals to check for bias, skewness, and autocorrelation.
    
    Calculates the Lag-1 Autocorrelation as a simplified Durbin-Watson proxy. 
    High autocorrelation indicates the model missed a temporal signal.
    
    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        dict: Dictionary of calculated residual statistics.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: 
        return {}
    
    res = y_true[mask] - y_pred[mask]
    
    if len(res) > 1:
        lag1_corr = np.corrcoef(res[:-1], res[1:])[0, 1]
    else:
        lag1_corr = np.nan
        
    return {
        'Mean Error (Bias)': np.mean(res),
        'Std Error': np.std(res),
        'Skewness': skew(res),
        'Kurtosis': kurtosis(res),
        'Lag-1 AutoCorr': lag1_corr
    }
#: </performance_analyzer_residuals>

#: <performance_analyzer_degradation>
def detect_temporal_degradation(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'RMSE') -> float:
    """
    Splits the test set in half to detect if performance is decaying over time.
    
    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.
        metric (str): The metric to evaluate degradation against. Defaults to RMSE.
        
    Returns:
        float: The degradation ratio percentage. A value of +20.0 means 
               errors grew by 20% in the second half of the data.
    """
    from .metrics import calculate_regression_metrics
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    
    if len(yt) < 4: 
        return np.nan
    
    midpoint = len(yt) // 2
    h1_metrics = calculate_regression_metrics(yt[:midpoint], yp[:midpoint])
    h2_metrics = calculate_regression_metrics(yt[midpoint:], yp[midpoint:])
    
    h1_val = h1_metrics.get(metric, np.nan)
    h2_val = h2_metrics.get(metric, np.nan)
    
    if np.isnan(h1_val) or h1_val == 0: 
        return np.nan
    
    degradation_pct = ((h2_val - h1_val) / h1_val) * 100.0
    return degradation_pct
#: </performance_analyzer_degradation>