r"""
Safety Module & Uncertainty Quantification.

Implements Conformal Prediction methods to statistically guarantee 
confidence intervals.

References:
    - Gibbs & Candès (2021): Adaptive Conformal Inference (ACI)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

class SafetyTAM:
    r"""
    Risk and confidence interval manager.
    
    Offers two operating modes:
    1. Static (Split Conformal): Valid guarantee if data is i.i.d.
    2. Adaptive (ACI): Valid guarantee even under distribution shift (non-stationary).
    """
    
    def __init__(self, alpha: float = 0.1):
        r"""
        Args:
            alpha (float): The target error rate (e.g., 0.1 for 90% coverage).
        """
        self.alpha_target = alpha
        self.residuals_calib_ = None
        
#: <calibrate>
    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Calibrates the model on a 'hold-out' dataset (Calibration Set).
        Calibrates the residuals based on the absolute error abs(y - y_pred).
        
        Args:
            y_true: True values.
            y_pred: Values predicted by the model.
        """
        # Compute non-conformity scores s(x,y) = |y - f(x)| 
        self.residuals_calib_ = np.abs(y_true - y_pred)
        print(f"Safety calibrated on {len(y_true)} samples.")
#: </calibrate>

#: <quantile>
    def _get_quantile(self, current_alpha: float) -> float:
        r"""Computes the empirical quantile (1 - alpha) on calibration residuals."""
        # Finite sample correction: q_level = (1 - alpha) * (1 + 1/n)
        n = len(self.residuals_calib_)
        q_level = np.clip((1 - current_alpha) * (1 + 1/n), 0, 1)
        return np.quantile(self.residuals_calib_, q_level)
#: </quantile>

#: <aci_loop>
    def predict_intervals(
        self, 
        y_pred: np.ndarray, 
        y_true_online: Optional[np.ndarray] = None,
        method: str = 'aci',
        gamma: float = 0.05
    ) -> pd.DataFrame:
        r"""
        Generates confidence intervals [Lower, Upper].
        
        Args:
            y_pred: Future predictions.
            y_true_online: (For ACI only) True values observed sequentially. Required for the feedback loop.
            method: 'static' (Split Conformal) or 'aci' (Adaptive).
            gamma: Learning rate for ACI.
            
        Returns:
            DataFrame with columns ['Lower', 'Upper', 'Alpha_t'].
        """
        if self.residuals_calib_ is None:
            raise RuntimeError("You must call .calibrate() before predicting intervals.")
            
        n_test = len(y_pred)
        widths = np.zeros(n_test)
        alphas = np.zeros(n_test)
        
        # --- Mode 1: Static (Standard Split Conformal) ---
        if method == 'static':
            # Fixed width calculated once and for all
            q_static = self._get_quantile(self.alpha_target)
            widths[:] = q_static
            alphas[:] = self.alpha_target
            
        # --- Mode 2: Adaptive (ACI - Gibbs & Candès) ---
        elif method == 'aci':
            if y_true_online is None:
                raise ValueError("ACI requires 'y_true_online' to update its risk level.")
            
            # Initialization at target risk
            current_alpha = self.alpha_target
            
            for t in range(n_test):
                q_t = self._get_quantile(current_alpha)
                widths[t] = q_t
                alphas[t] = current_alpha
                
                y_t = y_true_online[t]
                y_hat = y_pred[t]
                
                is_covered = (y_hat - q_t <= y_t <= y_hat + q_t)
                
                err_t = 0 if is_covered else 1
                current_alpha = current_alpha + gamma * (self.alpha_target - err_t)
                
                # Safety bounds for alpha (between 0.001 and 0.999)
                current_alpha = np.clip(current_alpha, 0.001, 0.999)
                
        else:
            raise ValueError("Method must be 'static' or 'aci'.")
            
        # Result construction
        df_res = pd.DataFrame({
            'Predicted': y_pred,
            'Lower': y_pred - widths,
            'Upper': y_pred + widths,
            'Alpha_t': alphas,
            'Width': 2 * widths
        })
        
        if y_true_online is not None:
            df_res['Actual'] = y_true_online
            df_res['Covered'] = (df_res['Actual'] >= df_res['Lower']) & (df_res['Actual'] <= df_res['Upper'])
            
        return df_res
#: </aci_loop>