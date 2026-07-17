# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Implements the Kalman TAM (KalmanTAM) Meta-Learner.

This module applies a Dynamic Extended Kalman Filter (EKF) to track the 
drifting linear coefficients of a pre-trained base model's outputs (such as 
spatial wavelets or sliding-window residuals) over time.

Architectural Highlights:
    1. Residual Tracking: The filter initializes at zero and tracks the residual 
       of the base model, ensuring predictions default safely to the base physics 
       if no drift is present.
    2. GPU Saturation via Woodbury & TorchScript: The algorithm upgrades standard 
       recursion to a Rank-B Block update using the Woodbury Matrix Identity, compiled 
       via TorchScript to eliminate Python loop overhead and maximize GPU utilization.
    3. Asymmetric Process Noise (Q): Injects targeted process noise, allowing the 
       global offset to adapt rapidly to concept drift while protecting the structural 
       integrity of the base model's physical features.
    4. Systematic Offset & Scaling: The target space is group-normalized to stabilize
       noise matrices (Q, R), and a constant offset is systematically appended to the 
       design matrix to track global bias drift.

References:
    - [Statistical Theory] de Vilmarest, J., Wintenberger, O. (2020). Stochastic 
      Online Optimization using Kalman Recursion. Journal of Machine Learning Research.
    - [Linear Algebra] Hager, W. W. (1989). Updating the inverse of a matrix. SIAM review.
    - [Adaptive Filtering] Sayed, A. H. (2008). Adaptive Filters. John Wiley & Sons.
"""

import torch
import pandas as pd
import numpy as np
import itertools
import warnings
from typing import Tuple, Dict, Any, Optional

from tam.common.utils import (
    TORCH_DEVICE, _balance_groups, 
    _ensure_dummies, _cleanup_dummies
)
from .additive import StaticTAM
from ._data import _reassemble_predictions

#: <jit_woodbury_block>
@torch.jit.script
def _kalman_block_loop_optimized(
    phi_matrix: torch.Tensor,
    y_stacked: torch.Tensor,
    base_pred_stacked: torch.Tensor,
    B: int,
    P_init_diag: float,
    observation_noise_var: float,
    process_noise_var: float,
    eps: float,
    offset_boost: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compiled TorchScript loop for the Block Kalman Filter.
    Executes the Predict and Update phases across all groups simultaneously 
    while keeping the temporal recursion entirely within the compiled C++ backend.
    """
    G, N, d = phi_matrix.shape
    dtype = phi_matrix.dtype
    device = phi_matrix.device
    
    # Initialize State, Covariance, and Process Noise.
    theta_t = torch.zeros((G, d, 1), device=device, dtype=dtype)
    P_t = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * P_init_diag
    Q_matrix = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * process_noise_var
    
    # Asymmetric Tracking: Boost the variance of the offset (index 0) 
    # to allow rapid bias correction without destroying the base physics.
    if d > 0:
        P_t[:, 0, 0] = P_t[:, 0, 0] * offset_boost
        Q_matrix[:, 0, 0] = Q_matrix[:, 0, 0] * offset_boost
    
    predictions = torch.zeros((G, N, 1), device=device, dtype=dtype)
    num_blocks = (N + B - 1) // B
    state_history_gpu = torch.zeros((num_blocks, G, d, 1), device=device, dtype=dtype)
    
    # Compiled sequential block loop
    for i in range(num_blocks):
        t = i * B
        curr_B = B if t + B <= N else N - t
        
        X_B = phi_matrix[:, t : t + curr_B, :]
        Y_B = y_stacked[:, t : t + curr_B, :]
        Y_base_B = base_pred_stacked[:, t : t + curr_B, :]
        
        # --- PREDICT PHASE (A-Priori) ---
        delta_B = torch.bmm(X_B, theta_t)
        y_hat_B = Y_base_B + delta_B 
        predictions[:, t : t + curr_B, :] = y_hat_B
        state_history_gpu[i] = theta_t 
        
        # --- UPDATE PHASE (A-Posteriori) ---
        innovations = Y_B - y_hat_B
        
        # Resilience: Skip updates for missing target values to prevent state corruption.
        nan_mask = torch.isnan(innovations)
        innovations = torch.where(nan_mask, torch.zeros_like(innovations), innovations)
        
        # Calculate Innovation Covariance.
        R_B = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * observation_noise_var
        P_X_T = torch.bmm(P_t, X_B.transpose(-2, -1))
        S = R_B + torch.bmm(X_B, P_X_T)
        
        # Numerical Stability: Ensure matrix is positive-definite before Cholesky decomposition.
        jitter = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * eps
        L = torch.linalg.cholesky(S + jitter)
        S_inv = torch.cholesky_inverse(L)
        
        # Update State via Kalman Gain.
        K = torch.bmm(P_X_T, S_inv)
        theta_t = theta_t + torch.bmm(K, innovations)
        
        # Update Covariance and inject Process Noise (Random Walk).
        P_t = P_t - torch.bmm(K, torch.bmm(X_B, P_t))
        P_t = P_t + (Q_matrix * float(curr_B))
        P_t = (P_t + P_t.transpose(-2, -1)) * 0.5 # Maintain Symmetry.

    return predictions, state_history_gpu
#: </jit_woodbury_block>

class KalmanTAM:
    r"""
    A Meta-Learner that tracks drifting physical effects using a Block Kalman Filter.
    Optimized for GPU batch processing, TorchScript compilation, and missing-data resilience.
    Automatically standardizes the target space to stabilize hyperparameters.
    """
    
    def __init__(
        self,
        kalman_formula: str,
        base_model: Optional[Any] = None,
        group_col: Optional[str] = None,
        date_col: Optional[str] = None,
        use_decomposition: bool = True,
        block_size: int = 128,
        horizon_steps: int = 1,
        default_alpha_p: float = -9.0,
        eps: float = 1e-6,
        offset_boost: float = 100.0,
        process_noise_var: float = 1e-4,
        observation_noise_var: float = 1.0,
        P_init_diag: float = 1.0,
        add_base_effects: bool = False
    ):
        """
        Initializes the Kalman Tracker.

        Args:
            kalman_formula: Formula defining which effects to track.
            base_model: A fitted StaticTAM or AdaptiveTAM model.
            group_col: Optional column name for grouping data.
            date_col: Optional column name for time indexing.
            use_decomposition: If True, extracts components from the base model.
            block_size: The chunk size (B) for Woodbury matrix inversion.
            horizon_steps: Number of forward steps to delay the state application, preventing target leakage in multi-step forecasting. Default is 1 (standard online filtering).
            default_alpha_p: Default regularization parameter for feature scaling.
            eps: Small constant added to the diagonal for numerical stability.
            offset_boost: Multiplier for the offset's process noise to accelerate bias correction.
        """
        if base_model is not None and getattr(base_model, 'coefficients_', None) is None:
            raise ValueError("The base_model must be fitted before initializing KalmanTAM.")
        
        if add_base_effects and base_model is not None:
            for effect in base_model.effects_list_:
                effect_col = f"effect_{effect.feature_name}"
                if effect_col not in kalman_formula:
                    kalman_formula += f" + l({effect_col})"
            use_decomposition = True

        self.base_model_ = base_model
        self.kalman_formula_ = kalman_formula
        
        # Inherit from base_model if available, otherwise apply transitive dummies
        self.group_col_ = group_col or getattr(base_model, 'group_col_', "__dummy_group__")
        self.date_col_ = date_col or getattr(base_model, 'date_col_', "__dummy_date__")
        
        self.use_decomposition_ = use_decomposition if base_model is not None else False
        self.block_size_ = block_size
        self.horizon_steps_ = horizon_steps
        self.eps = eps 
        self.offset_boost = offset_boost
        self.process_noise_var_ = process_noise_var
        self.observation_noise_var_ = observation_noise_var
        self.P_init_diag_ = P_init_diag

        # Internal feature extractor built with the provided group_col logic.
        self.feature_extractor_ = StaticTAM(
            formula=kalman_formula,
            group_col=self.group_col_,
            date_col=self.date_col_,
            default_alpha_p=default_alpha_p
        )
        
        self.states_history_ = None 
        self.target_col_ = self.feature_extractor_.target_col_

        if self.base_model_ is not None:
            base_target = getattr(self.base_model_, 'target_col_', None)
            if base_target and base_target != self.target_col_:
                warnings.warn(
                    f"Kalman target '{self.target_col_}' does not match base model target '{base_target}'. "
                    "This will cause a cross-target correction, which is usually unintended.",
                    UserWarning
                )

    def _prepare_kalman_features(
        self, 
        data: pd.DataFrame,
        is_inference: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, list, pd.DataFrame]:
        r"""
        Extracts tracking features and the base model's full prediction.
        Builds the normalized Design Matrix required for Residual Tracking,
        and systematically injects a constant offset tracker.
        """
        if self.base_model_ is not None:
            # Get the full base prediction to compute residuals.
            df_base = self.base_model_.predict(data)
            
            # The StaticTAM base model cleans dummy columns upon returning.
            # Re-inject the dummy columns for Kalman's internal sequential processing.
            df_base = _ensure_dummies(df_base, self.group_col_, self.date_col_)
            
            base_pred_col = f"Estimated{self.base_model_.target_col_}"
            
            if base_pred_col not in df_base.columns:
                 base_pred_col = "EstimatedY" 
                 
            # Extract specific features/wavelets to track.
            if self.use_decomposition_:
                df_features = self.base_model_.decompose_prediction(df_base)
                df_features = _ensure_dummies(df_features, self.group_col_, self.date_col_)
            else:
                df_features = df_base.copy()
        else:
            # Bypass base model entirely, treating base prediction as strictly 0
            df_features = data.copy()
            base_pred_col = "__dummy_base_pred__"
            df_features[base_pred_col] = 0.0
            
        # Build the normalized Kalman design matrix and extract targets.
        target_col_to_use = None if is_inference else self.target_col_
        x_stacked, y_stacked, unique_groups = self.feature_extractor_._prepare_data(
            df_features, target_col=target_col_to_use
        )
        
        # Extract base predictions for GPU arithmetic.
        _, base_pred_stacked, _ = self.feature_extractor_._prepare_data(
            df_features, target_col=base_pred_col
        )
        
        phi_matrix = self.feature_extractor_._build_design_matrix(x_stacked)
        
        # Systematic Offset: Prepend a column of ones to force global bias tracking.
        G, N, d = phi_matrix.shape
        offset_tensor = torch.ones((G, N, 1), device=phi_matrix.device, dtype=phi_matrix.dtype)
        phi_matrix_with_offset = torch.cat([offset_tensor, phi_matrix], dim=-1)
        
        return phi_matrix_with_offset, y_stacked, base_pred_stacked, unique_groups, df_features

    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Balances groups, prepares tensors, and normalizes the target space 
        to stabilize Kalman process matrices. Compatible with older PyTorch versions.
        """
        # Inject dummies before balancing to maintain chronologies
        data = _ensure_dummies(data, self.group_col_, self.date_col_)
        
        mask, balanced_data = _balance_groups(
            dataset=data, 
            group_col=self.group_col_, 
            date_col=self.date_col_, 
            method="fill"
        )
        
        phi_matrix, y_stacked, base_pred_stacked, unique_groups, df_original = self._prepare_kalman_features(balanced_data)
        
        y_stacked = y_stacked.to(TORCH_DEVICE)
        base_pred_stacked = base_pred_stacked.to(TORCH_DEVICE)
        
        # Replace NaNs with limits to safely ignore missing data
        y_for_max = torch.where(torch.isnan(y_stacked), torch.tensor(-float('inf'), device=TORCH_DEVICE, dtype=y_stacked.dtype), y_stacked)
        y_for_min = torch.where(torch.isnan(y_stacked), torch.tensor(float('inf'), device=TORCH_DEVICE, dtype=y_stacked.dtype), y_stacked)
        
        y_max = torch.max(y_for_max, dim=1, keepdim=True)[0]
        y_min = torch.min(y_for_min, dim=1, keepdim=True)[0]
        
        # Fallback for completely empty or constant groups
        y_max = torch.where(torch.isinf(y_max), torch.zeros_like(y_max), y_max)
        y_min = torch.where(torch.isinf(y_min), torch.zeros_like(y_min), y_min)
        
        amplitude = y_max - y_min
        amplitude = torch.where(amplitude == 0.0, torch.ones_like(amplitude), amplitude)
        
        center = (y_max + y_min) / 2.0
        scale = amplitude / 2.0
        
        # Standardize the target and base predictions
        y_norm = (y_stacked - center) / scale
        base_pred_norm = (base_pred_stacked - center) / scale
        
        return {
            "phi_matrix": phi_matrix.to(TORCH_DEVICE),
            "y_stacked": y_norm,
            "base_pred_stacked": base_pred_norm,
            "y_center": center,
            "y_scale": scale,
            "unique_groups": unique_groups,
            "df_original": df_original,
            "mask": mask
        }

    def _run_filter(
        self,
        prepared_data: Dict[str, Any],
        P_init_diag: float, 
        observation_noise_var: float,
        process_noise_var: float
    ) -> torch.Tensor:
        """
        Executes the core Kalman filter loop on normalized data and 
        denormalizes the results before returning.
        """
        phi_matrix = prepared_data["phi_matrix"]
        y_stacked = prepared_data["y_stacked"]
        base_pred_stacked = prepared_data["base_pred_stacked"]

        actual_block_size = 1 if self.horizon_steps_ > 1 else self.block_size_

        predictions_norm, state_history_gpu = _kalman_block_loop_optimized(
            phi_matrix=phi_matrix,
            y_stacked=y_stacked,
            base_pred_stacked=base_pred_stacked,
            B=actual_block_size,
            P_init_diag=float(P_init_diag),
            observation_noise_var=float(observation_noise_var),
            process_noise_var=float(process_noise_var),
            eps=self.eps,
            offset_boost=float(self.offset_boost)
        )

        self.states_history_ = state_history_gpu.squeeze(-1).cpu()

        if self.horizon_steps_ > 1:
            shift = self.horizon_steps_ - 1
            shifted_states = torch.zeros_like(state_history_gpu)
            shifted_states[shift:] = state_history_gpu[:-shift]
            shifted_states_G = shifted_states.permute(1, 0, 2, 3) 
            delta = torch.matmul(phi_matrix.unsqueeze(-2), shifted_states_G).squeeze(-1) 
            predictions_norm = base_pred_stacked + delta
        
        # --- Denormalize Predictions ---
        predictions_denorm = (predictions_norm * prepared_data["y_scale"]) + prepared_data["y_center"]
        
        return predictions_denorm

    def tune_hyperparameters(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, list],
        lookback_days: int = 30
    ) -> Tuple[Dict[str, float], float]:
        """
        Optimizes noise parameters by dynamically locating valid validation windows.
        Because targets are internally normalized, the hyperparameter grid is highly 
        stable and invariant to the original data's scale.

        Args:
            data: The full dataset for preparation.
            param_grid: Dictionary of hyperparameters to test.
            lookback_days: Number of days to include in the validation slice.

        Returns:
            Tuple containing the best parameters and corresponding RMSE.
        """
        print("1. Preparing data, standardizing targets, and searching for valid validation window...")
        prepared_data = self.prepare_data(data)
        
        y_orig = (prepared_data["y_stacked"] * prepared_data["y_scale"]) + prepared_data["y_center"]
        
        valid_indices = torch.where(torch.isfinite(y_orig).any(dim=0).any(dim=-1))[0]
        
        if len(valid_indices) == 0:
            raise ValueError("No valid target data found in the entire dataset. Tuning is impossible.")
        
        last_valid_idx = valid_indices[-1].item()
        val_start_idx = max(0, last_valid_idx - (lookback_days * 24)) 
        
        print(f"   -> Data density confirmed. Validation window: indices {val_start_idx} to {last_valid_idx}.")
        
        y_val_orig = y_orig[:, val_start_idx:last_valid_idx+1, :]
        best_score, best_params = float('inf'), None
        combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        
        print(f"2. Testing {len(combinations)} scale-invariant hyperparameter combinations...")
        for i, params in enumerate(combinations):
            preds_denorm = self._run_filter(
                prepared_data, 
                params.get("P_init_diag", 1.0), 
                params["observation_noise_var"], 
                params["process_noise_var"]
            )
            
            preds_val = preds_denorm[:, val_start_idx:last_valid_idx+1, :]
            
            sq_err = (y_val_orig - preds_val)**2
            valid_mask = ~torch.isnan(sq_err)
            
            if valid_mask.sum() > 0:
                rmse = torch.sqrt(sq_err[valid_mask].mean()).item()
            else:
                rmse = float('nan')
            
            print(f"  - Combo {i+1}/{len(combinations)}: {params} -> RMSE: {rmse:.4f}")
            
            if rmse < best_score and not np.isnan(rmse):
                best_score, best_params = rmse, params
                
        return best_params, best_score

    def predict_online(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Main interface to adapt a base model to current drift.
        Reassembles tensors into the user's original DataFrame shape.
        """
        prepared_data = self.prepare_data(data)
        preds_denorm = self._run_filter(
            prepared_data, 
            self.P_init_diag_,
            self.observation_noise_var_, 
            self.process_noise_var_
        )
        
        df_final = _reassemble_predictions(
            prepared_data["df_original"], 
            preds_denorm.squeeze(-1).cpu(),
            self.group_col_, 
            prepared_data["unique_groups"], 
            self.target_col_,
            date_col=self.date_col_
        )
        
        res_col = f"Estimated{self.target_col_}"
        df_final = df_final.rename(columns={res_col: f"KalmanAdapted_{self.target_col_}"})
        
        df_final_masked = df_final[prepared_data["mask"]]

        # Save frozen states and scales for out-of-sample inference
        self.last_state_dict_ = {}
        self.scale_dict_ = {}
        for i, g in enumerate(prepared_data["unique_groups"]):
            self.last_state_dict_[g] = self.states_history_[-1, i, :].clone()
            self.scale_dict_[g] = prepared_data["y_scale"][i, 0, 0].cpu().clone()

        return _cleanup_dummies(df_final_masked, self.group_col_, self.date_col_)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'KalmanTAM':
        r"""
        Fits the Kalman filter by running the historical tracking simulation.
        Learns and saves the optimal state (drift weights) per group.

        Args:
            data: Training DataFrame.
            
        Returns:
            self: The fitted KalmanTAM instance.
        """
        self.predict_online(data, **kwargs)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Applies the frozen end-of-training Kalman state to new (test) data.
        
        Uses the last updated tracking weights from the historical simulation 
        to project the drift forward as a stable, static rule.

        Args:
            df: New DataFrame containing the features.

        Returns:
            pd.DataFrame: DataFrame augmented with the `KalmanAdapted_{target}` column.
        """
        if getattr(self, 'last_state_dict_', None) is None:
            raise RuntimeError("Call fit() first to populate the Kalman states history.")

        df = _ensure_dummies(df, self.group_col_, self.date_col_)
        mask, balanced_data = _balance_groups(
            dataset=df, group_col=self.group_col_, date_col=self.date_col_, method="fill"
        )

        phi_matrix, _, base_pred_stacked, unique_groups_pred, df_original = self._prepare_kalman_features(balanced_data, is_inference=True)

        G_pred, N_pred, d_pred = phi_matrix.shape
        device = phi_matrix.device
        dtype = phi_matrix.dtype

        # Construct the frozen state and scale tensors aligned with the inference groups
        last_state_tensor = torch.zeros((G_pred, d_pred, 1), device=device, dtype=dtype)
        scale_tensor = torch.ones((G_pred, 1, 1), device=device, dtype=dtype)

        for i, g in enumerate(unique_groups_pred):
            if g in self.last_state_dict_:
                last_state_tensor[i, :, 0] = self.last_state_dict_[g].to(device)
                scale_tensor[i, 0, 0] = self.scale_dict_[g].to(device)
            else:
                # If a completely new group is encountered during inference, 
                # the tensors remain zero -> zero drift (base model physics only).
                pass

        # Compute the correction delta in normalized space
        delta_norm = torch.bmm(phi_matrix, last_state_tensor)
        
        # Scale back to real target space (base_pred is already in real space)
        delta_real = delta_norm * scale_tensor
        predictions_denorm = base_pred_stacked + delta_real

        df_final = _reassemble_predictions(
            df_original,
            predictions_denorm.squeeze(-1).cpu(),
            self.group_col_,
            unique_groups_pred,
            self.target_col_,
            date_col=self.date_col_
        )

        res_col = f"Estimated{self.target_col_}"
        df_final = df_final.rename(columns={res_col: f"KalmanAdapted_{self.target_col_}"})

        df_final_masked = df_final[mask]
        return _cleanup_dummies(df_final_masked, self.group_col_, self.date_col_)