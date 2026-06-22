r"""
Implements the Adaptive TAM (AdaptiveTAM) model.

This module defines a two-stage "online" model architecture:

1. A pre-trained ``base_model`` (StaticTAM) provides an initial, stable,
   long-term prediction.
2. An ``adaptive_model`` (also an StaticTAM instance) learns to correct
   the base model's residuals using a sliding-window simulation.

This approach allows the system to adapt to short-term drifts and anomalies
by mapping the base model's decomposed effects to its current error.
"""

from typing import Dict, List, Any, Union, Tuple, Optional
import torch
import pandas as pd
import numpy as np
import warnings

from .additive import StaticTAM
from tam.common.utils import (
    TORCH_DEVICE, _check_features, _balance_groups,
    _ensure_dummies, _cleanup_dummies
)
from tam.common.hardware import hw
from ._memory import get_safe_window_batch_size

from ._data import (
    _fit_normalization_params,
    _transform_data_adaptive,
    _reassemble_predictions
)
from ._math import (
    _predict_from_coeffs,
    _compute_weighted_covariances,
    solve_linear_system
)
from .spectrum import (
    BaseEffect,
    create_effects_from_parsed_terms,
    build_phi_from_effects,
    build_penalty_from_effects
)

#: <init_adaptive>
class AdaptiveTAM:
    r"""
    Initializes the AdaptiveTAM model.
    """
    
    def __init__(
        self,
        adaptive_formula: str,
        update_interval_periods: int,
        training_window_periods: int,
        steps_per_period: int,
        base_model: Optional[StaticTAM] = None,
        horizon_steps: int = 1,
        default_alpha_p: float = -9.0,
        group_col: Optional[str] = None,
        date_col: Optional[str] = None,
        add_base_effects: bool = False
    ):
        r"""
        Initializes the AdaptiveTAM model.

        Args:
            base_model: A fitted StaticTAM model instance.
            adaptive_formula: Formula for the adaptive correction model.
                              Features must be columns produced by base_model.decompose_prediction()
                              (e.g., 'effect_temp').
            update_interval_periods (int): The number of periods to skip before updating 
                the coefficients for a specific group (determines n_windows).
            training_window_periods (int): The historical look-back period used to solve 
                the local linear system for an independent group (determines num_samples_train).
            steps_per_period (int): The number of observations a single group experiences 
                within one logical period. 
                - If the data is monthly and the period is a month: 1.
                - If the data is 30-min, but group_col is 'Time of Day': 1 (since each group only sees one observation per day).
            horizon_steps (int): The forecasting horizon (H) used to prevent target leakage. 
                Enforces an information delay by truncating the last (H-1) samples from the 
                training buffer of every group simultaneously.
            default_alpha_p: Default regularization strength (log10).
        
        Raises:
            ValueError: If the base_model has not been fitted.
        """
        if base_model is not None and getattr(base_model, 'coefficients_', None) is None:
            raise ValueError("The base_model must be fitted before initializing AdaptiveTAM.")
        
        if add_base_effects and base_model is not None:
            for effect in base_model.effects_list_:
                effect_col = f"effect_{effect.feature_name}"
                if effect_col not in adaptive_formula:
                    adaptive_formula += f" + l({effect_col})"

        self.base_model_ = base_model
        self.adaptive_formula_ = adaptive_formula
        self.update_interval_periods_ = update_interval_periods
        self.training_window_periods_ = training_window_periods
        self.steps_per_period_ = steps_per_period
        self.horizon_steps_ = horizon_steps
        
        self.group_col_ = group_col or getattr(base_model, 'group_col_', "__dummy_group__")
        self.date_col_ = date_col or getattr(base_model, 'date_col_', "__dummy_date__")

        self.adaptive_model_ = StaticTAM(
            formula=adaptive_formula,
            group_col=self.group_col_,
            date_col=self.date_col_,
            default_alpha_p=default_alpha_p
        )
        
        self.coefficients_ = None
        self.norm_params_ = None
        self.unique_groups_ = None
        
        self.simulation_data_ = None
        self.predictions_ = None
        
        target_col_bm = getattr(base_model, 'target_col_', self.adaptive_model_.target_col_)
        self.target_col_ = self.adaptive_model_.target_col_ or f'Residual{target_col_bm}'

        if self.base_model_ is not None:
            base_target = getattr(self.base_model_, 'target_col_', None)
            if base_target and self.adaptive_model_.target_col_:
                expected_residual = f'Residual{base_target}'
                if self.adaptive_model_.target_col_ not in [expected_residual, base_target]:
                    warnings.warn(
                        f"Adaptive target '{self.adaptive_model_.target_col_}' does not match the base target "
                        f"'{base_target}' or its expected residual '{expected_residual}'. "
                        "This will cause a cross-target correction.",
                        UserWarning
                    )
        
        self.last_state_dict_ = None
        self.max_res_ = None
        self.min_res_ = None
#: </init_adaptive>

#: <prepare_sim>
    def prepare_simulation(self, data: pd.DataFrame) -> 'AdaptiveTAM':
        r"""
        Prepares tensors for the adaptive sliding-window simulation.

        This process:
        1. Computes base model effects (features) and residuals (targets).
        2. Normalizes the adaptive features.
        3. Constructs sliding-window tensors (X_train, Y_train, X_predict).

        Args:
            data: The dataset (validation or test) for the simulation.

        Returns:
            self: The instance with populated ``simulation_data_``.
        """
        
        if self.base_model_ is not None:
            data_bm = self.base_model_.decompose_prediction(data) 
            data_pred = self.base_model_.predict(data)
            target_col_bm = self.base_model_.target_col_
        else:
            data_bm = data.copy()
            data_pred = data.copy()
            target_col_bm = self.target_col_
       
        data_bm = _ensure_dummies(data_bm, self.group_col_, self.date_col_)
        data_pred = _ensure_dummies(data_pred, self.group_col_, self.date_col_)
        
        cols_float = data_bm.select_dtypes(include=['float64']).columns
        data_bm[cols_float] = data_bm[cols_float].astype('float32')

        est_col_name = f'Estimated{target_col_bm}'

        if est_col_name not in data_bm.columns:
            data_bm[est_col_name] = data_pred.get(est_col_name, 0.0)

        default_res_col = f'Residual{target_col_bm}'
        data_bm[default_res_col] = data_bm[target_col_bm] - data_bm[est_col_name]
        
        adaptive_features_config = self.adaptive_model_.features_config_
        
        if 'features' not in adaptive_features_config or not isinstance(adaptive_features_config['features'], list):
            raise KeyError("Invalid feature configuration in adaptive model.")
        
        adaptive_features = adaptive_features_config['features']
        
        required_cols = adaptive_features + [self.group_col_, self.target_col_, self.date_col_]
        _check_features(dataset=data_bm, required_features=required_cols)
        mask, balanced_data = _balance_groups(
            dataset=data_bm, group_col=self.group_col_, date_col=self.date_col_, method="fill"
        )
        
        data_info = self.adaptive_model_._get_data_info(balanced_data)

        if not self.adaptive_model_.effects_list_ and not self.adaptive_model_.is_grid_search_template_:
            self.adaptive_model_.effects_list_ = create_effects_from_parsed_terms(
                self.adaptive_model_.parsed_terms_, 
                token_values={}, 
                default_alpha_p=self.adaptive_model_.default_alpha_p_,
                data_info=data_info
            )

        self.norm_params_, self.unique_groups_ = _fit_normalization_params(
            data=balanced_data, 
            features=adaptive_features, 
            group_col=self.group_col_
        )
        
        hw.empty_cache()

        x_stacked, y_stacked, x_to_predict = _transform_data_adaptive(
            data=balanced_data, 
            features=adaptive_features, 
            group_col=self.group_col_,
            norm_params=self.norm_params_,
            unique_groups=self.unique_groups_,
            target_col=self.target_col_,
            update_interval_periods=self.update_interval_periods_,
            training_window_periods=self.training_window_periods_,
            steps_per_period=self.steps_per_period_,
            horizon_steps=self.horizon_steps_
        )
        
        self.simulation_data_ = (
            x_stacked.cpu(), 
            y_stacked.cpu(), 
            x_to_predict.cpu(), 
            balanced_data, data_bm, target_col_bm, mask
        )
        del x_stacked, y_stacked, x_to_predict
        hw.empty_cache()

        return self
#: </prepare_sim>
    
#: <run_sim>
    def simulation(self) -> pd.DataFrame:
        r"""
        Executes the sliding-window simulation using scalable batch processing.
        
        It flattens the group and window dimensions to treat each sliding window 
        as an independent linear system. This strictly preserves the mathematical 
        regularization scale and prevents VRAM exhaustion when processing deep 
        historical data across multiple groups.
        """
        if self.simulation_data_ is None:
            raise RuntimeError("Simulation data is uninitialized. Call 'prepare_simulation()' first.")
            
        if self.adaptive_model_.is_grid_search_template_:
            raise RuntimeError("Model contains grid search tokens. Call 'grid_search_fit()' first.")

        x_stacked, y_stacked, x_to_predict, balanced_data, data_bm, target_col_bm, mask = self.simulation_data_
        
        n_groups = x_stacked.shape[0]
        n_windows = x_stacked.shape[1]
        num_samples_train = x_stacked.shape[2]
        window_size_steps = x_to_predict.shape[2]
        
        total_items = n_groups * n_windows
        
        x_flat = x_stacked.view(total_items, num_samples_train, -1)
        y_flat = y_stacked.view(total_items, num_samples_train, -1)
        x_pred_flat = x_to_predict.view(total_items, window_size_steps, -1)
        
        run_device = TORCH_DEVICE
        
        sobolev_matrix = self.adaptive_model_._build_penalty_matrix().to(run_device)
        loss_L_star_L = self.adaptive_model_._build_loss_matrix().to(run_device)

        dummy_x = x_flat[0:1].to(run_device)
        dummy_phi = self.adaptive_model_._build_design_matrix(dummy_x)
        n_coeffs = dummy_phi.shape[-1]
        del dummy_x, dummy_phi
        
        safe_batch_size = get_safe_window_batch_size(
            num_samples_per_window=num_samples_train,
            total_d=n_coeffs,
            device=run_device
        )
        safe_batch_size = min(safe_batch_size, total_items)
        all_predictions = []
        
        start_idx = 0
        while start_idx < total_items:
            end_idx = min(start_idx + safe_batch_size, total_items)
            
            try:
                batch_x = x_flat[start_idx:end_idx].to(run_device)
                batch_y = y_flat[start_idx:end_idx].to(run_device)
                batch_x_pred = x_pred_flat[start_idx:end_idx].to(run_device)

                phi_batch = self.adaptive_model_._build_design_matrix(batch_x)

                cov_X, cov_XY = _compute_weighted_covariances(phi_batch, batch_y, loss_L_star_L)
                coeffs_batch = solve_linear_system(cov_X, cov_XY, sobolev_matrix, num_samples_train)
                
                phi_pred_batch = self.adaptive_model_._build_design_matrix(batch_x_pred)
                preds_batch = _predict_from_coeffs(phi_pred_batch, coeffs_batch)
                
                all_predictions.append(preds_batch.detach().cpu())
                
                del batch_x, batch_y, batch_x_pred, phi_batch, cov_X, cov_XY, coeffs_batch, phi_pred_batch, preds_batch
                start_idx += safe_batch_size
                
            except (torch.OutOfMemoryError, MemoryError):
                safe_batch_size, run_device = hw.handle_oom(
                    current_batch=safe_batch_size, 
                    device=run_device, 
                    context="adaptive simulation batch reduction", 
                    allow_cpu_fallback=True
                )
                
                sobolev_matrix = sobolev_matrix.to(run_device)
                loss_L_star_L = loss_L_star_L.to(run_device)
                continue

        hw.empty_cache()

        predictions_flat = torch.cat(all_predictions, dim=0).squeeze(-1)
        predictions_cpu = predictions_flat.view(n_groups, n_windows, window_size_steps)

        data_with_predictions = _reassemble_predictions(
            original_data=balanced_data, 
            predictions_stacked=predictions_cpu,
            group_col=self.group_col_,
            unique_groups=self.unique_groups_, 
            target_col=self.target_col_
        )

        max_res = np.float32(data_bm[self.target_col_].max())
        min_res = np.float32(data_bm[self.target_col_].min())
        est_col = f'Estimated{self.target_col_}'
        
        data_with_predictions.loc[data_with_predictions[est_col] >= max_res, est_col] = max_res
        data_with_predictions.loc[data_with_predictions[est_col] <= min_res, est_col] = min_res
                
        adapted_col = f"AdaptedEstimated{target_col_bm}"
        if self.target_col_ == target_col_bm:
            data_with_predictions[adapted_col] = data_with_predictions[est_col].fillna(0)
        else:
            data_with_predictions[adapted_col] = (
                data_with_predictions[f'Estimated{target_col_bm}'] + 
                data_with_predictions[est_col].fillna(0)
            )
                                                 
        self.predictions_ = _cleanup_dummies(data_with_predictions[mask], self.group_col_, self.date_col_)
        return self.predictions_
#: </run_sim>

    def _save_final_state(self):
        r"""
        Extracts the final window from the prepared simulation tensors, 
        solves the linear system to get the most recent adaptive parameters, 
        and saves them for out-of-sample inference.
        """
        x_stacked, y_stacked, _, _, data_bm, _, _ = self.simulation_data_
        
        # Save historical bounds for safety clipping during inference
        self.max_res_ = np.float32(data_bm[self.target_col_].max())
        self.min_res_ = np.float32(data_bm[self.target_col_].min())
        
        run_device = TORCH_DEVICE
        num_samples_train = x_stacked.shape[2]
        
        # Extract ONLY the last available training window (index -1)
        x_last = x_stacked[:, -1, :, :].to(run_device)
        y_last = y_stacked[:, -1, :, :].to(run_device)
        
        loss_L_star_L = self.adaptive_model_._build_loss_matrix().to(run_device)
        sobolev_matrix = self.adaptive_model_._build_penalty_matrix().to(run_device)
        
        phi_train = self.adaptive_model_._build_design_matrix(x_last)
        cov_X, cov_XY = _compute_weighted_covariances(phi_train, y_last, loss_L_star_L)
        
        final_coeffs = solve_linear_system(cov_X, cov_XY, sobolev_matrix, num_samples_train)
        
        self.last_state_dict_ = {}
        for i, g in enumerate(self.unique_groups_):
            self.last_state_dict_[g] = final_coeffs[i].cpu().clone()

    def predict_online(self, data: pd.DataFrame) -> pd.DataFrame:
        r"""
        Runs the full adaptive historical pipeline: preparation + simulation.
        """
        self.prepare_simulation(data)
        self.simulation()
        self._save_final_state()
        return self.predictions_

    def fit(self, data: pd.DataFrame) -> 'AdaptiveTAM':
        r"""
        Fits the adaptive model by solving the regularized linear system
        ONLY for the final available training window per group.
        This provides the optimal final parameters for out-of-sample inference
        without executing the full historical sliding-window simulation.
        """
        self.prepare_simulation(data)
        self._save_final_state()
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Predicts on new data using the frozen adaptive state learned from 
        the final training window.
        """
        if getattr(self, 'last_state_dict_', None) is None:
            raise RuntimeError("Call fit() first to train the final adaptive state.")

        # --- 1. Base Model Extraction ---
        if self.base_model_ is not None:
            data_bm = self.base_model_.decompose_prediction(df)
            data_pred = self.base_model_.predict(df)
            target_col_bm = self.base_model_.target_col_
        else:
            data_bm = df.copy()
            data_pred = df.copy()
            target_col_bm = self.target_col_

        data_bm = _ensure_dummies(data_bm, self.group_col_, self.date_col_)
        data_pred = _ensure_dummies(data_pred, self.group_col_, self.date_col_)

        est_col_name = f'Estimated{target_col_bm}'
        if est_col_name not in data_bm.columns:
            data_bm[est_col_name] = data_pred.get(est_col_name, 0.0)

        mask, balanced_data = _balance_groups(
            dataset=data_bm, group_col=self.group_col_, date_col=self.date_col_, method="fill"
        )
        
        # --- 2. Adaptive Feature Matrix ---
        x_pred, _, unique_groups_pred = self.adaptive_model_._prepare_data(
            balanced_data, target_col=None, ignore_template_check=True
        )
        
        run_device = TORCH_DEVICE
        x_pred = x_pred.to(run_device)
        phi_pred = self.adaptive_model_._build_design_matrix(x_pred)
        
        # --- 3. Construct Frozen Coefficient Tensor ---
        G_pred, N_pred, d_pred = phi_pred.shape
        coeffs_tensor = torch.zeros((G_pred, d_pred, 1), device=run_device, dtype=phi_pred.dtype)
        
        for i, g in enumerate(unique_groups_pred):
            if g in self.last_state_dict_:
                coeffs_tensor[i, :, 0] = self.last_state_dict_[g].to(run_device).squeeze(-1)
                
        res_pred_tensor = _predict_from_coeffs(phi_pred, coeffs_tensor)
        
        data_with_predictions = _reassemble_predictions(
            original_data=balanced_data,
            predictions_stacked=res_pred_tensor.squeeze(-1).cpu(),
            group_col=self.group_col_,
            unique_groups=unique_groups_pred,
            target_col=self.target_col_
        )
        
        # --- 4. Safety Bounding and Reassembly ---
        est_col = f'Estimated{self.target_col_}'
        adapted_col = f"AdaptedEstimated{target_col_bm}"
        
        if getattr(self, 'max_res_', None) is not None:
             data_with_predictions.loc[data_with_predictions[est_col] >= self.max_res_, est_col] = self.max_res_
             data_with_predictions.loc[data_with_predictions[est_col] <= self.min_res_, est_col] = self.min_res_

        if self.target_col_ == target_col_bm:
            data_with_predictions[adapted_col] = data_with_predictions[est_col].fillna(0)
        else:
            data_with_predictions[adapted_col] = (
                data_with_predictions[est_col_name] + 
                data_with_predictions[est_col].fillna(0)
            )

        return _cleanup_dummies(data_with_predictions[mask], self.group_col_, self.date_col_)

    def _evaluate_adaptive_config(
        self, 
        effects_list: List[BaseEffect], 
        token_values: Dict 
    ) -> float:
        r"""
        Evaluates a specific hyperparameter configuration during grid search.
        
        Uses the exact same flattened dimension strategy as the main simulation
        to preserve mathematical integrity and avoid OOM during optimization.
        """
        x_stacked, y_stacked, x_to_predict, balanced_data, data_bm, target_col_bm, mask = self.simulation_data_
        
        if x_stacked.dtype == torch.float64: x_stacked = x_stacked.float()
        if y_stacked.dtype == torch.float64: y_stacked = y_stacked.float()
        if x_to_predict.dtype == torch.float64: x_to_predict = x_to_predict.float()

        n_groups = x_stacked.shape[0]
        n_windows = x_stacked.shape[1]
        num_samples_train = x_stacked.shape[2]
        window_size_steps = x_to_predict.shape[2]
        
        total_items = n_groups * n_windows

        x_flat = x_stacked.view(total_items, num_samples_train, -1)
        y_flat = y_stacked.view(total_items, num_samples_train, -1)
        x_pred_flat = x_to_predict.view(total_items, window_size_steps, -1)
        
        run_device = TORCH_DEVICE
        
        dummy_x = x_flat[0:1].to(run_device)
        try:
            dummy_phi = build_phi_from_effects(dummy_x, effects_list)
            n_coeffs = dummy_phi.shape[-1]
            del dummy_x, dummy_phi
        except Exception:
            return float('inf') 

        safe_batch_size = get_safe_window_batch_size(
            num_samples_per_window=num_samples_train,
            total_d=n_coeffs,
            device=run_device
        )
        safe_batch_size = min(safe_batch_size, total_items)
        
        loss_L_star_L = None
        penalty_M_star_M = None

        try:
            loss_L_star_L = self.adaptive_model_._build_loss_matrix().to(run_device)
            penalty_M_star_M = build_penalty_from_effects(effects_list).to(run_device)
        except Exception:
             return float('inf')
        
        all_preds_cpu = []
        start_idx = 0

        try:
            while start_idx < total_items:
                end_idx = min(start_idx + safe_batch_size, total_items)
                
                try:
                    batch_x = x_flat[start_idx:end_idx].to(run_device)
                    batch_y = y_flat[start_idx:end_idx].to(run_device)
                    batch_x_pred = x_pred_flat[start_idx:end_idx].to(run_device)

                    phi_train = build_phi_from_effects(batch_x, effects_list)
                    phi_val = build_phi_from_effects(batch_x_pred, effects_list)
                    
                    cov_X, cov_XY = _compute_weighted_covariances(phi_train, batch_y, loss_L_star_L)
                    
                    adaptive_coeffs = solve_linear_system(
                        cov_X, cov_XY, penalty_M_star_M, num_samples_train
                    )
                    
                    batch_preds = _predict_from_coeffs(phi_val, adaptive_coeffs)
                    all_preds_cpu.append(batch_preds.detach().cpu())
                    
                    del batch_x, batch_y, batch_x_pred, phi_train, phi_val, cov_X, cov_XY, adaptive_coeffs
                    start_idx += safe_batch_size
                    
                except (torch.OutOfMemoryError, MemoryError):
                    try:
                        safe_batch_size, run_device = hw.handle_oom(
                            current_batch=safe_batch_size, 
                            device=run_device, 
                            context="adaptive evaluation batch reduction", 
                            allow_cpu_fallback=True
                        )
                        loss_L_star_L = loss_L_star_L.to(run_device)
                        penalty_M_star_M = penalty_M_star_M.to(run_device)
                        continue
                    except MemoryError:
                        return float('inf')
                
            predictions_flat = torch.cat(all_preds_cpu, dim=0).squeeze(-1)
            predictions_stacked = predictions_flat.view(n_groups, n_windows, window_size_steps)
            
            data_with_predictions = _reassemble_predictions(
                original_data=balanced_data.copy(), 
                predictions_stacked=predictions_stacked,
                group_col=self.group_col_,
                unique_groups=self.unique_groups_, 
                target_col=self.target_col_
            )

            max_res = np.float32(data_bm[self.target_col_].max())
            min_res = np.float32(data_bm[self.target_col_].min())
            est_col = f'Estimated{self.target_col_}'
            
            data_with_predictions.loc[data_with_predictions[est_col] >= max_res, est_col] = max_res
            data_with_predictions.loc[data_with_predictions[est_col] <= min_res, est_col] = min_res
            
            adapted_col = f"AdaptedEstimated{target_col_bm}"
            if self.target_col_ == target_col_bm:
                data_with_predictions[adapted_col] = data_with_predictions[est_col].fillna(0)
            else:
                data_with_predictions[adapted_col] = (
                    data_with_predictions[f'Estimated{target_col_bm}'] + 
                    data_with_predictions[est_col].fillna(0)
                )
            
            rmse_df = data_with_predictions[mask]
            rmse_df = rmse_df[[target_col_bm, adapted_col]].copy().dropna()
            
            if rmse_df.empty:
                return float('inf')
            
            gt = rmse_df[target_col_bm].values
            pred = rmse_df[adapted_col].values
            return float(np.sqrt(np.mean((gt - pred)**2)))

        except Exception:
            return float('inf')
        
        finally:
            del loss_L_star_L, penalty_M_star_M
            hw.empty_cache()

#: <grid_search_adaptive>
    def grid_search_fit(
            self,
            data_val: pd.DataFrame,
            grid_search_config: dict
        ) -> 'AdaptiveTAM':
            r"""
            Optimizes the adaptive model using Multi-Start Coordinate Descent.

            Optimizes hyperparameters to minimize the RMSE of the final
            adapted prediction over the simulation period.

            Args:
                data_val: Validation DataFrame for simulation.
                grid_search_config: Dictionary mapping tokens to value lists.

            Returns:
                AdaptiveTAM: A new fitted model instance.
            """
            print("--- Starting Grid Search (Multi-Start Coordinate Descent) ---")
            
            self.prepare_simulation(data_val)
            
            search_axes, token_names = self.adaptive_model_._parse_grid_axes(grid_search_config)
            
            data_info = self.adaptive_model_._get_data_info(data_val)

            if not token_names:
                raise ValueError("No grid tokens found in adaptive formula or config.")

            print(f"Optimizing axes: {token_names}")
            
            start_points = [
                {"name": "Conservative", "tokens": {t: max(vals) if ('ap' in t or 'lambda_p' in t) else min(vals) for t, vals in search_axes.items()}},
                {"name": "Median", "tokens": {t: vals[len(vals)//2] for t, vals in search_axes.items()}},
                {"name": "Aggressive", "tokens": {t: min(vals) if ('ap' in t or 'lambda_p' in t) else max(vals) for t, vals in search_axes.items()}}
            ]

            min_global_rmse = float('inf')
            optimal_effects_list = None
            current_best_tokens_global = None

            for strategy in start_points:
                print(f"\n=== Strategy: {strategy['name']} Start ===")
                current_best_tokens = strategy["tokens"].copy()
                
                try:
                    start_effects_list = create_effects_from_parsed_terms(
                        self.adaptive_model_.parsed_terms_,
                        current_best_tokens,
                        self.adaptive_model_.default_alpha_p_,
                        data_info=data_info
                    )
                    current_rmse = self._evaluate_adaptive_config(start_effects_list, current_best_tokens)
                    current_optimal_effects = start_effects_list
                except Exception:
                    continue
                    
                if current_rmse >= float('inf'):
                    continue
                
                print(f"  Start RMSE: {current_rmse:.4f}")

                cycle = 0
                while True:
                    cycle += 1
                    has_improved_in_cycle = False
                    
                    for token_name in token_names:
                        best_value = current_best_tokens[token_name]
                        original_val = best_value
                        
                        for value in search_axes[token_name]:
                            if value == original_val: continue 
                                
                            tokens_to_test = current_best_tokens.copy()
                            tokens_to_test[token_name] = value
                            
                            try:
                                effects_list = create_effects_from_parsed_terms(
                                    self.adaptive_model_.parsed_terms_,
                                    tokens_to_test,
                                    self.adaptive_model_.default_alpha_p_,
                                    data_info=data_info
                                )
                                rmse = self._evaluate_adaptive_config(effects_list, tokens_to_test)

                                if rmse < current_rmse:
                                    current_rmse = rmse
                                    current_optimal_effects = effects_list
                                    best_value = value
                                    has_improved_in_cycle = True
                            except Exception:
                                continue
                        
                        current_best_tokens[token_name] = best_value
                    
                    if not has_improved_in_cycle or cycle >= 5: break

                if current_rmse < min_global_rmse:
                    print(f"  >>> New Global Best found by {strategy['name']}! ({current_rmse:.4f})")
                    min_global_rmse = current_rmse
                    optimal_effects_list = current_optimal_effects
                    current_best_tokens_global = current_best_tokens

            print("-" * 30)
            print(f"Grid Search complete. Best RMSE: {min_global_rmse:.4f}")
            
            if optimal_effects_list is None:
                raise RuntimeError("Grid search failed.")

            print(f"Best tokens: {current_best_tokens_global}")

            final_model = AdaptiveTAM(
                base_model=self.base_model_,
                adaptive_formula=self.adaptive_formula_,
                update_interval_periods=self.update_interval_periods_,
                training_window_periods=self.training_window_periods_,
                steps_per_period=self.steps_per_period_,
                horizon_steps=self.horizon_steps_
            )
            
            final_adaptive_model_internal = StaticTAM(
                formula=self.adaptive_model_.formula_,
                group_col=self.base_model_.group_col_ if self.base_model_ is not None else self.group_col_,
                date_col=self.base_model_.date_col_ if self.base_model_ is not None else self.date_col_,
                _internal_effects_list=optimal_effects_list,
                _internal_features_config=self.adaptive_model_.features_config_
            )
            
            final_model.adaptive_model_ = final_adaptive_model_internal
            final_model.norm_params_ = self.norm_params_
            final_model.unique_groups_ = self.unique_groups_
            final_model.target_col_ = self.target_col_

            final_model.predict_online(data_val)
            
            return final_model
#: </grid_search_adaptive>