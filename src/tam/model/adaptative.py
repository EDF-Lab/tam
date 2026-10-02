# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France) et Sorbonne Université
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-License-Identifier: LGPL-3.0-or-later
# Authors : Yann Allioux, Nathan Doumèche, Éloi Bedek

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
from tam.common.exceptions import warn_extrapolation
from tam.common.utils import (
    TORCH_DEVICE, _check_features, _balance_groups,
    _ensure_dummies, _cleanup_dummies
)
from tam.common.hardware import hw
from ._memory import get_safe_window_batch_size

from ._data import (
    _transform_data_adaptive,
    _transform_data_stacked,
    _window_starts,
    _groups_in_data,
    _check_known_groups,
    _reassemble_predictions
)
from ._math import (
    _predict_from_coeffs,
    _compute_weighted_covariances,
    decomposition_names,
    solve_linear_system
)
from .spectrum import (
    BaseEffect,
    create_effects_from_parsed_terms,
    categorical_ranges,
    extrapolating_features,
    initialize_effects,
    build_phi_from_effects,
    build_penalty_from_effects
)

def rolling_windows(
    data: pd.DataFrame,
    training_window_periods: int,
    update_interval_periods: int,
    steps_per_period: int = 1,
    horizon_steps: int = 1,
    group_col: Optional[str] = None,
    date_col: Optional[str] = None
):
    r"""
    The windows an operational refit would use, and the ones ``AdaptiveTAM`` simulates: ``(train_rows, forecast_rows)``.

    Windows are anchored on the start of the data. The first one trains on the first ``training_window_periods * steps_per_period``
    rows of every group and forecasts the ``update_interval_periods * steps_per_period`` rows that follow, ``horizon_steps - 1`` rows
    after the end of the training rows (the target of a row is known ``horizon_steps`` rows later). Each next window moves forward by
    the forecast length, and the last one is cut at the end of the data. Rows before the first forecast are never forecast.

    Each element holds the index labels of ``data`` of all the groups together, so that, for the same windows,
    ``StaticTAM(...).fit(data.loc[train_rows]).predict(data.loc[forecast_rows])`` is the model ``AdaptiveTAM`` is equal to.

    Args:
        data: The DataFrame the windows are cut in (its index must hold unique labels).
        training_window_periods: Training length, in periods.
        update_interval_periods: Forecast length, in periods.
        steps_per_period: Rows of one group in a period.
        horizon_steps: Forecasting horizon.
        group_col: Grouping column; the windows are cut inside each group.
        date_col: Column rows are ordered by (the order of ``data`` when None).

    Yields:
        Tuple[pd.Index, pd.Index]: the labels of the training rows and of the forecast rows of each window.
    """
    training_steps = training_window_periods * steps_per_period
    window_steps = update_interval_periods * steps_per_period
    if group_col is not None and group_col in data.columns:
        groups = [frame for _, frame in data.groupby(group_col, sort=True)]
    else:
        groups = [data]
    per_group = []
    for frame in groups:
        if date_col is not None and date_col in frame.columns:
            frame = frame.sort_values(date_col, kind="stable")
        labels = frame.index
        starts = _window_starts(len(labels), training_steps, window_steps, horizon_steps)
        per_group.append((labels, starts))
    n_windows = max((len(starts) for _, starts in per_group), default=0)
    for k in range(n_windows):
        train, forecast = [], []
        for labels, starts in per_group:
            if k < len(starts):
                s = starts[k]
                train.append(labels[s - (horizon_steps - 1) - training_steps: s - (horizon_steps - 1)])
                forecast.append(labels[s: s + window_steps])
        yield train[0].append(train[1:]), forecast[0].append(forecast[1:])


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
        add_base_effects: bool = False,
        clip_to_train_range: bool = False
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
            clip_to_train_range: Clips the forecast of every window to the range of the target in the training rows of that
                same window (False by default: no clipping).
        
        Raises:
            ValueError: If the base_model has not been fitted.
        """
        if base_model is not None and getattr(base_model, 'coefficients_', None) is None:
            raise ValueError("The base_model must be fitted before initializing AdaptiveTAM.")
        
        if add_base_effects and base_model is not None:
            # Column names as decompose_prediction emits them (shared features are basis-prefixed).
            for name in decomposition_names(base_model.effects_list_):
                effect_col = f"effect_{name}"
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
        
        self.clip_to_train_range_ = clip_to_train_range
        self.last_state_dict_ = None
        self.window_layout_ = None
        self.final_target_range_ = None
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

        hw.empty_cache()

        # Only real rows are windowed: the rows added to balance the groups never train and are never forecast.
        real = mask.to_numpy()
        real_data = balanced_data[real]
        self.unique_groups_ = sorted(real_data[self.group_col_].unique())
        x_stacked, y_stacked, x_to_predict, self.window_layout_ = _transform_data_adaptive(
            data=real_data,
            features=adaptive_features,
            group_col=self.group_col_,
            unique_groups=self.unique_groups_,
            target_col=self.target_col_,
            update_interval_periods=self.update_interval_periods_,
            training_window_periods=self.training_window_periods_,
            steps_per_period=self.steps_per_period_,
            horizon_steps=self.horizon_steps_,
            date_col=self.date_col_,
            fixed_ranges=categorical_ranges(self.adaptive_model_.effects_list_)
        )
        # Data-dependent state (spline knots) comes from the training windows, never from a memory probe.
        initialize_effects(
            x_stacked.reshape(-1, x_stacked.shape[2], x_stacked.shape[3]), self.adaptive_model_.effects_list_,
            feature_columns=adaptive_features
        )
        valid = torch.zeros(x_to_predict.shape[:3], dtype=torch.bool)
        for i, counts in enumerate(self.window_layout_.n_valid):
            for j, count in enumerate(counts):
                valid[i, j, :count] = True
        self._warn_extrapolation(x_to_predict.cpu(), valid)
        # Positions in `real_data` -> positions in `balanced_data` (real rows come first when groups are balanced by filling).
        real_positions = np.flatnonzero(real)
        self.window_layout_.positions = [real_positions[p] for p in self.window_layout_.positions]

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

        mask = self.simulation_data_[6]
        self.predictions_ = _cleanup_dummies(self._forecast_frame(predictions_cpu)[mask], self.group_col_, self.date_col_)
        return self.predictions_
#: </run_sim>

    def _warn_extrapolation(self, x: torch.Tensor, valid: torch.Tensor) -> None:
        r"""
        One ``TAMExtrapolationWarning`` per feature (once per model) when forecast rows leave the range of the training rows of
        their own window on a non-linear effect; the windows are aggregated: the message gives how many of them are affected.

        Args:
            x: Normalised forecast features, (n_groups, n_windows, rows, features).
            valid: Which of those rows are real forecast rows, (n_groups, n_windows, rows).
        """
        warned = self.__dict__.setdefault("_extrapolation_warned_", set())
        features = self.adaptive_model_.features_config_['features']
        nonlinear = extrapolating_features(self.adaptive_model_.effects_list_)
        for j, name in enumerate(features):
            if name not in nonlinear or name in warned:
                continue
            beyond = (x[..., j].abs() - 1.0).clamp(min=0)
            outside = (beyond > 1e-9) & valid
            if outside.any():
                warned.add(name)
                warn_extrapolation(name,
                    f"'{name}' leaves the range of its training window on {outside.sum().item() / valid.sum().item():.1%} of the "
                    f"forecast rows ({outside.any(dim=-1).sum().item()} of {valid.any(dim=-1).sum().item()} windows), by up to "
                    f"{beyond[valid].max().item():.2f} half-ranges: its non-linear effect is extrapolated and can be far off. "
                    f"Set extrapolate='constant' on the term to hold it at the edge of the trained range.", stacklevel=4)

    def _forecast_frame(self, predictions: torch.Tensor) -> pd.DataFrame:
        r"""
        Puts the (n_groups, n_windows, window) forecasts back on the rows they forecast.

        A row that no window forecasts (the first rows, and any row of a group too short for a window) is NaN in
        ``Estimated{target}``: it is never 0. ``AdaptedEstimated{base target}`` is NaN there when there is no base model, and the
        base model forecast when there is one (no correction is known yet, the base forecast stands).
        """
        _, _, _, balanced_data, _, target_col_bm, mask = self.simulation_data_
        layout = self.window_layout_
        values = predictions.double()
        if self.clip_to_train_range_:
            values = torch.minimum(torch.maximum(values, layout.y_min.view(*layout.y_min.shape, 1)), layout.y_max.view(*layout.y_max.shape, 1))
        values = values.numpy()

        est_col = f'Estimated{self.target_col_}'
        estimate = np.full(len(balanced_data), np.nan)
        for i in range(len(self.unique_groups_)):
            for j, (start, n_valid) in enumerate(zip(layout.starts[i], layout.n_valid[i])):
                if n_valid > 0:
                    estimate[layout.positions[i][start:start + n_valid]] = values[i, j, :n_valid]

        frame = balanced_data.copy()
        frame[est_col] = estimate
        adapted_col = f"AdaptedEstimated{target_col_bm}"
        if self.target_col_ == target_col_bm:
            frame[adapted_col] = frame[est_col]
        else:
            frame[adapted_col] = frame[f'Estimated{target_col_bm}'] + frame[est_col].fillna(0)
        return frame

    def _save_final_state(self):
        r"""
        Solves the window an operational refit holds after the last row (the last window of the simulation, which may not
        have a forecast row yet) and keeps its coefficients and its normalisation for ``predict()``.
        """
        x_stacked, y_stacked, _, _, _, _, _ = self.simulation_data_
        layout = self.window_layout_
        features = self.adaptive_model_.features_config_['features']
        with_state = [i for i, k in enumerate(layout.final_index) if k >= 0]
        if not with_state:
            raise ValueError("No window could be built: the data are shorter than the training window.")

        run_device = TORCH_DEVICE
        num_samples_train = x_stacked.shape[2]
        last = [layout.final_index[i] for i in with_state]
        x_last = x_stacked[with_state, last].to(run_device)
        y_last = y_stacked[with_state, last].to(run_device)

        loss_L_star_L = self.adaptive_model_._build_loss_matrix().to(run_device)
        sobolev_matrix = self.adaptive_model_._build_penalty_matrix().to(run_device)

        phi_train = self.adaptive_model_._build_design_matrix(x_last)
        cov_X, cov_XY = _compute_weighted_covariances(phi_train, y_last, loss_L_star_L)
        final_coeffs = solve_linear_system(cov_X, cov_XY, sobolev_matrix, num_samples_train)

        self.last_state_dict_, self.norm_params_, self.final_target_range_ = {}, {}, {}
        for n, (i, k) in enumerate(zip(with_state, last)):
            g = self.unique_groups_[i]
            self.last_state_dict_[g] = final_coeffs[n].cpu().clone()
            self.norm_params_[g] = {
                'min': pd.Series(layout.norm_min[i, k].numpy(), index=features),
                'max': pd.Series(layout.norm_max[i, k].numpy(), index=features),
            }
            self.final_target_range_[g] = (float(layout.y_min[i, k]), float(layout.y_max[i, k]))

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
        
        _check_known_groups(balanced_data, self.group_col_, self.unique_groups_)

        # --- 2. Adaptive Feature Matrix: normalised with the final window, like StaticTAM.predict after that window's fit ---
        features = self.adaptive_model_.features_config_['features']
        x_pred, _ = _transform_data_stacked(
            data=balanced_data, features=features, group_col=self.group_col_, norm_params=self.norm_params_,
            unique_groups=self.unique_groups_, date_col=self.date_col_
        )
        groups_stacked = _groups_in_data(balanced_data, self.group_col_, self.unique_groups_, self.norm_params_)

        self._warn_extrapolation(x_pred.unsqueeze(1).cpu(), torch.ones(x_pred.shape[0], 1, x_pred.shape[1], dtype=torch.bool))

        run_device = TORCH_DEVICE
        x_pred = x_pred.to(run_device)
        phi_pred = self.adaptive_model_._build_design_matrix(x_pred)

        # --- 3. Frozen Coefficient Tensor ---
        coeffs_tensor = torch.stack([self.last_state_dict_[g] for g in groups_stacked]).to(device=run_device, dtype=phi_pred.dtype)
        res_pred_tensor = _predict_from_coeffs(phi_pred, coeffs_tensor)
        if self.clip_to_train_range_:
            for i, g in enumerate(groups_stacked):
                low, high = self.final_target_range_[g]
                res_pred_tensor[i] = res_pred_tensor[i].clamp(low, high)

        data_with_predictions = _reassemble_predictions(
            original_data=balanced_data,
            predictions_stacked=res_pred_tensor.squeeze(-1).cpu(),
            group_col=self.group_col_,
            unique_groups=groups_stacked,
            target_col=self.target_col_,
            date_col=self.date_col_
        )

        # --- 4. Reassembly ---
        est_col = f'Estimated{self.target_col_}'
        adapted_col = f"AdaptedEstimated{target_col_bm}"
        if self.target_col_ == target_col_bm:
            data_with_predictions[adapted_col] = data_with_predictions[est_col]
        else:
            data_with_predictions[adapted_col] = data_with_predictions[est_col_name] + data_with_predictions[est_col].fillna(0)

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

            data_with_predictions = self._forecast_frame(predictions_stacked)
            adapted_col = f"AdaptedEstimated{target_col_bm}"

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
                horizon_steps=self.horizon_steps_,
                clip_to_train_range=self.clip_to_train_range_
            )
            
            final_adaptive_model_internal = StaticTAM(
                formula=self.adaptive_model_.formula_,
                group_col=self.base_model_.group_col_ if self.base_model_ is not None else self.group_col_,
                date_col=self.base_model_.date_col_ if self.base_model_ is not None else self.date_col_,
                _internal_effects_list=optimal_effects_list,
                _internal_features_config=self.adaptive_model_.features_config_
            )
            
            final_model.adaptive_model_ = final_adaptive_model_internal
            final_model.target_col_ = self.target_col_

            final_model.predict_online(data_val)
            
            return final_model
#: </grid_search_adaptive>