# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Abstract base class for TAM models.

This module defines the `BaseTAM` class, which orchestrates the core optimization 
logic. It creates a standardized API (`fit`, `predict`) that relies on 
solving a per-group, weighted, and regularized linear system.
"""

import torch
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from tam.common.utils import (
    TORCH_DEVICE, _check_features, _balance_groups, 
    _ensure_dummies, _cleanup_dummies
)
from tam.common.hardware import hw
from ._math import _predict_from_coeffs
from ._data import _reassemble_predictions, _check_known_groups, _coefficients_of_groups
from ._dispatcher import smart_solve

# Outer refits of the means when a family estimates a dispersion (negative binomial): it converges in a few.
_DISPERSION_REFITS = 10

def _concat_states(states: list):
    r"""Concatenates the per-chunk states of one effect along the group axis (tensors and nested dicts; other values: the last chunk's)."""
    first = states[0]
    if isinstance(first, dict):
        return {k: _concat_states([s[k] for s in states]) for k in first}
    if torch.is_tensor(first) and first.dim() > 0:
        return torch.cat(states, dim=0)
    return states[-1]


def _index_state(state, index):
    r"""The rows ``index`` (a slice or an index tensor) along the group axis of a state."""
    if isinstance(state, dict):
        return {k: _index_state(v, index) for k, v in state.items()}
    if torch.is_tensor(state) and state.dim() > 0:
        return state[index]
    return state


#: <class_def>
class BaseTAM(ABC):
    r"""
    Abstract Base Class for TAM models.

    Defines the skeleton for training and prediction. Subclasses (e.g., `StaticTAM`)
    must implement the abstract methods to define how design matrices (Phi),
    penalty matrices (P), and loss matrices (L) are constructed.

    """
    
    def __init__(self):
        # --- Fitted Model Attributes ---
        self.coefficients_: Optional[torch.Tensor] = None
        self.norm_params_: Optional[Dict] = None
        self.unique_groups_: Optional[List] = None
        self.effects_list_: Optional[List] = None

        # --- Configuration Attributes ---
        self.features_config_: Optional[Dict] = None
        self.group_col_: Optional[str] = None
        self.target_col_: Optional[str] = None
        self.date_col_: Optional[str] = None

    @abstractmethod
    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        r"""
        Transforms the raw DataFrame into normalized, 3D-stacked tensors.

        Args:
            data: Input DataFrame (must be pre-balanced).
            target_col: Name of the target column (None for inference).

        Returns:
            Tuple containing:
            - Feature tensor (x_stacked)
            - Target tensor (y_stacked) or None
            - List of unique groups processed
        """
        raise NotImplementedError

    @abstractmethod
    def _build_design_matrix(self, x_data: torch.Tensor) -> torch.Tensor:
        r"""
        Constructs the design matrix Phi from input features.

        Args:
            x_data: Input feature tensor (n_groups, n_samples, n_features).

        Returns:
            Design matrix (n_groups, n_samples, n_total_coeffs).
        """
        raise NotImplementedError

    @abstractmethod
    def _build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Constructs the global regularization matrix P (or M*M).

        Returns:
            Penalty matrix (n_total_coeffs, n_total_coeffs).
        """
        raise NotImplementedError

    @abstractmethod
    def _build_loss_matrix(self) -> torch.Tensor:
        r"""
        Constructs the loss weighting matrix L*L.

        Returns:
            Loss matrix (n_samples, n_samples).
        """
        raise NotImplementedError
#: </class_def>

#: <predictive_chunking>
    def _get_chunked_predictions(self, x_data: torch.Tensor, coefficients: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Memory-safe chunked computation of Phi.theta.
        
        Dynamically limits the number of groups processed simultaneously 
        to prevent Out-Of-Memory (OOM) crashes on large multi-entity datasets.
        """
        total_groups = x_data.shape[0]
        num_samples = x_data.shape[1]
        run_device = TORCH_DEVICE

        coefficients = self.coefficients_ if coefficients is None else coefficients
        if total_groups <= 1:
            safe_group_batch = 1
        else:
            if coefficients is not None and coefficients.dim() >= 2:
                total_d = int(coefficients.shape[1])
            elif coefficients is not None:
                total_d = int(coefficients.shape[-1])
            else:
                dummy_x = x_data[0:1, 0:1, :].to(run_device)
                self._state_chunk_ = slice(0, 1)
                dummy_phi = self._build_design_matrix(dummy_x)
                total_d = dummy_phi.shape[-1]
                del dummy_x, dummy_phi

            available_bytes = hw.get_available_memory()
            allocatable_bytes = available_bytes * 0.8
            bytes_per_group_full_n = num_samples * total_d * 8 * 3.0
            safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1

        beta = coefficients.to(run_device)
        predictions = []
        g_start = 0

        while g_start < total_groups:
            g_end = min(g_start + safe_group_batch, total_groups)
            current_sub_batch_size = g_end - g_start
            
            try:
                x_chunk = x_data[g_start:g_end, :, :].to(run_device)
                self._state_chunk_ = slice(g_start, g_end)
                beta_chunk = beta[g_start:g_end]
                y_hat_chunk = self._predict_chunk_from_x(x_chunk, beta_chunk)
                offset_chunk = self._offset_of(x_chunk)
                if offset_chunk is not None:
                    y_hat_chunk = y_hat_chunk + offset_chunk.to(device=y_hat_chunk.device, dtype=y_hat_chunk.dtype)
                predictions.append(y_hat_chunk.cpu())

                del x_chunk, y_hat_chunk, beta_chunk
                g_start += current_sub_batch_size
                
            except (torch.OutOfMemoryError, MemoryError):
                if safe_group_batch > 1:
                    safe_group_batch, run_device = hw.handle_oom(
                        current_batch=safe_group_batch, 
                        device=run_device, 
                        context="predictive group reduction", 
                        allow_cpu_fallback=True
                    )
                    beta = coefficients.to(run_device)
                    continue
                else:
                    raise RuntimeError("A single full group exceeds available physical memory during prediction.")
                    
        self._state_chunk_ = None
        del beta
        hw.empty_cache()
        return torch.cat(predictions, dim=0)
#: </predictive_chunking>

    def _offset_of(self, x_data: torch.Tensor) -> Optional[torch.Tensor]:
        r"""The fixed part of the linear predictor, shape (n_groups, n_samples, 1), or None when the model has none (the default)."""
        return None

#: <pwls_atom>
    def _solve_pwls_step(
        self,
        x_data: torch.Tensor,
        z_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve one penalized weighted least-squares system: the atom of every TAM fit.

        Solves (Phi.T * W * Phi + n*S) * theta = Phi.T * W * z, where z_target is the (working) response
        and weights is an optional per-observation weight w of shape (n_groups, n_samples, 1).
        weights=None is the plain penalized solve (W = I). The reweighting loop calls this repeatedly with
        updated (z, w); a plain Gaussian fit calls it once.

        Args:
            x_data: feature tensor (n_groups, n_samples, n_features).
            z_target: (working) response (n_groups, n_samples, 1).
            weights: optional per-observation weights (n_groups, n_samples, 1), non-negative.
        """
        penalty_M_star_M = self._build_penalty_matrix()
        loss_L_star_L = self._build_loss_matrix()
        offset = self._offset_of(x_data)
        if offset is not None:
            z_target = z_target - offset.to(device=z_target.device, dtype=z_target.dtype)  # the offset has coefficient 1: solve for the rest
        return smart_solve(
            x_data=x_data,
            y_data=z_target,
            effects_list=self.effects_list_,
            penalty_matrix=penalty_M_star_M,
            loss_matrix=loss_L_star_L,
            num_samples=x_data.shape[1],
            sample_weights=weights,
        )
#: </pwls_atom>

    def _predict_chunk_from_x(self, x_chunk: torch.Tensor, beta_chunk: torch.Tensor) -> torch.Tensor:
        r"""Evaluates ``Phi(x_chunk) @ beta_chunk`` for one group chunk (overridden by StaticTAM for streamed contraction)."""
        phi_chunk = self._build_design_matrix(x_chunk)
        return _predict_from_coeffs(phi_chunk, beta_chunk)

    def _stacked_column(self, data: pd.DataFrame, column: str) -> torch.Tensor:
        r"""A column of ``data`` stacked like the target (same groups, same order): shape (n_groups, n_samples, 1)."""
        raise NotImplementedError

    def _solve_prepared(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        design_matrix: Optional[torch.Tensor] = None,
        initial_theta: Optional[torch.Tensor] = None,
        max_dispersion_refits: int = _DISPERSION_REFITS,
    ) -> None:
        r"""Solves the coefficients on prepared tensors: one exact solve for ``l2`` (weighted when ``weights`` is given), else the reweighted loop (and the dispersion refits)."""
        reweighting_strategy = getattr(self, "_reweighting_strategy_", None)
        if reweighting_strategy is None:
            self.coefficients_ = self._solve_pwls_step(x_train, y_train, weights=weights)
        else:
            if weights is not None:
                raise ValueError("Observation weights are available for the squared loss only.")
            from .statistics.estimation._reweighting import reweighted_penalized_fit, _linear_predictor, dense_design
            validate = getattr(reweighting_strategy, "validate_target", None)
            if validate is not None:
                validate(y_train)
            irls = dict(max_iter=getattr(self, "_irls_max_iter_", 25), tol=getattr(self, "_irls_tol_", 1e-6))
            if design_matrix is None:
                design_matrix = dense_design(self, x_train)  # built once, shared by every pass and every dispersion refit
            self.coefficients_ = reweighted_penalized_fit(
                self,
                x_train,
                y_train,
                reweighting_strategy,
                design_matrix=design_matrix,
                initial_theta=initial_theta,
                **irls,
            )
            # A family with a dispersion (negative binomial, Tweedie) estimates it from the fitted means and refits the means
            # with the new variance until it stops moving.
            estimate = getattr(reweighting_strategy, "estimate_dispersion", None)
            if estimate is not None:
                y_fit = y_train.to(torch.get_default_dtype())
                for _ in range(max_dispersion_refits):
                    mean = reweighting_strategy.inverse_link(_linear_predictor(self, x_train, self.coefficients_, design_matrix))
                    if not estimate(y_fit, mean):
                        break
                    self.coefficients_ = reweighted_penalized_fit(
                        self,
                        x_train,
                        y_train,
                        reweighting_strategy,
                        design_matrix=design_matrix,
                        initial_theta=self.coefficients_,
                        **irls,
                    )
        self.dispersion_ = getattr(reweighting_strategy, "dispersion", None)

#: <fit_method>
    def fit(self, data_train: pd.DataFrame) -> 'BaseTAM':
        r"""
        Fits the analytical model to the training data.

        Solves the regularized linear system for each group defined by `group_col_`.
        Delegates memory-safe evaluation and sparse Conjugate Gradient (CG) routing 
        to the mathematical dispatcher.
        
        Args:
            data_train: Training DataFrame containing features, target, and optional grouping columns.
        
        Returns:
            self: The fitted model instance.
            
        Raises:
            RuntimeError: If the model is not initialized with a formula/config.
            ValueError: If training data is empty after processing.
        """
        # --- Initialization Check (Relaxed for Transitivity) ---
        if not all([self.features_config_, self.target_col_]):
             raise RuntimeError("Model configuration incomplete. Ensure formula is set.")
        
        # --- Transitivity Shield: Inject Dummies ---
        # Ensures that cross-sectional or single time-series data can mathematically 
        # map to the 3D grouped tensors required by the GPU solvers.
        data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
        
        #  Data Preparation
        required_cols = self.features_config_['features'] + [self.group_col_, self.target_col_, self.date_col_]
        _check_features(dataset=data_train, required_features=required_cols)
        
        # Balance groups (drop to smallest size for batching)
        _, balanced_data = _balance_groups(
            dataset=data_train, 
            group_col=self.group_col_, 
            date_col=self.date_col_, 
            method="drop"
        )

        x_train, y_train, self.unique_groups_ = self._prepare_data(
            balanced_data, target_col=self.target_col_
        )
        
        if x_train.shape[0] == 0:
            raise ValueError("No valid training data found after balancing.")
            
        weights = None
        weight_column = getattr(self, "_weight_col_", None)
        if weight_column is not None:
            weights = self._stacked_column(balanced_data, weight_column)
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool((weights > 0).any()):
                raise ValueError(f"The observation weights in '{weight_column}' must be finite, non-negative and not all zero.")
        self._solve_prepared(x_train, y_train, weights)
        self._effect_states_ = self._collect_effect_states(x_train)

        return self
#: </fit_method>

    def _group_chunks(self, x_data: torch.Tensor) -> List[slice]:
        r"""Slices of groups that fit in memory together when the design matrix is built (the rule of the chunked prediction)."""
        total_groups, num_samples = x_data.shape[0], x_data.shape[1]
        total_d = max(1, sum(int(e.get_n_coeffs()) for e in (self.effects_list_ or [])))
        allocatable_bytes = hw.get_available_memory() * 0.8
        batch = max(1, int(allocatable_bytes // (num_samples * total_d * 8 * 3.0)))
        return [slice(start, min(start + batch, total_groups)) for start in range(0, total_groups, batch)]

    def _collect_effect_states(self, x_data: torch.Tensor) -> Dict[int, Dict]:
        r"""End-of-training causal state of every stateful effect, one row per group (training group order).

        The design matrix is built chunk by chunk (every chunk sets the ``_state_out`` of the stateful effects for its own
        groups); the states of the chunks are concatenated along the group axis. Returns ``{}`` when no effect is stateful, so the
        default path costs nothing.
        """
        stateful = [(i, e) for i, e in enumerate(self.effects_list_ or []) if getattr(e, "stateful", False) or hasattr(e, "_state_out")]
        if not stateful:
            return {}
        parts: Dict[int, list] = {i: [] for i, _ in stateful}
        for chunk in self._group_chunks(x_data):
            self._build_design_matrix(x_data[chunk])
            for i, e in stateful:
                state = getattr(e, "_state_out", None)
                if state is not None:
                    parts[i].append(state)
        return {i: _concat_states(states) for i, states in parts.items() if states}

    def compact(self) -> 'BaseTAM':
        r"""
        Drops what grows with the data and ``predict()`` does not need; forecasts are unchanged to the last bit.

        A fitted ``StaticTAM`` holds its coefficients only, so there is nothing to drop: the size does not depend on the number of training rows.

        Returns:
            self
        """
        return self

#: <predict_method>
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        r"""
        Generates predictions using the fitted coefficients.

        Automatically handles group alignment and temporal data padding (filling) 
        to ensure predictions are generated for all input rows simultaneously.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame containing the original data plus the estimated target column.
        
        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.coefficients_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # --- Transitivity Shield: Inject Dummies ---
        data = _ensure_dummies(data, self.group_col_, self.date_col_)

        #  Data Preparation
        required_cols = self.features_config_['features'] + [self.group_col_, self.date_col_]
        _check_features(dataset=data, required_features=required_cols)
        _check_known_groups(data, self.group_col_, self.unique_groups_)
        
        # Balance by 'fill' to preserve all rows
        mask, balanced_data = _balance_groups(
            dataset=data, 
            group_col=self.group_col_, 
            date_col=self.date_col_, 
            method="fill"
        )

        x_predict, _, unique_groups_pred = self._prepare_data(balanced_data, target_col=None)
        
        if self.unique_groups_ is None:
             raise RuntimeError("Training groups not found. Model state is corrupted.")

        carried = getattr(self, "_active_state_in_", None)
        if carried is not None:
            trained = list(self.unique_groups_)
            predicted = list(unique_groups_pred)
            if set(map(str, trained)) != set(map(str, predicted)):
                raise ValueError(f"continue_state=True needs the same groups as training: got {len(predicted)} groups, trained on {len(trained)}")
            position = {str(g): j for j, g in enumerate(trained)}
            order = torch.tensor([position[str(g)] for g in predicted], dtype=torch.long)
            self._active_state_in_ = {i: _index_state(s, order) for i, s in carried.items()}

        #  Prediction
        predictions_stacked = self._get_chunked_predictions(
            x_predict, _coefficients_of_groups(self.coefficients_, unique_groups_pred, self.unique_groups_)
        )

        #  Reassembly
        data_with_predictions = _reassemble_predictions(
            original_data=balanced_data, 
            predictions_stacked=predictions_stacked.squeeze(-1),
            group_col=self.group_col_,
            unique_groups=unique_groups_pred,
            target_col=self.target_col_,
            date_col=self.date_col_
        )

        # --- Transitivity Cleanup ---
        # Strip away the structural dummies before returning the DataFrame to the user
        return _cleanup_dummies(data_with_predictions[mask], self.group_col_, self.date_col_)
#: </predict_method>