# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

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
from ._data import _reassemble_predictions
from ._dispatcher import smart_solve

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
    def _get_chunked_predictions(self, x_data: torch.Tensor) -> torch.Tensor:
        r"""
        Memory-safe chunked computation of Phi.theta.
        
        Dynamically limits the number of groups processed simultaneously 
        to prevent Out-Of-Memory (OOM) crashes on large multi-entity datasets.
        """
        total_groups = x_data.shape[0]
        num_samples = x_data.shape[1]
        run_device = TORCH_DEVICE
        
        # Build a minimal dummy matrix to gauge memory footprint
        dummy_x = x_data[0:1, 0:1, :].to(run_device)
        dummy_phi = self._build_design_matrix(dummy_x)
        total_d = dummy_phi.shape[-1]
        del dummy_x, dummy_phi
        
        available_bytes = hw.get_available_memory()
        allocatable_bytes = available_bytes * 0.8
        bytes_per_group_full_n = num_samples * total_d * 8 * 3.0 
        safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1
        
        beta = self.coefficients_.to(run_device)
        predictions = []
        g_start = 0

        while g_start < total_groups:
            g_end = min(g_start + safe_group_batch, total_groups)
            current_sub_batch_size = g_end - g_start
            
            try:
                x_chunk = x_data[g_start:g_end, :, :].to(run_device)
                phi_chunk = self._build_design_matrix(x_chunk)
                
                beta_chunk = beta[g_start:g_end] 
                y_hat_chunk = _predict_from_coeffs(phi_chunk, beta_chunk)
                predictions.append(y_hat_chunk.cpu())
                
                del x_chunk, phi_chunk, y_hat_chunk, beta_chunk
                g_start += current_sub_batch_size
                
            except (torch.OutOfMemoryError, MemoryError):
                if safe_group_batch > 1:
                    safe_group_batch, run_device = hw.handle_oom(
                        current_batch=safe_group_batch, 
                        device=run_device, 
                        context="predictive group reduction", 
                        allow_cpu_fallback=True
                    )
                    beta = self.coefficients_.to(run_device)
                    continue
                else:
                    raise RuntimeError("A single full group exceeds available physical memory during prediction.")
                    
        del beta
        hw.empty_cache()
        return torch.cat(predictions, dim=0)
#: </predictive_chunking>

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
            
        num_samples_train = x_train.shape[1]

        #  Static Matrix Construction
        penalty_M_star_M = self._build_penalty_matrix()
        loss_L_star_L = self._build_loss_matrix()

        #  Dynamic Memory Routing & System Resolution
        self.coefficients_ = smart_solve(
            x_data=x_train,
            y_data=y_train,
            effects_list=self.effects_list_,
            penalty_matrix=penalty_M_star_M,
            loss_matrix=loss_L_star_L,
            num_samples=num_samples_train
        )
        
        return self
#: </fit_method>

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

        #  Prediction
        predictions_stacked = self._get_chunked_predictions(x_predict)

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