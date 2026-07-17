# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France) et Sorbonne Université
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-License-Identifier: LGPL-3.0-or-later
# Authors : Yann Allioux, Nathan Doumèche

"""
Implements the core Additive TAM (StaticTAM) model.

This module defines the primary user-facing class, `StaticTAM`.
It orchestrates the assembly of effects defined in the formula, manages
data preparation, solves the primal optimization problem, and provides
methods for hyperparameter tuning and model interpretation.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
import torch
import pandas as pd
import numpy as np 

from tam.common.utils import (
    TORCH_DEVICE, _check_features, _balance_groups, parse_formula_to_terms, 
    _ensure_dummies, _cleanup_dummies
)
from ._base import BaseTAM
from ._data import (
    _fit_normalization_params,
    _transform_data_stacked,
    _reassemble_decomposed_predictions
)

from ._dispatcher import smart_solve, smart_evaluate_rmse, smart_decompose
from ._dispatcher_gcv import smart_solve_gcv

from .spectrum import (
    BaseEffect, 
    OffsetEffect, LinearEffect, FourierEffect, SplineEffect,
    CategoricalEffect, ChebyshevEffect, WaveletEffect,
    NeuralEffect, RBFEffect, UniversalPhysicsEffect,
    TensorProductEffect, TreeEffect, LinearTreeEffect,
    create_effects_from_parsed_terms,
    build_phi_from_effects,
    build_penalty_from_effects
)

class StaticTAM(BaseTAM):
    """
    Implements the formula-driven Additive TAM model.

    This class builds interpretable, physics-informed models by summing
    various functional effects (Linear, Splines, Fourier, Neural, etc.) defined
    via an R-style formula.

    The model minimizes the penalized empirical risk by solving the regularized
    normal equations in the primal space. See the theoretical documentation for
    exact mathematical proofs.

    Attributes:
        effects_list_ (List[BaseEffect]): The list of instantiated effect objects.
        coefficients_ (torch.Tensor): The fitted coefficients (batch, n_coeffs, 1).
        norm_params_ (dict): Parameters used for data normalization.
    """
    
#: <init_additive>
    def __init__(
        self,
        formula: str,
        group_col: str = None,
        date_col: str = None,
        default_alpha_p: float = -9.0,
        _internal_effects_list: Optional[List[BaseEffect]] = None,
        _internal_features_config: Optional[dict] = None 
    ):
        """
        Initializes the StaticTAM model.

        Args:
            formula: R-style formula defining the model structure
                     (e.g., "Y ~ s(x) + l(t)").
            group_col: Column name used for grouping data (e.g., 'ID').
            date_col: Column name for time indexing. Defaults to None, in which
                case rows are tensorized in whatever order they were passed in.
                For formulas with temporal terms (lags, trend, or any term whose
                penalty assumes row-adjacency means chronological adjacency),
                you must pass date_col explicitly to guarantee correct results
                on data that isn't already sorted by time.
            default_alpha_p: Default log10(lambda_p) regularization strength.
            _internal_effects_list: (Internal) Used for restoring state during grid search.
            _internal_features_config: (Internal) Used for restoring state during grid search.
        """
        super().__init__()
        
        self.effects_list_ = []
        self.formula_ = formula 
        self.default_alpha_p_ = default_alpha_p
        self.group_col_ = group_col or "__dummy_group__"
        self.date_col_ = date_col or "__dummy_date__"
        
        if _internal_effects_list:
            self.effects_list_ = _internal_effects_list
            self.features_config_ = _internal_features_config
            self.is_grid_search_template_ = False

        elif formula:
            self.target_col_, self.parsed_terms_ = parse_formula_to_terms(formula)
            real_features = self._extract_recursive_features(self.parsed_terms_)
            self.features_config_ = { "features": real_features }
            
            self.is_grid_search_template_ = False
            
            try:
                # Attempt standard instantiation; valid string hyperparams will pass.
                self.effects_list_ = create_effects_from_parsed_terms(
                    self.parsed_terms_, 
                    token_values={}, 
                    default_alpha_p=self.default_alpha_p_
                )
            except Exception as e:
                # Missing categorical counts are filled later in _prepare_data
                if isinstance(e, ValueError) and "requires 'n_cat'" in str(e):
                    pass
                else:
                    # If instantiation structurally fails and strings are present, it implies grid tokens are actively blocking the types.
                    has_str_vals = any(isinstance(v, str) for t in self.parsed_terms_ for v in t['params'].values())
                    if has_str_vals:
                        self.is_grid_search_template_ = True
                        print("Model initialized with Grid Search tokens. Use 'grid_search_fit()'.")
                    else:
                        raise e
        else:
            raise ValueError("`formula` must be provided to initialize StaticTAM.")
#: </init_additive>

    def _extract_recursive_features(self, terms: List[Dict]) -> List[str]:
        """
        Recursively extracts physical feature names, preserving the order of appearance.
        Dives into Tensor Products and parses 'others' arguments.
        """
        features_ordered = []
        seen = set()

        def add_feature(f_name: str):
            if f_name not in seen:
                seen.add(f_name)
                features_ordered.append(f_name)

        for term in terms:
            if term['type'] != 'te':
                add_feature(term['feature'])
            
            if 'others' in term['params'] and isinstance(term['params']['others'], str):
                others_str = term['params']['others']
                extras = [s.strip() for s in others_str.split('|') if s.strip()]
                for extra in extras:
                    add_feature(extra)

            if 'slope' in term['params'] and isinstance(term['params']['slope'], str):
                add_feature(term['params']['slope'].strip())

            if term['type'] == 'te':
                for sub_term_raw in term['params'].keys():
                    if not sub_term_raw: continue
                    try:
                        _, sub_parsed = parse_formula_to_terms(f"DUMMY ~ {sub_term_raw}")
                        sub_features = self._extract_recursive_features(sub_parsed)
                        for sub_f in sub_features:
                            add_feature(sub_f)
                    except Exception:
                        pass
        
        return features_ordered
    
    def _get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Safely infers the number of categories (max + 1) for features."""
        info = {}
        for col in self.features_config_.get('features', []):
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                info[col] = int(data[col].max(skipna=True)) + 1
        return info
    
    def summary(self) -> pd.DataFrame:
        """
        Generates a structured summary of the model architecture.
        
        Returns:
            pd.DataFrame: A table describing each effect, its complexity (degrees of freedom),
                          structural parameters (e.g., knots, PDE weights), and regularization strength.
        """
        if not self.effects_list_:
            return pd.DataFrame()
            
        summary_data = []
        
        for effect in self.effects_list_:
            eff_type_raw = effect.__class__.__name__.replace('Effect', '')
            name = effect.feature_name
            complexity = effect.get_n_coeffs()
            details = ""
            
            if isinstance(effect, OffsetEffect):
                details = "Global Bias"
            elif isinstance(effect, LinearEffect):
                details = "Slope (Ridge)"
            elif isinstance(effect, FourierEffect):
                details = f"m={effect.m}, Sobolev(s={effect.s})"
            elif isinstance(effect, SplineEffect) and not isinstance(effect, UniversalPhysicsEffect):
                details = f"Knots={effect.n_knots}, Diff(d={effect.penalty_order})"
            elif isinstance(effect, CategoricalEffect):
                details = f"Cats={effect.n_categories}, Topo={effect.topology}"
            elif isinstance(effect, NeuralEffect):
                details = f"Neurons={effect.n_neurons}, Act={effect.activation}"
            elif isinstance(effect, UniversalPhysicsEffect):
                ops = [f"{k}={v}" for k, v in effect.diff_weights.items()]
                details = f"Basis={effect.basis_type}, PDE[{', '.join(ops)}]"
                eff_type_raw = "Physics"
            elif isinstance(effect, WaveletEffect):
                details = f"Scales={effect.n_scales}, Locs={effect.n_locations}"
            elif isinstance(effect, RBFEffect):
                details = f"Centers={effect.n_centers}" + (f", Gamma={effect.gamma:.2f}" if effect.gamma else "")
            elif isinstance(effect, ChebyshevEffect):
                details = f"Deg={effect.degree}, Sobolev(s={effect.s})"
            elif isinstance(effect, TensorProductEffect):
                details = f"Interaction ({len(effect.effects)} terms)"
            elif isinstance(effect, TreeEffect):
                details = f"Trees={effect.n_trees}, MaxDepth={effect.max_depth}"
            else:
                details = "Custom"

            lambda_p_log = np.log10(effect.lambda_p) if effect.lambda_p > 0 else -np.inf

            summary_data.append({
                "Feature": name,
                "Type": eff_type_raw,
                "Complexity (D)": complexity,
                "Structure / Params": details,
                "Reg (log10)": round(lambda_p_log, 2)
            })
            
        return pd.DataFrame(summary_data)

    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: Optional[str] = None,
        ignore_template_check: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Prepares data for the additive model (normalization and stacking)."""
        
        if self.coefficients_ is None and self.is_grid_search_template_ and not ignore_template_check:
            raise RuntimeError("Model has grid tokens. Call 'grid_search_fit()' first.")
                             
        current_target_col = target_col if target_col is not None else self.target_col_

        if not self.effects_list_ and not self.is_grid_search_template_:
            data_info = self._get_data_info(data)
            self.effects_list_ = create_effects_from_parsed_terms(
                self.parsed_terms_, 
                token_values={}, 
                default_alpha_p=self.default_alpha_p_,
                data_info=data_info
            )

        if self.norm_params_ is None:
            self.norm_params_, self.unique_groups_ = _fit_normalization_params(
                data=data, 
                features=self.features_config_["features"], 
                group_col=self.group_col_
            )
            
        x_stacked, y_stacked = _transform_data_stacked(
            data=data, 
            features=self.features_config_["features"], 
            group_col=self.group_col_,
            norm_params=self.norm_params_,
            target_col=current_target_col, 
            unique_groups=self.unique_groups_,
            date_col=self.date_col_
        )
        
        if self.features_config_:
             feature_names = self.features_config_.get('features', [])
             for i, effect in enumerate(self.effects_list_):
                 if i > 0 and i <= len(feature_names):
                     effect.feature_name = feature_names[i-1]

        if torch.isnan(x_stacked).any():
            raise ValueError(
                "TAM [Data Error]: The input features (X) contain NaN values. "
                "Please clean or impute your dataset."
            )
            
        if target_col is not None and y_stacked is not None:
            if torch.isnan(y_stacked).any():
                raise ValueError(
                    f"TAM [Data Error]: The target column '{target_col}' "
                    "contains NaN values. Cannot proceed with optimization."
                )
        return x_stacked, y_stacked, self.unique_groups_

    def _build_design_matrix(self, x_data: torch.Tensor) -> torch.Tensor:
        """Builds the global design matrix."""
        feature_names = self.features_config_['features'] if self.features_config_ else None
        return build_phi_from_effects(x_data, self.effects_list_, feature_columns=feature_names)

    def _build_penalty_matrix(self) -> torch.Tensor:
        """Builds the global penalty matrix."""
        return build_penalty_from_effects(self.effects_list_)

    def _build_loss_matrix(self) -> torch.Tensor:
        """Builds the Identity loss matrix (Standard Least Squares)."""
        return torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    
#: <decompose_pred>
    def decompose_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes the prediction into additive components per feature.

        Args:
            data: DataFrame containing input features.

        Returns:
            DataFrame with original data and additional 'effect_feature' columns.
        """
        data = _ensure_dummies(data, self.group_col_, self.date_col_)

        if self.coefficients_ is None:
            raise RuntimeError("Model must be fitted first.")

        required_cols = self.features_config_['features'] + [self.group_col_, self.date_col_]
        _check_features(dataset=data, required_features=required_cols)
        
        mask, balanced_data = _balance_groups(
            dataset=data, group_col=self.group_col_, date_col=self.date_col_, method="fill"
        )

        x_predict, _, _ = self._prepare_data(balanced_data)
        
        final_decomposed_effects = smart_decompose(x_predict, self.coefficients_, self.effects_list_)

        decomposed_df = _reassemble_decomposed_predictions(
            balanced_data, final_decomposed_effects, self.group_col_, self.unique_groups_, date_col=self.date_col_
        )

        return _cleanup_dummies(decomposed_df[mask], self.group_col_, self.date_col_)
#: </decompose_pred>
    
    # --- Grid Search (Coordinate Descent) Methods ---
    
    def _parse_grid_axes(self, grid_search_config: dict) -> Tuple[Dict[str, list], List[str]]:
        """Identifies which parameters are tokens to be optimized."""
        if not self.is_grid_search_template_:
            return {}, []
        
        tokens = {} 
        for term in self.parsed_terms_:
            for key, val in term['params'].items():
                if isinstance(val, str) and val in grid_search_config:
                    if val not in tokens:
                         tokens[val] = grid_search_config[val]

        token_names = self._extract_recursive_tokens(self.parsed_terms_, grid_search_config)
        search_axes = {tname: grid_search_config[tname] for tname in token_names}

        if token_names:
            print(f"Identified Coordinate Descent axes: {token_names}")
        return search_axes, token_names
    
    def _extract_recursive_tokens(self, terms: List[Dict], grid_config: Dict) -> List[str]:
        """Recursively extracts all token names used in the formula, diving into Tensor Products."""
        tokens = set()
        CONFIG_KEYS = {'basis', 'activation', 'act', 'topo', 'topology', 'op', 'others', 'extrapolate'}
        
        for term in terms:
            for key, val in term['params'].items():
                if isinstance(val, str) and val in grid_config:
                    tokens.add(val)
                if isinstance(key, str) and key in grid_config:
                    tokens.add(key)
            
            if term['type'] == 'te':
                for sub_term_raw in term['params'].keys():
                    if not sub_term_raw: continue
                    token_name_regex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)')
                    dummy_token_map = {t: t for t in grid_config.keys()}
                    
                    def find_token_name(match):
                        token = match.group(1)
                        if token in dummy_token_map:
                            return token
                        return ""
                        
                    found_tokens = [t for t in token_name_regex.findall(sub_term_raw) if t in grid_config]
                    tokens.update(found_tokens)
                    
        return sorted(list(tokens))

    def _build_combo_from_tokens(self, token_values: Dict[str, Any], data_info: Optional[Dict[str, Any]] = None) -> Dict:
        """Creates a concrete configuration from tokens."""
        effects_list = create_effects_from_parsed_terms(
            self.parsed_terms_,
            token_values,
            self.default_alpha_p_,
            data_info=data_info
        )
        return {
            "effects_list": effects_list,
            "token_values": token_values
        }

    def _evaluate_combination(
        self, combo: dict, 
        x_train: torch.Tensor, y_train: torch.Tensor,
        x_val: torch.Tensor, y_val: torch.Tensor,
        num_samples_train: int, loss_L_star_L: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluates a single hyperparameter configuration on validation data.
        
        Uses dynamic memory chunking to prevent OutOfMemory errors when 
        building the design matrix for massive models.
        """
        try:
            effects_list = combo["effects_list"]
            penalty_M_star_M = build_penalty_from_effects(effects_list)
            self.effects_list_ = effects_list 
            
            adaptive_coeffs_batch = smart_solve(
                x_data=x_train,
                y_data=y_train,
                effects_list=effects_list,
                penalty_matrix=penalty_M_star_M,
                loss_matrix=loss_L_star_L,
                num_samples=num_samples_train
            )
            
            rmse = smart_evaluate_rmse(x_val, y_val, adaptive_coeffs_batch, effects_list)
            
            return rmse, adaptive_coeffs_batch

        except Exception:
            return torch.tensor(float('inf')), None

#: <grid_search_logic>
    def grid_search_fit(
            self, 
            data_train: pd.DataFrame, 
            data_val: pd.DataFrame, 
            grid_search_config: dict
        ):
            """
            Finds optimal hyperparameters via Multi-Start Coordinate Descent.

            Args:
                data_train (pd.DataFrame): Training data.
                data_val (pd.DataFrame): Validation data for scoring.
                grid_search_config (dict): Dictionary mapping tokens to lists of values.

            Returns:
                StaticTAM: A new fitted model with optimal parameters.
            """
            print("--- Starting Grid Search (Multi-Start Coordinate Descent) ---")
            
            data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
            data_val = _ensure_dummies(data_val, self.group_col_, self.date_col_)
            
            temp_model = StaticTAM(self.formula_, self.group_col_, self.date_col_)
            temp_model.features_config_ = self.features_config_
            temp_model.target_col_ = self.target_col_
            
            required_cols_tr = self.features_config_['features'] + [self.group_col_, self.target_col_, self.date_col_]
            _check_features(dataset=data_train, required_features=required_cols_tr)
            _, balanced_data_train = _balance_groups(dataset=data_train, group_col=self.group_col_, date_col=self.date_col_, method="drop")
            
            required_cols_val = self.features_config_['features'] + [self.group_col_, self.target_col_, self.date_col_]
            _check_features(dataset=data_val, required_features=required_cols_val)
            _, balanced_data_val = _balance_groups(dataset=data_val, group_col=self.group_col_, date_col=self.date_col_, method="drop")
            
            x_train, y_train, unique_groups = temp_model._prepare_data(balanced_data_train, self.target_col_, ignore_template_check=True)
            x_val, y_val, _ = temp_model._prepare_data(balanced_data_val, self.target_col_, ignore_template_check=True)
            
            num_samples_train = x_train.shape[1]
            loss_L_star_L = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) 

            search_axes, token_names = self._parse_grid_axes(grid_search_config)
            data_info = self._get_data_info(balanced_data_train)

            if not token_names:
                print("No grid tokens found. Fitting single configuration.")
                combo = self._build_combo_from_tokens({}, data_info=data_info)
                rmse, coeffs = self._evaluate_combination(
                    combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                )
                optimal_params_combo, optimal_coeffs, min_global_rmse = combo, coeffs, rmse
            else:
                start_points = [
                    {"name": "Conservative", "tokens": {t: max(vals) if ('ap' in t or 'lambda_p' in t) else min(vals) for t, vals in search_axes.items()}},
                    {"name": "Median", "tokens": {t: vals[len(vals)//2] for t, vals in search_axes.items()}},
                    {"name": "Aggressive", "tokens": {t: min(vals) if ('ap' in t or 'lambda_p' in t) else max(vals) for t, vals in search_axes.items()}}
                ]

                global_best_rmse = float('inf')
                global_best_combo = None
                global_best_coeffs = None

                for strategy in start_points:
                    print(f"\n=== Strategy: {strategy['name']} Start ===")
                    current_best_tokens = strategy["tokens"].copy()
                    
                    complete_token_map = {}
                    for tname, tvals in search_axes.items():
                        if tname in current_best_tokens:
                            complete_token_map[tname] = current_best_tokens[tname]
                        else:
                            complete_token_map[tname] = tvals[0]
                    
                    current_best_tokens = complete_token_map.copy()

                    start_combo = self._build_combo_from_tokens(current_best_tokens, data_info=data_info)
                    current_rmse, current_coeffs = self._evaluate_combination(
                        start_combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                    )
                    current_optimal_combo = start_combo

                    if current_rmse >= float('inf'):
                        continue

                    cycle = 0
                    while True:
                        cycle += 1
                        has_improved_in_cycle = False
                        
                        for token_name in token_names:
                            best_val_for_axis = current_best_tokens[token_name]
                            original_val = best_val_for_axis
                            possible_values = search_axes[token_name]
                            
                            for value in possible_values:
                                if value == original_val: continue 
                                tokens_to_test = current_best_tokens.copy()
                                tokens_to_test[token_name] = value
                                
                                combo = self._build_combo_from_tokens(tokens_to_test, data_info=data_info)
                                rmse, coeffs = self._evaluate_combination(
                                    combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                                )

                                if rmse < current_rmse:
                                    current_rmse = rmse
                                    current_optimal_combo = combo
                                    current_coeffs = coeffs
                                    best_val_for_axis = value
                                    has_improved_in_cycle = True
                            
                            current_best_tokens[token_name] = best_val_for_axis
                        
                        print(f"  Cycle {cycle} | Current RMSE: {current_rmse:.4f}")
                        if not has_improved_in_cycle or cycle >= 5: break
                    
                    if current_rmse < global_best_rmse:
                        print(f"  >>> New Global Best found by {strategy['name']}! ({current_rmse:.4f})")
                        global_best_rmse = current_rmse
                        global_best_combo = current_optimal_combo
                        global_best_coeffs = current_coeffs

                optimal_params_combo = global_best_combo
                optimal_coeffs = global_best_coeffs
                min_global_rmse = global_best_rmse

            print("-" * 30)
            print(f"Grid search complete. Optimal Validation RMSE found: {min_global_rmse:.2f}")
            
            if optimal_params_combo is None:
                raise RuntimeError("Grid search failed to find any valid configuration.")

            print(f"Best tokens: {optimal_params_combo.get('token_values', 'N/A')}")

            model = self.__class__(
                formula=self.formula_, 
                group_col=self.group_col_,
                date_col=self.date_col_,
                _internal_effects_list=optimal_params_combo['effects_list'],
                _internal_features_config=self.features_config_ 
            )
            
            model.coefficients_ = optimal_coeffs
            model.target_col_ = self.target_col_ 
            model.norm_params_ = temp_model.norm_params_
            model.unique_groups_ = unique_groups

            return model
#: </grid_search_logic>
    
#: <auto_fit>
    def auto_fit(
        self, 
        data_train: pd.DataFrame, 
        alpha_p_bounds: Tuple[float, float] = (-30.0, 6.0),
        number_of_steps: int = 10,
        alpha_p_list: Optional[List[float]] = None,
        gamma: float = 1.4,
        verbose: bool = False
    ) -> 'StaticTAM':
        """
        Automatically trains the model by finding the optimal regularization per effect.
        
        Uses Generalized Cross Validation (GCV) with Multiple Smoothing Parameter (MSP)
        estimation to balance model fit (bias) and complexity (variance) without 
        requiring a validation set. 
        
        It utilizes a Coordinate Descent algorithm starting from the formula's initial values.
        
        Args:
            data_train: Training DataFrame.
            alpha_p_bounds: Search range constraints for local steps (e.g., -30.0 to 6.0).
            number_of_steps: Number of subdivisions within alpha_p_bounds to define the step size.
            alpha_p_list: Explicit list of log10(lambda_p) coordinates to test. If provided,
                        alpha_p_bounds and number_of_steps are ignored.
            gamma: Inflation factor for the effective degrees of freedom. Values > 1.0 
                   (typically 1.4 to 1.5) force smoother models, preventing GCV 
                   from overfitting when data has autocorrelated errors.
            verbose: print verbose details if True
            
        Returns:
            self: The trained model instance with optimal lambda_ps applied.
        """
        print("--- Auto-Fitting with MSP-GCV (Multiple Smoothing Parameters) ---")

        data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
        
        _, balanced_data = _balance_groups(
            dataset=data_train, group_col=self.group_col_, date_col=self.date_col_, method="drop"
        )
        x_train, y_train, self.unique_groups_ = self._prepare_data(
            balanced_data, target_col=self.target_col_
        )
        
        loss_L = self._build_loss_matrix()
        
        self.coefficients_, best_lambda_ps, gcv_score = smart_solve_gcv(
            x_data=x_train,
            y_data=y_train,
            effects_list=self.effects_list_,
            loss_matrix=loss_L,
            alpha_p_bounds=alpha_p_bounds,
            number_of_steps=number_of_steps,
            alpha_p_list=alpha_p_list,
            gamma=gamma,
            verbose=verbose
        )
        
        print(f"\nFinal GCV Score: {gcv_score:.4f}")
        print("Optimal lambda_ps found per effect:")
        for i, effect in enumerate(self.effects_list_):
            effect.lambda_p = best_lambda_ps[i]
            print(f" - {effect.feature_name}: {best_lambda_ps[i]:.2e} (log10 = {np.log10(best_lambda_ps[i]):.2f})")
        
        return self
#: </auto_fit>
    
    @staticmethod
    def show_syntax():
        """
        Displays a quick reference guide for formula syntax in the console.
        """

        data = [
            {
                "Token": "l(x)", 
                "Effect": "Linear / Ridge", 
                "Syntax Example": "l(trend, scaled=3.14)",
                "Specific Params": "scaled (data scaling)"
            },
            {
                "Token": "s(x)", 
                "Effect": "P-Spline", 
                "Syntax Example": "s(temp, k=12, deg=3, p=2)",
                "Specific Params": "k (knots), deg (degree), p (penalty order)"
            },
            {
                "Token": "f(x)", 
                "Effect": "Fourier", 
                "Syntax Example": "f(doy, m=6, s=1, cyclic=True)",
                "Specific Params": "m (harmonics), s (smoothness), cyclic (bool)"
            },
            {
                "Token": "c(x)", 
                "Effect": "Categorical", 
                "Syntax Example": "c(day, topo='ordinal', p_order=1)",
                "Specific Params": "n_cat (auto-inferred), topo ('nominal'/'ordinal'), p_order"
            },
            {
                "Token": "p(x)", 
                "Effect": "Chebyshev", 
                "Syntax Example": "p(time, deg=5, s=0)",
                "Specific Params": "deg (degree), s (smoothness)",
                "Usage": "Secular trends, stable extrapolation (Anti-Runge)"
            },
            {
                "Token": "rbf(x)", 
                "Effect": "RBF / Matérn Kernel", 
                "Syntax Example": "rbf(Lat, others='Lon', n_centers=50)",
                "Specific Params": "n_centers, gamma, nu, others (multivariate)"
            },
            {
                "Token": "n(x)", 
                "Effect": "Neural Network", 
                "Syntax Example": "n(Income, others='Age', n_neurons=500, seed=42)",
                "Specific Params": "n_neurons, act ('relu'/'cos'/'tanh'), n_hidden_layers, seed, others"
            },
            {
                "Token": "te(a, b)", 
                "Effect": "Tensor Product", 
                "Syntax Example": "te(s(Temp), f(Hour))",
                "Specific Params": "List of functional sub-effects"
            },
            {
                "Token": "w(x)", 
                "Effect": "Wavelet (Ricker)", 
                "Syntax Example": "w(signal, n_scales=5, n_locations=20)",
                "Specific Params": "n_scales, n_locations"
            },
            {
                "Token": "phys(x)", 
                "Effect": "Physics (PIKL/ODE)", 
                "Syntax Example": "phys(t, basis='spline', k=20, D2=1.0)",
                "Specific Params": "basis, k/n_coeffs, D{n} (derivative weights)"
            },
            {
                "Token": "pid(x)", 
                "Effect": "PID (Control)", 
                "Syntax Example": "pid(y_lag, w=7, d_pen=10.0)",
                "Specific Params": "w (rolling window), d_pen (derivative stiffness)"
            },
            {
                "Token": "t(x)", 
                "Effect": "Tree / Random Forest", 
                "Syntax Example": "t(x, n_trees=50, sp_alpha=0.5, split_strategy='quantile')",
                "Specific Params": "n_trees, max_depth, max_leaves, sp_alpha, split_strategy, seed, others"
            },
            {
                "Token": "lt(x)", 
                "Effect": "Linear Tree (VC Model)", 
                "Syntax Example": "lt(x_part, slope='x_slope', max_leaves=8, sp_alpha=0.5)",
                "Specific Params": "slope, max_depth, max_leaves, sp_alpha, split_strategy, seed, others"
            }
        ]
        
        # Using .fillna("") ensures the 'Usage' column doesn't print 'NaN' for effects without it.
        df = pd.DataFrame(data).fillna("")
        
        print("\n" + "=" * 120)
        print(" " * 45 + "TAM FORMULA SYNTAX GUIDE")
        print("=" * 120)
        print("GLOBAL ARGUMENTS (Applicable to ALL effects):")
        print("  * 'ap'          : log10(lambda_p) regularization strength (e.g., ap=-5 means lambda_p = 1e-5).")
        print("  * 'extrapolate' : OOD behavior ('continue', 'constant', 'linear', 'saturation').")
        print("                  Note: Defaults vary logically by effect topology (e.g., Splines default to 'linear',")
        print("                        Chebyshev defaults to 'saturation', Trees default to 'continue').\n")
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1200):
            print(df.to_string(index=False, justify='left'))
        print("=" * 120 + "\n")