# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Feature Engineer for Automated TAM (AutoTAM).

This module applies mathematical transformations to enrich the dataset while 
strictly safeguarding the numerical stability of the downstream primal solver.

A critical challenge when training Generalized Additive Models (GAMs) via 
Penalized Iteratively Reweighted Least Squares (P-IRLS) or Conjugate Gradient 
descent is design matrix singularity. To guarantee Krylov subspace convergence, 
this module enforces strict multicollinearity filtering on all augmented features. 

Furthermore, it utilizes stateful tracking to ensure that features generated and 
dropped during the training phase are identically replicated and purged during 
inference, strictly preventing Target Leakage and Train-Test Skew.
"""

#: <feature_engineer_imports>
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
#: </feature_engineer_imports>

#: <feature_engineer_class>
class FeatureEngineer:
    """
    Automated Data Augmentation and Feature Filtering module.
    
    Expands the predictive signal via chronological transformations (e.g., EWMA, 
    rolling windows) while safely managing multicollinearity and topological guardrails.
    """
    
#: <feature_engineer_init>
    def __init__(self, collinearity_threshold: float = 0.95):
        """
        Initializes the FeatureEngineer.

        Args:
            collinearity_threshold (float): The maximum allowed Pearson correlation 
                between augmented features. Features exceeding this threshold are 
                purged to guarantee the convergence of the Conjugate Gradient algorithm.
        """
        self.collinearity_threshold = collinearity_threshold
        
        self.generated_features: List[str] = []
        self.dropped_features: List[str] = []
        self.is_fitted: bool = False
        
        self.purge_log: List[Dict[str, Any]] = []
        self.do_not_smooth_features: Set[str] = set()
        self.learned_temporal_params: Dict[str, Dict[str, Any]] = {}
#: </feature_engineer_init>

#: <engineer_features_method>
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        metadata: Dict[str, Any], 
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Applies mathematical augmentations to the dataset and filters violating features.

        During the initial fit, it learns which features are highly collinear and drops them. 
        During subsequent calls, it identically drops the exact same features to maintain 
        structural parity with the trained models and prevent target leakage.

        Args:
            df (pd.DataFrame): The pre-cleaned dataset.
            config (Dict[str, Any]): Parsed formula configuration (targets, features).
            metadata (Dict[str, Any]): Pipeline metadata (time series flags, group cols).
            date_col (str, optional): The column representing time.

        Returns:
            pd.DataFrame: The structurally verified and augmented dataset.
        """
        data = df.copy()
        data = self._transform_string_into_categorical(data)
        
        targets = config.get("targets", [])
        target_col = targets[0] if targets else None
        is_time_series = metadata.get("is_time_series", False)
        group_col = metadata.get("group_col", None)
        
        if not self.is_fitted:
            user_categoricals = metadata.get("categorical_features", [])
            self.do_not_smooth_features.update(user_categoricals)
            self._detect_protected_features(data, date_col, target_col)
            
        original_cols = list(data.columns)
        numeric_cols = [c for c in data.select_dtypes(include=[np.number]).columns 
                        if c not in targets and c not in [date_col, group_col]]
        
        if is_time_series and date_col:
            data = data.sort_values(by=date_col)
            
            base_lags = list(config.get("lags", {}).values())
            max_base_lag = max(base_lags) if base_lags else 0
            max_safe_footprint = max(1, (len(data) // 4) - max_base_lag)
            
            data = self._create_temporal_features(data, numeric_cols, group_col, date_col, max_safe_footprint)
            
        data = self._create_cross_interactions(data, numeric_cols)
        
        current_cols = list(data.columns)
        new_cols = [c for c in current_cols if c not in original_cols]
        
        if not self.is_fitted:
            self.generated_features = new_cols
            data = self._filter_collinearity(data, protected_cols=original_cols, target_col=target_col)
            self.is_fitted = True
        else:
            cols_to_drop = [c for c in self.dropped_features if c in data.columns]
            if cols_to_drop:
                data.drop(columns=cols_to_drop, inplace=True)
                
        return data
#: </engineer_features_method>

#: <feature_engineer_guardrails>
    def _detect_protected_features(self, df: pd.DataFrame, date_col: Optional[str] = None, target_col: Optional[str] = None) -> None:
        """
        Dynamically detects features that should never be smoothed (Temporal, Categorical, or Lags)
        using mathematical topologies rather than hardcoded string names.
        """
        protected = set()
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs().values
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            dynamic_corr_threshold = min(0.99, np.nanpercentile(upper_triangle, 99))
        else:
            dynamic_corr_threshold = 0.95
        
        if date_col and date_col in df.columns:
            try:
                dt_series = pd.to_datetime(df[date_col])
                temporal_signals = [
                    dt_series.dt.second, dt_series.dt.minute, dt_series.dt.hour,
                    dt_series.dt.dayofweek, dt_series.dt.day, dt_series.dt.month,
                    dt_series.dt.quarter, dt_series.dt.dayofyear, dt_series.dt.year
                ]
                
                for col in df.columns:
                    if col in [date_col, target_col] or not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    
                    for sig in temporal_signals:
                        if sig.nunique() > 1 and df[col].nunique() > 1:
                            corr = abs(df[col].corr(sig))
                            if corr > dynamic_corr_threshold:
                                protected.add(col)
                                break
            except Exception:
                pass

        for col in df.columns:
            if col in [date_col, target_col] or col in protected: 
                continue
                
            if df[col].nunique() < min(25, max(3, len(df) * 0.05)):
                if pd.api.types.is_integer_dtype(df[col]) or (df[col] % 1 == 0).all():
                    protected.add(col)

        if target_col and target_col in df.columns:
            try:
                target_var = round(float(df[target_col].var()), 4)
                target_mean = round(float(df[target_col].mean()), 2)
                
                for col in df.columns:
                    if col in [date_col, target_col] or col in protected: 
                        continue
                        
                    col_var = round(float(df[col].var()), 4)
                    col_mean = round(float(df[col].mean()), 2)
                    
                    if col_var == target_var and col_mean == target_mean:
                        protected.add(col)
            except Exception:
                pass

        self.do_not_smooth_features.update(protected)

    def _should_augment(self, feature_name: str) -> bool:
        """Safe check to verify if a feature can be smoothed."""
        return feature_name not in self.do_not_smooth_features
#: </feature_engineer_guardrails>

#: <feature_engineer_transformations>
    def _create_temporal_features(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str], 
        group_col: Optional[str], 
        date_col: Optional[str] = None,
        max_safe_footprint: int = 12
    ) -> pd.DataFrame:
        """
        Generates stateful chronological features (Rolling Means, EWMA).
        Safely groups operations by spatial entities (panel data) if applicable.
        Dynamically limits rolling windows based on the dataset's available degrees of freedom,
        and persistently locks the learned windows for accurate out-of-sample inference.
        """
        for col in numeric_cols:
            if not self._should_augment(col):
                continue
                
            if not self.is_fitted:
                sample_series = df[col] if not group_col else df[df[group_col] == df[group_col].iloc[0]][col]
                sample_series = sample_series.dropna()

                search_limit = min(50, max_safe_footprint, len(sample_series) // 3)
                window_size = 1
                
                if search_limit >= 2:
                    max_ac = 0.0
                    for lag in range(2, search_limit + 1): 
                        ac = abs(sample_series.autocorr(lag=lag))
                        if pd.notna(ac) and ac > max_ac:
                            max_ac = ac
                            window_size = lag

                window_size = max(1, window_size)
                dynamic_alpha = max(0.01, min(0.30, 2.0 / (window_size + 1.0)))
                
                self.learned_temporal_params[col] = {
                    "window_size": window_size,
                    "dynamic_alpha": dynamic_alpha
                }
            else:
                if col in self.learned_temporal_params:
                    window_size = int(self.learned_temporal_params[col]["window_size"])
                    dynamic_alpha = float(self.learned_temporal_params[col]["dynamic_alpha"])
                else:
                    continue
            
            window_label = f"{window_size}_steps"
            
            roll_name = f"{col}_rolling_mean_{window_label}"
            ewma_name = f"{col}_ewma_alpha{int(dynamic_alpha*100)}"

            if group_col and group_col in df.columns:
                df[roll_name] = df.groupby(group_col)[col].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
                df[ewma_name] = df.groupby(group_col)[col].transform(lambda x: x.ewm(alpha=dynamic_alpha, adjust=False, ignore_na=True).mean())
            else:
                df[roll_name] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[ewma_name] = df[col].ewm(alpha=dynamic_alpha, adjust=False, ignore_na=True).mean()
                
        return df

    def _create_cross_interactions(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Generates static mathematical interactions between the primary continuous features.
        Includes a safety constant to prevent zero-division errors.
        """
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2].abs() + 1e-6)
        return df
    
    def _transform_string_into_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts object and string columns into integer-based categorical codes.
        Instantly protects these columns from temporal smoothing, regardless of cardinality.
        """
        for col in df.columns:
            if df[col].dtype == 'str' or df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                self.do_not_smooth_features.add(col)
                df[col] = df[col].astype('category').cat.codes
                df[col] = df[col].astype(np.int32)
        return df
#: </feature_engineer_transformations>

#: <feature_engineer_filter>
    def _filter_collinearity(self, df: pd.DataFrame, protected_cols: List[str], target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the correlation matrix to filter out highly collinear generated features.
        Tracks dropped features internally to enforce consistency during prediction.
        """
        cols_to_check = [c for c in protected_cols + self.generated_features if c in df.columns]
        numeric_check_cols = df[cols_to_check].select_dtypes(include=[np.number]).columns
        
        corr_matrix = df[numeric_check_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = []
        for col in upper.columns:
            if col in self.generated_features:
                max_corr = upper[col].max()
                if max_corr > self.collinearity_threshold:
                    correlated_with = upper[col].idxmax()
                    col_to_drop = col

                    if target_col and target_col in df.columns:
                        corr_new = abs(df[col].corr(df[target_col]))
                        corr_old = abs(df[correlated_with].corr(df[target_col]))
                        
                        if corr_new > corr_old and correlated_with not in protected_cols:
                            col_to_drop = correlated_with

                    to_drop.append(col_to_drop)
                    
                    self.purge_log.append({
                        "Dropped_Feature": col_to_drop,
                        "Correlated_With": correlated_with if col_to_drop == col else col,
                        "Pearson_Correlation": round(max_corr, 4)
                    })
                
        if to_drop:
            to_drop = list(set(to_drop))
            self.dropped_features.extend(to_drop)
            df.drop(columns=to_drop, inplace=True, errors='ignore')
            
        return df
#: </feature_engineer_filter>
#: </feature_engineer_class>