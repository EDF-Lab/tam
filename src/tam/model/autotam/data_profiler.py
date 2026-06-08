#: <data_profiler_module_doc>
r"""
Data Profiler for Automated TAM (AutoTAM).

This module analyzes raw input data, establishes the cross-validation strategy, handles missing values, and applies strict outlier clipping.

To guarantee Reproducing Kernel Hilbert Space (RKHS) stability and completely prevent target leakage, this profiler operates as a stateful safeguard:
1. Group-Aware Outlier Bounding: Learns Interquartile Range (IQR) bounds independently for each entity in panel data to prevent global scaling discrepancies from destroying local entity signals.
2. Time-Based Resampling: Enforces continuous chronological frequencies (gap filling) to ensure that autoregressive lagged features strictly map to physical time.
3. Cold-Start Safe: Retains global statistical bounds as fallbacks for completely unseen entities during the prediction phase.
"""
#: </data_profiler_module_doc>

#: <data_profiler_imports>
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
#: </data_profiler_imports>

#: <data_profiler_class>
class DataProfiler:
    """
    Diagnostic, Evaluation, and Cleaning module for AutoTAM.
    
    Acts as the first line of defense before data reaches the mathematical engines. 
    It is strictly stateful: it learns the statistical properties of the training fold 
    and rigidly applies those exact properties to validation/test folds to simulate 
    a true production environment.
    """
    
#: <data_profiler_init>
    def __init__(self, na_threshold: float = 0.40, outlier_iqr_multiplier: float = 1.5):
        """
        Initializes the DataProfiler with strict filtering thresholds.

        Args:
            na_threshold (float): Maximum tolerable ratio of missing values. Features 
                                  exceeding this are deemed unrecoverable and purged.
            outlier_iqr_multiplier (float): Multiplier for the IQR anomaly bounds. 
                                            1.5 is standard for capturing severe outliers 
                                            without dampening natural variance.
        """
        self.na_threshold = na_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.metadata: Dict[str, Any] = {}
#: </data_profiler_init>

#: <profile_and_clean>
    def profile_and_clean(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        date_col: Optional[str] = None, 
        group_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Profiles the training dataset, learns group-aware bounds, and enforces continuity.
        
        This method must ONLY be called on the training fold. It populates the internal 
        `metadata` dictionary, which acts as the frozen rulebook for the `transform` method.
        
        Args:
            df (pd.DataFrame): The raw training data.
            config (Dict[str, Any]): Parsed formula configuration (targets, features).
            date_col (str, optional): The column representing time.
            group_col (str, optional): The column representing distinct panel entities.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The cleaned dataset and the learned metadata state.
        """
        if df is None or df.empty:
            raise ValueError("DataProfiler received an empty or None DataFrame.")

        data = df.copy()
        targets = config.get("targets", [])
        
        self.metadata["is_time_series"] = date_col is not None and date_col in data.columns
        self.metadata["date_col"] = date_col
        self.metadata["group_col"] = group_col

        if self.metadata["is_time_series"]:
            data[date_col] = pd.to_datetime(data[date_col])
            data = self._enforce_time_continuity(data, date_col, group_col, is_fit=True)

        drop_cols = []
        for col in data.columns:
            if col not in targets and col not in [date_col, group_col]:
                if data[col].isna().mean() > self.na_threshold:
                    drop_cols.append(col)
                    
        if drop_cols:
            data.drop(columns=drop_cols, inplace=True)
            self.metadata["dropped_na_cols"] = drop_cols

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in targets and c not in [date_col, group_col]]
        
        self.metadata["bounds"] = self._learn_group_bounds(data, feature_cols, group_col)
        self.metadata["clipped_outlier_cols"] = feature_cols
        
        data = self._apply_clipping(data, feature_cols, group_col)

        return data, self.metadata
#: </profile_and_clean>

#: <transform>
    def transform(self, df: pd.DataFrame, config: Dict[str, Any], date_col: Optional[str] = None) -> pd.DataFrame:
        """
        Applies exact cleaning and clipping rules strictly to unseen validation/test data.
        
        Guarantees zero target leakage by refusing to recalculate IQR bounds or drop metrics 
        based on the new data. Safely handles "Cold-Start" groups by falling back to the 
        global statistical bounds learned during `profile_and_clean`.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        data = df.copy()
        group_col = self.metadata.get("group_col")
        
        if self.metadata.get("is_time_series") and date_col and date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data = self._enforce_time_continuity(data, date_col, group_col, is_fit=False)

        if "bounds" in self.metadata:
            feature_cols = self.metadata.get("clipped_outlier_cols", [])
            valid_feats = [c for c in feature_cols if c in data.columns]
            data = self._apply_clipping(data, valid_feats, group_col)
                    
        return data
#: </transform>

#: <data_profiler_helpers>
    def _enforce_time_continuity(
        self, 
        data: pd.DataFrame, 
        date_col: str, 
        group_col: Optional[str], 
        is_fit: bool
    ) -> pd.DataFrame:
        """
        Transforms raw, row-based datasets into mathematically continuous physical time series.
        
        Args:
            data (pd.DataFrame): The input dataset containing a datetime column.
            date_col (str): The name of the datetime column.
            group_col (Optional[str]): The column name representing distinct entities.
            is_fit (bool): Flag indicating if the pipeline is in the training phase.

        Returns:
            pd.DataFrame: A chronologically continuous dataframe.
        """
        data = data.sort_values(by=date_col)
        
        if is_fit:
            if group_col and group_col in data.columns:
                sample_group = data[group_col].iloc[0]
                sample_dates = data[data[group_col] == sample_group][date_col].dropna().unique()
            else:
                sample_dates = data[date_col].dropna().unique()
                
            dates_series = pd.Series(sample_dates).sort_values().reset_index(drop=True)
            
            if len(dates_series) < 3:
                self.metadata["delta_t"] = "1D" 
            else:
                inferred_freq = pd.infer_freq(dates_series.head(10))
                
                if inferred_freq is not None:
                    self.metadata["delta_t"] = inferred_freq
                else:
                    diffs = dates_series.diff().dropna()
                    median_delta = diffs.median()
                    days = median_delta.days
                    
                    if 360 <= days <= 370:
                        self.metadata["delta_t"] = "YS"
                    elif 88 <= days <= 93:
                        self.metadata["delta_t"] = "QS"
                    elif 27 <= days <= 31:
                        self.metadata["delta_t"] = "MS"
                    elif days == 7:
                        self.metadata["delta_t"] = "W" 
                    elif days >= 1:
                        self.metadata["delta_t"] = f"{days}D"
                    else:
                        hours = int(median_delta.total_seconds() // 3600)
                        if hours >= 1:
                            self.metadata["delta_t"] = f"{hours}H"
                        else:
                            minutes = max(1, int(median_delta.total_seconds() // 60))
                            self.metadata["delta_t"] = f"{minutes}min"

        freq = self.metadata.get("delta_t", "1D")
        
        data = data.set_index(date_col)
        
        if group_col and group_col in data.columns:
            resampled = data.groupby(group_col).resample(freq).asfreq()
            
            if group_col in resampled.index.names:
                resampled = resampled.drop(columns=[group_col], errors='ignore')
            data = resampled.reset_index()
            
            data = data.groupby(group_col).ffill(limit=3).bfill()
        else:
            data = data.resample(freq).asfreq().reset_index()
            data = data.ffill(limit=3).bfill()
            
        return data

    def _learn_group_bounds(self, data: pd.DataFrame, feature_cols: List[str], group_col: Optional[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Learns the IQR boundaries globally and per-group.
        """
        bounds = {"global": {}}
        
        for col in feature_cols:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            bounds["global"][col] = (q1 - self.outlier_iqr_multiplier * iqr, q3 + self.outlier_iqr_multiplier * iqr)
            
        if group_col and group_col in data.columns:
            for name, group in data.groupby(group_col):
                bounds[name] = {}
                for col in feature_cols:
                    q1, q3 = group[col].quantile(0.25), group[col].quantile(0.75)
                    iqr = q3 - q1
                    bounds[name][col] = (q1 - self.outlier_iqr_multiplier * iqr, q3 + self.outlier_iqr_multiplier * iqr)
                    
        return bounds

    def _apply_clipping(self, data: pd.DataFrame, feature_cols: List[str], group_col: Optional[str]) -> pd.DataFrame:
        """
        Applies the bounds using vectorized pandas operations for maximum performance.
        Safely falls back to global boundaries if a requested group was not present during training.
        """
        bounds = self.metadata.get("bounds", {"global": {}})
        
        if group_col and group_col in data.columns:
            for name, group_idx in data.groupby(group_col).groups.items():
                bnd = bounds.get(name, bounds["global"])
                for col in feature_cols:
                    if col in bnd:
                        data.loc[group_idx, col] = data.loc[group_idx, col].clip(lower=bnd[col][0], upper=bnd[col][1])
        else:
            bnd = bounds.get("global", {})
            for col in feature_cols:
                if col in bnd:
                    data[col] = data[col].clip(lower=bnd[col][0], upper=bnd[col][1])
                    
        return data
#: </data_profiler_helpers>
#: </data_profiler_class>