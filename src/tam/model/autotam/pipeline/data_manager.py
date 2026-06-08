#: <data_manager_module_doc>
"""
Data Manager for Automated TAM (AutoTAM).

Handles data topology, semantic parsing, chronological splitting, 
and feature engineering for the AutoTAM pipeline.
"""
#: </data_manager_module_doc>

#: <data_manager_imports>
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .context import PipelineContext

from tam.model.autotam.parser import FormulaParser
from tam.model.autotam.data_profiler import DataProfiler
from tam.model.autotam.feature_engineer import FeatureEngineer
from tam.model.autotam.effect_selector import EffectSelector
#: </data_manager_imports>

#: <data_manager_class>
class DataManager:
    """
    Central data orchestrator.
    Manages the lifecycle of datasets through parsing, cleaning, and augmentation.
    """
    
#: <data_manager_init>
    def __init__(
        self, 
        formula: str, 
        lags: Optional[List[int]] = None,
        train_fraction: float = 0.70,
        dev_fraction: float = 0.15
    ):
        self.formula = formula
        self.user_lags = lags or []
        self.train_fraction = train_fraction
        self.dev_fraction = dev_fraction
        
        self.parser = FormulaParser()
        self.profiler = DataProfiler()
        self.engineer = FeatureEngineer(collinearity_threshold=0.99)
        self.selector = EffectSelector()
#: </data_manager_init>

#: <data_manager_prepare>
    def prepare(
        self, 
        df_train: Optional[pd.DataFrame] = None, 
        df_fit: Optional[pd.DataFrame] = None, 
        df_dev: Optional[pd.DataFrame] = None, 
        df_val: Optional[pd.DataFrame] = None, 
        date_col: Optional[str] = None, 
        group_col: Optional[str] = None,
        validation_strategy: str = 'auto'
    ) -> PipelineContext:
        """
        Executes the data preparation phase and populates the pipeline context.
        """
        ctx = PipelineContext(date_col=date_col, group_col=group_col)
        
        ctx.formula_config = self.parser.parse(self.formula)
        ctx.target = ctx.formula_config["targets"][0]
        
        parsed_lags = list(ctx.formula_config.get("lags", {}).values())
        ctx.lags = list(set(self.user_lags + parsed_lags))

        if df_train is not None:
            df_proc = df_train.sort_values(date_col) if date_col else df_train
            n = len(df_proc)
            fit_end = int(n * self.train_fraction)
            dev_end = int(n * (self.train_fraction + self.dev_fraction))
            
            raw_fit = df_proc.iloc[:fit_end].copy()
            raw_dev = df_proc.iloc[fit_end:dev_end].copy()
            raw_val = df_proc.iloc[dev_end:].copy()
        else:
            if df_fit is None or df_dev is None or df_val is None:
                raise ValueError("Provide df_train OR explicit df_fit, df_dev, df_val folds.")
            raw_fit = df_fit.copy()
            raw_dev = df_dev.copy()
            raw_val = df_val.copy()

        _, ctx.metadata = self.profiler.profile_and_clean(raw_fit, ctx.formula_config, date_col, group_col)
        
        df_all_raw = pd.concat([raw_fit, raw_dev, raw_val])
        if date_col: 
            df_all_raw = df_all_raw.sort_values(date_col)

        if ctx.lags:
            for d in ctx.lags: 
                if group_col and group_col in df_all_raw.columns:
                    df_all_raw[f"{ctx.target}_lag_{d}"] = df_all_raw.groupby(group_col)[ctx.target].shift(d)
                else: 
                    df_all_raw[f"{ctx.target}_lag_{d}"] = df_all_raw[ctx.target].shift(d)

        df_all_clean = self.profiler.transform(df_all_raw, ctx.formula_config, date_col) if hasattr(self.profiler, 'transform') else df_all_raw
        df_all_aug = self.engineer.engineer_features(df_all_clean, ctx.formula_config, ctx.metadata, date_col)
        
        ctx.df_all_aug = df_all_aug

        if date_col and date_col in df_all_aug.columns and date_col in raw_fit.columns:
            fit_max_date = raw_fit[date_col].max()
            dev_max_date = raw_dev[date_col].max()
            
            ctx.df_fit = df_all_aug[df_all_aug[date_col] <= fit_max_date].copy()
            ctx.df_dev = df_all_aug[(df_all_aug[date_col] > fit_max_date) & (df_all_aug[date_col] <= dev_max_date)].copy()
            ctx.df_val = df_all_aug[df_all_aug[date_col] > dev_max_date].copy()
        else:
            len_fit = len(raw_fit)
            len_dev = len(raw_dev)
            
            ctx.df_fit = df_all_aug.iloc[:len_fit].copy()
            ctx.df_dev = df_all_aug.iloc[len_fit:len_fit + len_dev].copy()
            ctx.df_val = df_all_aug.iloc[len_fit + len_dev:].copy()

        ctx.validation_strategy = validation_strategy
        ctx.cv_folds = self.generate_cv_folds(ctx.df_fit, ctx.df_dev, date_col, validation_strategy)

        max_look = max(ctx.lags) if ctx.lags else 24
        if group_col and group_col in df_all_aug.columns:
            ctx.historical_tail = df_all_aug.groupby(group_col).tail(max_look + 1).copy() 
        else:
            ctx.historical_tail = df_all_aug.tail(max_look + 1).copy()

        ctx.search_space = self.selector.build_search_space(ctx.df_fit, ctx.formula_config, ctx.metadata)

        reference_df = df_train if df_train is not None else ctx.df_fit
        target_series = reference_df[ctx.target]

        target_var = float(target_series.var())
        target_mean = float(target_series.mean())

        inferred_observation_noise = max(1e-5, target_var * 0.05)
        inferred_P_init = max(1e-2, target_var)
        inferred_offset_boost = abs(target_mean)

        ctx.kalman_priors = {
            "observation_noise_var": inferred_observation_noise,
            "P_init_diag": inferred_P_init,
            "offset_boost": inferred_offset_boost
        }

        return ctx
#: </data_manager_prepare>

#: <data_manager_cv>
    def generate_cv_folds(
        self, 
        df_fit: pd.DataFrame, 
        df_dev: pd.DataFrame, 
        date_col: Optional[str], 
        validation_strategy: str
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates cross-validation folds based on the specified strategy and data topology.
        """
        if validation_strategy == 'holdout':
            return [(df_fit.copy(), df_dev.copy())]

        df_pool = pd.concat([df_fit, df_dev])
        if date_col and date_col in df_pool.columns:
            df_pool = df_pool.sort_values(date_col)

        is_timeseries = False
        step_size = None

        if validation_strategy in ['auto', 'expanding_window'] and date_col and date_col in df_pool.columns:
            try:
                date_series = pd.to_datetime(df_pool[date_col])
                deltas = date_series.diff().dropna()
                if not deltas.empty:
                    median_delta = deltas.median()
                    if median_delta >= pd.Timedelta(days=28):
                        step_size = 12
                    elif median_delta >= pd.Timedelta(days=1):
                        step_size = 30
                    else:
                        step_size = max(1, len(df_pool) // 10)
                    is_timeseries = True
            except Exception:
                pass

        if validation_strategy == 'expanding_window' or is_timeseries:
            if step_size is None:
                step_size = max(1, len(df_pool) // 5)
            
            folds = []
            n_samples = len(df_pool)
            initial_train_size = max(step_size * 2, n_samples // 3)
            
            current_end = initial_train_size
            while current_end + step_size <= n_samples:
                train_fold = df_pool.iloc[:current_end].copy()
                val_fold = df_pool.iloc[current_end:current_end + step_size].copy()
                folds.append((train_fold, val_fold))
                current_end += step_size
            
            if current_end < n_samples and len(folds) > 0:
                train_fold = df_pool.iloc[:current_end].copy()
                val_fold = df_pool.iloc[current_end:].copy()
                folds.append((train_fold, val_fold))
                
            if not folds:
                folds = [(df_fit.copy(), df_dev.copy())]
            return folds
            
        folds = []
        indices = np.random.RandomState(seed=42).permutation(df_pool.index)
        fold_size = max(1, len(indices) // 5)
        
        for i in range(5):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < 4 else len(indices)
            val_idx = indices[start_idx:end_idx]
            train_idx = np.setdiff1d(indices, val_idx)
            
            train_fold = df_pool.loc[train_idx].copy()
            val_fold = df_pool.loc[val_idx].copy()
            folds.append((train_fold, val_fold))
            
        return folds
#: </data_manager_cv>

#: <data_manager_transform>
    def transform_test_data(self, df_test: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """
        Transforms unseen test data during the predict phase, utilizing the historical tail.
        """
        df_combined = df_test.copy()
        
        if ctx.historical_tail is not None and ctx.date_col and ctx.date_col in df_test.columns:
            if df_test[ctx.date_col].min() > ctx.historical_tail[ctx.date_col].max():
                df_combined = pd.concat([ctx.historical_tail, df_test], ignore_index=True)

        if ctx.lags:
            for d in ctx.lags: 
                if ctx.group_col and ctx.group_col in df_combined.columns:
                    df_combined[f"{ctx.target}_lag_{d}"] = df_combined.groupby(ctx.group_col)[ctx.target].shift(d)
                else: 
                    df_combined[f"{ctx.target}_lag_{d}"] = df_combined[ctx.target].shift(d)

        df_clean = self.profiler.transform(df_combined, ctx.formula_config, ctx.date_col) if hasattr(self.profiler, 'transform') else df_combined
        df_aug = self.engineer.engineer_features(df_clean, ctx.formula_config, ctx.metadata, ctx.date_col)

        return df_aug.iloc[-len(df_test):].copy(), df_aug
#: </data_manager_transform>
#: </data_manager_class>