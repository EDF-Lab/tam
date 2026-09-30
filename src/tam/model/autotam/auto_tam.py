# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
AutoTAM Orchestrator (TAM AutoML Layer).

This module manages the end-to-end automated machine learning flow for time-series forecasting.
It acts as the central execution engine, coordinating data topology mapping, evolutionary 
search, state-space expansions (Kalman/Adaptive), and Minimax aggregation (OPERA).
"""

#: <auto_tam_imports>
import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict

from .pipeline.data_manager import DataManager
from .pipeline.base_discoverer import BaseDiscoverer
from .pipeline.expert_expander import ExpertExpander
from .pipeline.ensemble_selector import EnsembleSelector
from .evaluation.autotam_report_generator import generate_autotam_report, print_model_recipe
from .evolution_reporter import EvolutionReporter
from tam.model.kalman import KalmanTAM
from tam.model.adaptative import AdaptiveTAM
from tam.model.opera import OperaTAM
#: </auto_tam_imports>

#: <auto_tam_class>
class AutoTAM:
    """The Clean AutoTAM Director."""
    
#: <auto_tam_init>
    def __init__(
        self, 
        formula: str, 
        lags: Optional[List[int]] = None, 
        mandatory_variables: Optional[List[str]] = None,
        n_experts: int = 15, 
        pop_size: int = 64, 
        use_opera: bool = True, 
        eta: float = 0.1, 
        complexity_penalty: float = 1.0,
        export_dir: str = "AutoTAM_exports", 
        **kwargs
    ):
        """
        Initializes the AutoTAM director.

        Args:
            formula (str): High-level AutoTAM formula with embedded mandatory terms and a
                pipeline macro (e.g., 'load ~ s(temp, k=10) + AutoPipe(temp, humidity)').
            lags (Optional[List[int]]): Explicit lag orders to evaluate during feature engineering.
            mandatory_variables (Optional[List[str]]): Variables required in all models.
            n_experts (int): Number of top expert architectures to retain in the ensemble.
            pop_size (int): Population size for island-based evolutionary exploration.
            use_opera (bool): Whether to use OPERA minimax aggregation for ensembling.
            eta (float): Learning rate for the estimation of distribution algorithm.
            complexity_penalty (float): Multiplier penalizing model structural complexity.
            export_dir (str): Directory where intermediate logs and exports are saved.
            **kwargs: Additional runtime options forwarded to sub-components.

        Raises:
            TypeError: If deprecated 'mandatory_terms' keyword argument is passed.
        """
        if "mandatory_terms" in kwargs:
            raise TypeError("AutoTAM.__init__() got an unexpected keyword argument 'mandatory_terms'")
        self.formula = formula
        self.n_experts = n_experts
        self.complexity_penalty = complexity_penalty
        if isinstance(mandatory_variables, str):
            mandatory_variables = [mandatory_variables]
        self.mandatory_variables = mandatory_variables or []
        
        self.data_manager = DataManager(
            formula=formula,
            lags=lags,
            mandatory_variables=self.mandatory_variables,
        )
        self.mandatory_terms = self.data_manager.mandatory_terms
        self.discoverer = BaseDiscoverer(
            pop_size=pop_size, 
            eta=eta, 
            mandatory_terms=self.mandatory_terms,
            mandatory_variables=self.mandatory_variables
        )
        self.expander = ExpertExpander()
        self.selector = EnsembleSelector(use_opera=use_opera)
        self.reporter = EvolutionReporter(export_dir=export_dir)
        
        self.export_directory = export_dir
        self.ctx = None
        self.trained_experts = [] 
        self.league_weights = {}
        self.weights_top10 = {}
        self.island_aliases = {}
        self.oof_predictions_ = {}
        self.chronological_test_log = []
        self.online_weights_ = {}   # populated by predict_online(): MLpol weight trajectories
        self.apex_members_ = []     # the whole Apex pool, aggregated by predict_online()
#: </auto_tam_init>

#: <auto_tam_fit>
    def fit(
        self, 
        df_train: Optional[pd.DataFrame] = None, 
        df_fit: Optional[pd.DataFrame] = None, 
        df_dev: Optional[pd.DataFrame] = None, 
        df_val: Optional[pd.DataFrame] = None, 
        date_col: Optional[str] = None, 
        group_col: Optional[str] = None, 
        expansions: Optional[Dict[str, bool]] = None, 
        validation_strategy: str = 'auto',
        optimization_metric: str = 'rmse',
        refit_on_full_train: bool = False
    ):
        print("AutoTAM: Starting Pipeline...")
        self.expander.expansions = expansions or {"prior": True, "autofit": True, "kalman": True, "adaptive": True, "grid": False}
        
        self.ctx = self.data_manager.prepare(
            df_train=df_train, 
            df_fit=df_fit, 
            df_dev=df_dev, 
            df_val=df_val, 
            date_col=date_col, 
            group_col=group_col,
            validation_strategy=validation_strategy
        )
        self.ctx.optimization_metric = optimization_metric
        self.ctx.complexity_penalty = self.complexity_penalty
        
        island_champions, draga_engine = self.discoverer.search(self.ctx)
        self.reporter.export_evolutionary_diagnostics(draga_engine)
        
        candidate_pool = self.expander.generate_experts(island_champions, self.ctx, self.chronological_test_log, self.reporter)
        
        if not candidate_pool:
            print("Warning: No candidates generated. The model will remain unfitted.")
            return self

        self.trained_experts, self.league_weights, self.weights_top10, self.island_aliases, self.oof_predictions_ = self.selector.evaluate_and_refit(
            candidate_pool, self.ctx, self.chronological_test_log, self.reporter, self.expander, refit_on_full_train=refit_on_full_train
        )
        self.apex_members_ = list(getattr(self.selector, "apex_members_", []))
        
        if not self.trained_experts:
            print("Warning: All experts failed evaluation. The model is effectively empty.")

        self.reporter.export_final_architectures(self.trained_experts)
        self.reporter.export_chronological_tests(self.chronological_test_log)
        self.reporter.export_collinearity_purge_log(self.data_manager.engineer.purge_log)
        return self
#: </auto_tam_fit>

#: <auto_tam_predict>
    def _expert_predictions(self, df_test: pd.DataFrame, date_col: Optional[str] = None):
        """Run every trained expert on df_test -> (expert prediction frame, df_test_clean).

        Shared by predict() and predict_online() so the dynamic-expert row alignment
        (_align_prediction_to_test) lives in exactly one place.
        """
        if not self.trained_experts:
            raise ValueError("Model not fitted. No valid experts survived the fit process.")

        if hasattr(self.ctx, "external_mandatory_features") and self.ctx.external_mandatory_features:
            missing_ext = set(self.ctx.external_mandatory_features) - set(df_test.columns)
            if missing_ext:
                raise ValueError(
                    f"Test data is missing required external mandatory features: {sorted(missing_ext)}"
                )

        df_test_clean, df_aug = self.data_manager.transform_test_data(df_test, self.ctx)
        predictions = {}
        
        for exp in self.trained_experts:
            m, name, m_type = exp["model"], exp["name"], exp["type"]
            try:
                base_m = m if m_type == "static" else getattr(m, '_saved_base_model', getattr(m, 'base_model_', None))
                f_ref = getattr(base_m, 'formula_', '')
                req_cols = list(set([c for c in re.findall(r'[a-zA-Z0-9_]+', f_ref) if c in df_aug.columns]))
                df_aug_clean = df_aug.dropna(subset=req_cols).copy()
                
                if m_type == "static": 
                    predictions[name] = base_m.predict(df_test_clean)[f"Estimated{self.ctx.target}"].values
                
                elif m_type == "kalman":
                    k_priors = getattr(self.ctx, 'kalman_priors', {})
                    obs_noise = k_priors.get("observation_noise_var", 0.1)
                    p_init = k_priors.get("P_init_diag", 1.0)
                    boost = k_priors.get("offset_boost", 100.0)

                    h_steps = getattr(self.ctx, 'inferred_horizon', len(df_test))
                    
                    fresh_kalman = KalmanTAM(
                        base_model=base_m, 
                        kalman_formula=exp["dynamic_formula"], 
                        date_col=self.ctx.date_col,
                        horizon_steps=h_steps, 
                        offset_boost=boost, 
                        observation_noise_var=obs_noise, 
                        P_init_diag=p_init,
                        process_noise_var=exp["params"]["process_noise_var"] 
                    )
                    p_all = fresh_kalman.predict_online(df_aug_clean)
                    raw_preds = pd.Series(p_all[f"KalmanAdapted_{self.ctx.target}"].values, index=df_aug_clean.index)
                    predictions[name] = self._align_prediction_to_test(raw_preds, df_aug_clean, df_test_clean)
                
                elif m_type == "adaptive":
                    h_steps = getattr(self.ctx, 'inferred_horizon', len(df_test))
                    s_per_period = getattr(self.ctx, 'steps_per_period', 1)
                    
                    fresh_adapt = AdaptiveTAM(
                        base_model=base_m, adaptive_formula=exp["dynamic_formula"], update_interval_periods=1,
                        training_window_periods=exp["params"]["training_window_periods"], 
                        steps_per_period=s_per_period, horizon_steps=h_steps
                    )
                    d_adapt = self.expander._prepare_meta_learning_data(df_aug_clean, base_m, self.ctx)
                    lag_cols = [c for c in d_adapt.columns if f"Residual{self.ctx.target}_lag_" in c]
                    d_adapt = d_adapt.dropna(subset=lag_cols).copy()
                    
                    if hasattr(fresh_adapt, 'predict_online'): 
                        fresh_adapt.predict_online(data=d_adapt)
                    else:
                        fresh_adapt.prepare_simulation(d_adapt)
                        fresh_adapt.simulation()
                        
                    raw_preds = fresh_adapt.predictions_[f"AdaptedEstimated{self.ctx.target}"]
                    predictions[name] = self._align_prediction_to_test(raw_preds, d_adapt, df_test_clean)
                    
            except Exception as e: 
                print(f"Warning: Prediction failed for {name}: {e}")
                
        preds_df = pd.DataFrame(predictions, index=df_test_clean.index)

        # Physical guardrail: Clip predictions to observed bounds (with margin) to prevent extreme OOD extrapolations from breaking static frozen-weight ensembles.
        bounds = getattr(self.ctx, "target_clip", None)
        if bounds is None and getattr(self.ctx, "df_fit", None) is not None:
            try:
                low = float(self.ctx.df_fit[self.ctx.target].min())
                high = float(self.ctx.df_fit[self.ctx.target].max())
                margin = 0.15 * (high - low)
                bounds = (low - margin, high + margin)
            except Exception:
                bounds = None
        if bounds is not None and not preds_df.empty:
            preds_df = preds_df.clip(lower=bounds[0], upper=bounds[1])

        for alias, real_name in self.island_aliases.items():
            if real_name in preds_df.columns:
                preds_df[alias] = preds_df[real_name]

        return preds_df, df_test_clean

    def predict(self, df_test: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """Frozen-weight inference (production path).

        League/apex weights learned on D_val during fit are applied unchanged across the whole
        horizon. Deterministic and target-free. For a chronological backtest where the realized
        target IS available, prefer predict_online(): MLpol then re-weights the experts
        sequentially, which matters a great deal across regime shifts.
        """
        preds_df, df_test_clean = self._expert_predictions(df_test, date_col)

        for league_name, weights in self.league_weights.items():
            if weights:
                valid_weights = {m: w for m, w in weights.items() if m in preds_df}
                w_sum = sum(valid_weights.values())
                if w_sum > 0:
                    final_name = league_name if league_name.startswith("Ensemble_") else f"Ensemble_{league_name}"
                    preds_df[final_name] = sum(preds_df[m] * (w / w_sum) for m, w in valid_weights.items())

        if self.weights_top10:
            valid_apex = {e: w for e, w in self.weights_top10.items() if e in preds_df}
            apex_sum = sum(valid_apex.values())
            if apex_sum > 0:
                preds_df["AutoTAM_Apex_Ensemble"] = sum(preds_df[e] * (w / apex_sum) for e, w in valid_apex.items())

        target = self.ctx.target
        if target in df_test.columns and df_test[target].notna().all():
            print("Hint: df_test carries a realized target, so this is a backtest. predict() applies "
                  "FROZEN league/apex weights learned on D_val; call predict_online() to let MLpol "
                  "re-weight the experts chronologically across regime shifts.")

        return preds_df

    def predict_online(self, df_test: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """Sequential inference: MLpol re-weights the experts chronologically.

        Mirrors OperaTAM.predict_online at the orchestrator level. Requires the realized target in
        df_test, because online aggregation scores every expert's loss at each step to update its
        weight; it is therefore a backtest/replay path, not a target-free forward forecast.

        Adds 'AutoTAM_Apex_Online' and 'Ensemble_<league>_Online' next to the individual experts,
        and stores the weight trajectories on self.online_weights_ for diagnostics/reporting.
        The online Apex aggregates the whole Apex pool (apex_members_); predict() averages only the
        members still weighted at the end of validation (weights_top10).
        """
        preds_df, df_test_clean = self._expert_predictions(df_test, date_col)
        target = self.ctx.target

        y_series = pd.to_numeric(df_test_clean[target], errors="coerce") if target in df_test_clean.columns else None
        if y_series is None or y_series.isna().any():
            raise ValueError(
                f"predict_online() requires the realized target column '{target}' (fully populated) "
                "in df_test: MLpol updates each expert's weight from its observed loss at every "
                "step. Use predict() for target-free forward inference."
            )

        d_col, g_col = self.ctx.date_col, self.ctx.group_col
        base = pd.DataFrame(index=df_test_clean.index)
        base[target] = y_series.to_numpy()
        if d_col and d_col in df_test_clean.columns:
            base[d_col] = pd.to_datetime(df_test_clean[d_col]).to_numpy()
        if g_col and g_col in df_test_clean.columns:
            base[g_col] = df_test_clean[g_col].to_numpy()

        self.online_weights_ = {}

        def _aggregate_online(members, out_name):
            usable = [m for m in members if m in preds_df.columns
                      and np.isfinite(pd.to_numeric(preds_df[m], errors="coerce").to_numpy()).all()]
            if len(usable) < 2:
                return
            frame = base.copy()
            for m in usable:
                frame[m] = pd.to_numeric(preds_df[m], errors="coerce").to_numpy()
            formula = f"{target} ~ " + " + ".join(f"l({m})" for m in usable)
            try:
                opera = OperaTAM(formula=formula, algorithm="MLPOL", date_col=d_col, group_col=g_col)
                result = opera.predict_online(frame)
                aggregated = pd.to_numeric(result["prediction_opera"], errors="coerce").to_numpy()
                if len(aggregated) != len(preds_df):
                    print(f"Warning: online aggregation for {out_name} returned {len(aggregated)} rows "
                          f"for {len(preds_df)} test rows; skipped.")
                    return
                preds_df[out_name] = aggregated
                weight_cols = [c for c in result.columns if c.startswith("weight_")]
                if weight_cols:
                    self.online_weights_[out_name] = result[weight_cols].reset_index(drop=True)
            except Exception as exc:
                print(f"Warning: online aggregation failed for {out_name}: {exc}")

        for league_name, weights in self.league_weights.items():
            if weights:
                stem = league_name if league_name.startswith("Ensemble_") else f"Ensemble_{league_name}"
                _aggregate_online(list(weights.keys()), f"{stem}_Online")

        # The online Apex dynamically aggregates the full pool; static predict() averages only the snapshot weights.
        apex_members = list(getattr(self, "apex_members_", None) or self.weights_top10.keys())
        if apex_members:
            _aggregate_online(apex_members, "AutoTAM_Apex_Online")

        return preds_df

    def _align_prediction_to_test(self, raw_preds, source_df, df_test_clean):
        """Realign dynamic expert predictions (augmented history + test) back
        to the original test rows using date/group keys.

        """
        n = len(df_test_clean)
        date_col, group_col = self.ctx.date_col, self.ctx.group_col
        try:
            if hasattr(raw_preds, "index") and raw_preds.index.isin(source_df.index).all():
                src_keys = source_df.loc[raw_preds.index]
                vals = np.asarray(raw_preds.values, dtype=float)
            elif len(raw_preds) == len(source_df):
                src_keys = source_df
                vals = np.asarray(getattr(raw_preds, "values", raw_preds), dtype=float)
            else:
                raise ValueError("raw_preds not alignable to source_df")

            if date_col and date_col in source_df.columns and date_col in df_test_clean.columns:
                gcol = group_col if (group_col and group_col in source_df.columns
                                     and group_col in df_test_clean.columns) else None
                keys = [date_col] + ([gcol] if gcol else [])
                freq = getattr(self.ctx, "metadata", {}).get("delta_t", "1D")
                src = src_keys[keys].copy()
                src["__v__"] = vals
                src[date_col] = pd.to_datetime(src[date_col])
                left = df_test_clean[keys].copy()
                left[date_col] = pd.to_datetime(left[date_col])
                try:
                    src[date_col] = src[date_col].dt.floor(freq)
                    left[date_col] = left[date_col].dt.floor(freq)
                except (ValueError, TypeError):
                    pass
                src = src.drop_duplicates(subset=keys, keep="last")
                left["__ord__"] = range(n)
                merged = left.merge(src, on=keys, how="left").sort_values("__ord__")
                out = merged["__v__"].to_numpy()
                if len(out) == n and np.isfinite(out).any():
                    return out
        except Exception:
            pass
        vals = np.asarray(getattr(raw_preds, "values", raw_preds), dtype=float)
        return vals[-n:] if len(vals) >= n else np.full(n, np.nan)
#: </auto_tam_predict>

#: <auto_tam_utils>
    def summary(self) -> None:
        self.reporter.print_summary(self.trained_experts, {}, {})

    def print_performance_board(self) -> None:
        print("\n" + "="*90 + "\nTAM: Global Model Performance Leaderboard\n" + "="*90)
        valid_logs = [log for log in self.chronological_test_log if pd.notna(log.get("Validation_RMSE"))]
        if not valid_logs: return
        
        valid_logs.sort(key=lambda x: float(x.get("Penalized_Score", x.get("Validation_RMSE"))))
        
        metric_display = getattr(self.ctx, 'optimization_metric', 'rmse').upper() if self.ctx else 'SCORE'
        
        for rank, log in enumerate(valid_logs, 1):
            raw_score = float(log.get('Validation_RMSE'))
            pen_score = float(log.get('Penalized_Score', raw_score))
            comp = log.get('Complexity', 'N/A')
            
            print(f"\n[{rank}] {log.get('Model_Name')} ({log.get('Model_Type')}) | Params: {comp} | Penalized {metric_display}: {pen_score:.5f} (Raw: {raw_score:.5f})\n    -> Formula: {log.get('Formula')}")
            params = log.get("Hyperparameters", "")
            if params and params not in ["{}", "Unpenalized_Ridge", "GCV_Auto_Penalized", "GCV_Auto"]: 
                print(f"    -> Params:  {params}")
        print("\n" + "="*90)

    def summary_report(self):
        """
        Convenience method to generate the report directly from the model object.
        """
        opt_metric = getattr(self.ctx, 'optimization_metric', 'RMSE') if self.ctx else 'RMSE'
        generate_autotam_report(export_path=self.export_directory, metric=opt_metric)

    def print_model_recipe(self, internal_model_name: str) -> None:
        """
        Prints the complete mathematical recipe for a given model or ensemble.
        Includes aggregation weights for OPERA models and base formulas for dynamic models.
        """
        print_model_recipe(self, internal_model_name)
#: </auto_tam_utils>
#: </auto_tam_class>