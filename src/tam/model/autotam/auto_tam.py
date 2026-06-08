#: <auto_tam_module_doc>
"""
AutoTAM Orchestrator (TAM AutoML Layer).

This module manages the end-to-end automated machine learning flow for time-series forecasting.
It acts as the central execution engine, coordinating data topology mapping, evolutionary 
search, state-space expansions (Kalman/Adaptive), and Minimax aggregation (OPERA).
"""
#: </auto_tam_module_doc>

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
#: </auto_tam_imports>

#: <auto_tam_class>
class AutoTAM:
    """The Clean AutoTAM Director."""
    
#: <auto_tam_init>
    def __init__(
        self, 
        formula: str, 
        lags: Optional[List[int]] = None, 
        n_experts: int = 15, 
        pop_size: int = 64, 
        use_opera: bool = True, 
        eta: float = 0.1, 
        complexity_penalty: float = 1.0,
        export_dir: str = "AutoTAM_exports", 
        **kwargs
    ):
        self.formula = formula
        self.n_experts = n_experts
        self.complexity_penalty = complexity_penalty
        
        self.data_manager = DataManager(formula=formula, lags=lags)
        self.discoverer = BaseDiscoverer(pop_size=pop_size, eta=eta)
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
        
        if not self.trained_experts:
            print("Warning: All experts failed evaluation. The model is effectively empty.")

        self.reporter.export_final_architectures(self.trained_experts)
        self.reporter.export_chronological_tests(self.chronological_test_log)
        self.reporter.export_collinearity_purge_log(self.data_manager.engineer.purge_log)
        return self
#: </auto_tam_fit>

#: <auto_tam_predict>
    def predict(self, df_test: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        if not self.trained_experts: 
            raise ValueError("Model not fitted. No valid experts survived the fit process.")
        
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
                    predictions[name] = raw_preds.reindex(df_test_clean.index).values
                
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
                    predictions[name] = raw_preds.reindex(df_test_clean.index).values
                    
            except Exception as e: 
                print(f"Warning: Prediction failed for {name}: {e}")
                
        preds_df = pd.DataFrame(predictions, index=df_test_clean.index)
        
        for alias, real_name in self.island_aliases.items():
            if real_name in preds_df.columns:
                preds_df[alias] = preds_df[real_name]
        
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
            
        return preds_df
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