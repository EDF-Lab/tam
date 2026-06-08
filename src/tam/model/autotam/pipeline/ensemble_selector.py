#: <ensemble_selector_module_doc>
"""
Ensemble Selector for Automated TAM (AutoTAM).

This module manages the evaluation and aggregation of the evolutionary candidates.
It leverages the OPERA (Online Prediction by Expert Aggregation) algorithm to 
dynamically assign weights to the best models, creating robust federations and 
apex ensembles resilient to structural breaks in time-series data.
"""
#: </ensemble_selector_module_doc>

#: <ensemble_selector_imports>
import pandas as pd
import numpy as np
import re
import datetime
from typing import Dict, Any, Tuple
from .context import PipelineContext
from tam.model.opera import OperaTAM
#: </ensemble_selector_imports>

#: <ensemble_selector_class>
class EnsembleSelector:
    """Selects best Island representations and aggregates them via OPERA Minimax."""
    
#: <ensemble_selector_init>
    def __init__(self, use_opera: bool = True, n_apex_experts: int = 5, ensemble_sparsity_threshold: float = 0.01):
        """
        Initializes the ensemble aggregator.

        Args:
            use_opera (bool): Whether to use the online learning Minimax aggregation.
            n_apex_experts (int): The maximum number of top models to include in the Apex ensemble.
            ensemble_sparsity_threshold (float): Minimum weight required to keep a model active in the ensemble.
        """
        self.use_opera = use_opera
        self.n_apex_experts = n_apex_experts
        self.ensemble_sparsity_threshold = ensemble_sparsity_threshold
#: </ensemble_selector_init>

#: <ensemble_selector_metric_calculator>
    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """
        Calculates the error between true values and predictions based on the chosen metric.
        """
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        if mask.sum() == 0:
            return float('inf')
        
        y_t, y_p = y_true[mask], y_pred[mask]
        
        if metric.lower() == 'rmse':
            return np.sqrt(np.mean((y_t - y_p)**2))
        elif metric.lower() == 'mae':
            return np.mean(np.abs(y_t - y_p))
        elif metric.lower() == 'mape':
            denom = np.abs(y_t)
            denom[denom == 0] = 1e-9
            return np.mean(np.abs((y_t - y_p) / denom)) * 100.0
        else:
            return np.sqrt(np.mean((y_t - y_p)**2))
#: </ensemble_selector_metric_calculator>

#: <evaluate_and_refit>
    def evaluate_and_refit(
        self, 
        candidate_pool: Dict[str, Any], 
        ctx: PipelineContext, 
        test_log: list, 
        reporter, 
        expander, 
        refit_on_full_train: bool = False
    ) -> Tuple[list, dict, dict, dict, dict]:
        """
        Evaluates the generated experts, constructs logical sub-ensembles based on 
        Cross-Validation resilience and complexity penalties, and refits the final base models.
        """
        df_cont_val = pd.concat([ctx.df_fit, ctx.df_dev, ctx.df_val])
        trained_experts = []
        league_weights = {}
        weights_top10 = {}
        island_aliases = {}

        y_true = ctx.df_val[ctx.target].values
        preds_dict = {ctx.target: y_true}
        if ctx.date_col and ctx.date_col in ctx.df_val.columns:
            preds_dict[ctx.date_col] = ctx.df_val[ctx.date_col].values

        categories = {"kalman": [], "adaptive": [], "static": []}
        best_rmse_global, best_expert_global = float('inf'), None
        
        opt_metric = getattr(ctx, 'optimization_metric', 'rmse')
        
        n_fit_samples = len(ctx.df_fit) if ctx.df_fit is not None else 100
        n_val_samples = len(ctx.df_val) if ctx.df_val is not None else 24
        penalty_factor = getattr(ctx, 'complexity_penalty', 1.0)

        for name, exp in candidate_pool.items():
            try:
                p = self._get_expert_predictions(exp, exp["model"], df_cont_val, ctx, expander, return_full=False)
                preds_dict[name] = p
                categories[exp["type"]].append(name)
                
                val_score = self._calculate_error(y_true, p, opt_metric)
                
                if exp["type"] in ["adaptive", "kalman"]:
                    cv_score = val_score
                else:
                    cv_score = exp.get("cv_rmse", val_score)
                
                dyn_formula = exp.get("dynamic_formula", getattr(exp["model"], "formula_", ""))
                base_model = getattr(exp["model"], "_saved_base_model", getattr(exp["model"], "base_model_", None))
                
                if exp["type"] == "static":
                    base_formula = dyn_formula
                    comp_base = ctx.estimate_complexity(base_formula)
                    comp_dyn = 0
                else:
                    base_formula = getattr(base_model, "formula_", "") if base_model else ""
                    comp_base = ctx.estimate_complexity(base_formula) if base_formula else 0
                    comp_dyn = ctx.estimate_complexity(dyn_formula) if dyn_formula else 0
                    
                complexity = comp_base + comp_dyn
                
                base_ratio = comp_base / max(1, n_fit_samples)
                dyn_ratio = comp_dyn / max(1, n_val_samples)
                penalized_score = cv_score * (1.0 + penalty_factor * (base_ratio + dyn_ratio))
                
                exp_record = {
                    **exp, 
                    "name": name, 
                    "val_rmse": val_score, 
                    "cv_rmse": cv_score, 
                    "complexity": complexity, 
                    "penalized_score": penalized_score
                }
                trained_experts.append(exp_record)
                self._update_log_rmse(test_log, name, val_score, penalized_score, complexity)
                
                if cv_score < best_rmse_global:
                    best_rmse_global, best_expert_global = cv_score, exp_record
            except Exception as e:
                print(f"Warning: Evaluation failed for {name} | Error: {e}")

        df_p = pd.DataFrame(preds_dict).dropna()

        if self.use_opera and trained_experts:
            def run_single_opera(league_name, model_names):
                valid_models = [m for m in model_names if m in df_p.columns]
                if len(valid_models) <= 1: return {}
                
                form = f"{ctx.target} ~ " + " + ".join([f"l({e})" for e in valid_models])
                opera = OperaTAM(formula=form, algorithm='MLPOL', date_col=ctx.date_col)
                
                test_log.append({
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    "Phase": "Minimax_Aggregation", "Model_Name": league_name,
                    "Model_Type": "OperaTAM", "Validation_RMSE": np.nan, "Formula": form, "Hyperparameters": str({"league": league_name})
                })
                
                cols = [ctx.target] + ([ctx.date_col] if ctx.date_col in df_p.columns else []) + valid_models
                res = opera.predict_online(df_p[cols])
                
                weights_df = res[[c for c in res.columns if c.startswith('weight_')]].copy()
                if ctx.date_col and ctx.date_col in df_p.columns: weights_df.insert(0, ctx.date_col, df_p[ctx.date_col].values)
                reporter.export_opera_weights_trajectory(weights_df, league_name)
                
                ensemble_pred, weights = np.zeros(len(df_p)), {}
                for e in valid_models:
                    w_series = res[f"weight_{e}"]
                    ensemble_pred += df_p[e].values * w_series.values
                    if w_series.iloc[-1] >= self.ensemble_sparsity_threshold: 
                        weights[e] = w_series.iloc[-1]
                
                df_p[league_name] = ensemble_pred
                y_t = df_p[ctx.target].values
                
                ensemble_score = self._calculate_error(y_t, ensemble_pred, opt_metric)
                
                comp_base = 0
                for m in valid_models:
                    expert = next((ex for ex in trained_experts if ex["name"] == m), None)
                    if expert: 
                        comp_base += expert.get("complexity", 1)
                
                opera_comp = max(1, len(valid_models))
                total_comp = comp_base + opera_comp
                
                base_penalty_ratio = comp_base / max(1, n_fit_samples)
                opera_penalty_ratio = opera_comp / max(1, n_val_samples)
                
                penalized_ens = ensemble_score * (1.0 + penalty_factor * (base_penalty_ratio + opera_penalty_ratio))

                if ensemble_score != float('inf'):
                    self._update_log_rmse(test_log, league_name, ensemble_score, penalized_ens, total_comp)
                
                return weights

            if categories["static"]: league_weights["Static"] = run_single_opera("Ensemble_Static", categories["static"])
            if categories["kalman"]: league_weights["Kalman"] = run_single_opera("Ensemble_Kalman", categories["kalman"])
            if categories["adaptive"]: league_weights["Adaptive"] = run_single_opera("Ensemble_Adaptive", categories["adaptive"])

            island_bests = {}
            for exp in trained_experts:
                island = exp.get("island")
                if not island or pd.isna(exp["cv_rmse"]): continue
                if island not in island_bests or exp["penalized_score"] < island_bests[island]["penalized_score"]:
                    island_bests[island] = exp
                    
            sorted_bests = sorted(island_bests.values(), key=lambda x: x["penalized_score"])
            top_k_islands = sorted_bests[:4]
            
            best_island_names = [e["name"] for e in top_k_islands]
            if len(best_island_names) > 1:
                for top_name, best_exp in zip([f"Top{e['island']}" for e in top_k_islands], top_k_islands):
                    island_aliases[top_name] = best_exp["name"]
                    df_p[top_name] = df_p[best_exp["name"]]
                league_weights["Island_Federation"] = run_single_opera("Ensemble_Island_Federation", list(island_aliases.keys()))

            valid_experts = [e for e in trained_experts if pd.notna(e["cv_rmse"]) and e["name"] in df_p.columns]
            valid_experts.sort(key=lambda x: x["penalized_score"])
            top_apex_names = [e["name"] for e in valid_experts[:self.n_apex_experts]]
            
            if len(top_apex_names) > 1: weights_top10 = run_single_opera("AutoTAM_Apex_Ensemble", top_apex_names)
            elif top_apex_names: weights_top10 = {top_apex_names[0]: 1.0}
        else:
            trained_experts = [best_expert_global] if best_expert_global else []

        oof_predictions = {}
        for exp in trained_experts:
            try:
                p_full = self._get_expert_predictions(exp, exp["model"], df_cont_val, ctx, expander, return_full=True)
                oof_predictions[exp["name"]] = p_full
            except Exception as e:
                print(f"Warning: OOF Intercept failed for {exp['name']} | Error: {e}")

        for alias, real_name in island_aliases.items():
            if real_name in oof_predictions:
                oof_predictions[alias] = oof_predictions[real_name]

        for league_name, weights in league_weights.items():
            if weights:
                w_sum = sum(weights.values())
                league_preds = sum(oof_predictions[m] * (w / w_sum) for m, w in weights.items() if m in oof_predictions)
                final_name = league_name if league_name.startswith("Ensemble_") else f"Ensemble_{league_name}"
                oof_predictions[final_name] = league_preds
                
        if weights_top10:
            w_sum = sum(weights_top10.values())
            oof_predictions["AutoTAM_Apex_Ensemble"] = sum(oof_predictions[m] * (w / w_sum) for m, w in weights_top10.items() if m in oof_predictions)

        if refit_on_full_train:
            refitted_bases = set()
            for exp in trained_experts:
                m, m_type = exp["model"], exp["type"]
                try:
                    base_m = m if m_type == "static" else getattr(m, '_saved_base_model', getattr(m, 'base_model_', None))
                    if base_m and base_m not in refitted_bases:
                        f_ref = getattr(base_m, 'formula_', '')
                        req_cols = list(set([c for c in re.findall(r'[a-zA-Z0-9_]+', f_ref) if c in ctx.df_all_aug.columns]))
                        if ctx.target not in req_cols: req_cols.append(ctx.target)
                        
                        full_train_clean = ctx.df_all_aug.dropna(subset=req_cols).copy()
                        base_m.fit(full_train_clean) 
                        refitted_bases.add(base_m)
                except Exception as e: 
                    print(f"Warning: Native Refit failed for base model of {exp['name']} | Error: {e}")
        else:
            print("EnsembleSelector: Skipping historical refit (refit_on_full_train=False). Base models are frozen on Fit set.")
            
        return trained_experts, league_weights, weights_top10, island_aliases, oof_predictions
#: </evaluate_and_refit>

#: <ensemble_selector_helpers>
    def _get_expert_predictions(self, exp, m_ref, df_cont_val, ctx, expander, return_full=False):
        """
        Extracts predictions from an expert model across the specified validation space.
        """
        f_ref = getattr(m_ref, 'formula_', getattr(getattr(m_ref, '_saved_base_model', getattr(m_ref, 'base_model_', None)), 'formula_', ''))
        req_cols = list(set([c for c in re.findall(r'[a-zA-Z0-9_]+', f_ref) if c in df_cont_val.columns]))
        if ctx.target not in req_cols: req_cols.append(ctx.target)
        df_cv_clean = df_cont_val.dropna(subset=req_cols).copy()

        if exp["type"] in ["static", "island_champion"]: 
            preds = m_ref.predict(df_cv_clean)
            col = f"Estimated{ctx.target}"
        elif exp["type"] == "kalman": 
            preds = m_ref.predict_online(df_cv_clean)
            col = f"KalmanAdapted_{ctx.target}"
        elif exp["type"] == "adaptive": 
            base_m = getattr(m_ref, '_saved_base_model', None)
            d_adapt = expander._prepare_meta_learning_data(df_cv_clean, base_m, ctx)
            lag_cols = [c for c in d_adapt.columns if f"Residual{ctx.target}_lag_" in c]
            d_adapt = d_adapt.dropna(subset=lag_cols).copy()
            
            if hasattr(m_ref, 'predict_online'): preds = m_ref.predict_online(d_adapt)
            else:
                m_ref.prepare_simulation(d_adapt)
                m_ref.simulation()
                preds = m_ref.predictions_
            col = f"AdaptedEstimated{ctx.target}"
            
        if return_full:
            return preds[col].reindex(df_cont_val.index).values
        return preds[col].reindex(ctx.df_val.index).values

    def _update_log_rmse(self, test_log, name, score, penalized_score=None, complexity=None):
        """
        Updates the global chronological test log with the calculated validation score.
        Key is retained as Validation_RMSE for dashboard compatibility.
        """
        for log in test_log:
            if log.get("Model_Name") == name: 
                log["Validation_RMSE"] = score
                if penalized_score is not None:
                    log["Penalized_Score"] = penalized_score
                if complexity is not None:
                    log["Complexity"] = complexity
#: </ensemble_selector_helpers>
#: </ensemble_selector_class>