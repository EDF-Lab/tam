# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Expert Expander for Automated TAM (AutoTAM).

Expands the top formulas from each evolutionary island into full 
static and dynamic state-spaces, including Kalman Filters and 
Adaptive Error Correction Models (ECMs).
"""

#: <expert_expander_imports>
import pandas as pd
import numpy as np
import datetime
import re
from typing import Dict, Any, Tuple, List
from .context import PipelineContext
from tam.common.utils import parse_formula_to_terms
from tam.model.additive import StaticTAM
from tam.model.kalman import KalmanTAM
from tam.model.adaptative import AdaptiveTAM
#: </expert_expander_imports>

#: <expert_expander_class>
class ExpertExpander:
    """Expands every island's top formulas into full static and dynamic state-spaces."""
    
#: <expert_expander_init>
    def __init__(self, expansions: Dict[str, bool] = None):
        self.expansions = expansions or {"prior": True, "autofit": True, "kalman": True, "adaptive": True, "grid": False}
#: </expert_expander_init>

#: <expert_expander_metric_calculator>
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
#: </expert_expander_metric_calculator>

#: <expert_expander_helpers>
    def _evaluate_model_cv(self, model, cv_folds, target_col, metric='rmse'):
        """Helper to calculate the Mean CV Score for a static base model."""
        scores = []
        for fold_train, fold_val in cv_folds:
            try:
                preds = model.predict(fold_val)[f"Estimated{target_col}"].values
                y_true = fold_val[target_col].values
                score = self._calculate_error(y_true, preds, metric)
                scores.append(score)
            except Exception:
                scores.append(float('inf'))
        return np.mean(scores) if scores else float('inf')
#: </expert_expander_helpers>

#: <expert_expander_generate>
    def generate_experts(self, island_champions: Dict[str, List[str]], ctx: PipelineContext, test_log: list, reporter) -> Dict[str, Any]:
        """
        Takes the best basic formulas and expands them into a diverse pool of 
        fully configured static and online-adaptive models.
        """
        candidate_pool = {}
        
        dfs_to_concat = [ctx.df_fit, ctx.df_dev]
        if ctx.df_val is not None: 
            dfs_to_concat.append(ctx.df_val)
        df_cont_all = pd.concat(dfs_to_concat)
        
        print(f"ExpertExpander: Exploding {sum(len(f) for f in island_champions.values())} Formulas into Dynamic States...")

        global_best_static = None
        global_best_cv = float('inf')
        opt_metric = getattr(ctx, 'optimization_metric', 'rmse')

        for island_name, base_formulas in island_champions.items():
            for rank, base_form in enumerate(base_formulas, 1):
                prefix = f"{island_name}_R{rank}"
                try:
                    all_tokens = re.findall(r'[a-zA-Z0-9_]+', base_form)
                    req_cols = list(set([c for c in all_tokens if c in df_cont_all.columns]))
                    if ctx.target not in req_cols: 
                        req_cols.append(ctx.target)
                    
                    df_fit_clean = df_cont_all.dropna(subset=req_cols).loc[ctx.df_fit.index.intersection(df_cont_all.index)].copy()
                    n_samples = len(df_fit_clean)

                    m_prior, m_auto, m_grid = None, None, None

                    if self.expansions.get("prior"):
                        try:
                            m_prior = StaticTAM(formula=base_form, group_col=ctx.group_col, date_col=ctx.date_col)
                            m_prior.fit(df_fit_clean)
                            cv_score = self._evaluate_model_cv(m_prior, ctx.cv_folds, ctx.target, metric=opt_metric)
                            comp = ctx.estimate_complexity(base_form)
                            pen_score = ctx.penalize_score(cv_score, base_form, n_samples)
                            name = f"Static_{prefix}_Prior"
                            candidate_pool[name] = {"type": "static", "model": m_prior, "island": island_name, "cv_rmse": cv_score}
                            self._log_test(test_log, name, "StaticTAM_priors", base_form, "Unpenalized_Ridge", cv_score, pen_score, comp)
                        except Exception: 
                            pass

                    if self.expansions.get("autofit"):
                        try:
                            m_auto = StaticTAM(formula=base_form, group_col=ctx.group_col, date_col=ctx.date_col)
                            m_auto.auto_fit(df_fit_clean)
                            cv_score = self._evaluate_model_cv(m_auto, ctx.cv_folds, ctx.target, metric=opt_metric)
                            comp = ctx.estimate_complexity(base_form)
                            pen_score = ctx.penalize_score(cv_score, base_form, n_samples)
                            name = f"Static_{prefix}_Auto"
                            candidate_pool[name] = {"type": "static", "model": m_auto, "island": island_name, "cv_rmse": cv_score}
                            self._log_test(test_log, name, "StaticTAM_gcv", base_form, "GCV_Auto_Penalized", cv_score, pen_score, comp)
                        except Exception: 
                            pass

                    if self.expansions.get("grid"):
                        try:
                            _, parsed_base = parse_formula_to_terms(base_form)
                            tokenized_terms, local_grid_config = [], {}
                            for term in parsed_base:
                                eff_type, term_feat = term['type'], term['feature']
                                feat_space = ctx.search_space.get(term_feat, {}).get("grids", {}).get(eff_type, {})
                                term_params = []
                                for p_name, p_val in term.get('params', {}).items():
                                    if p_name in feat_space and isinstance(feat_space[p_name], list) and len(feat_space[p_name]) > 1:
                                        token_name = f"grid_{p_name}_{term_feat}_{eff_type}"
                                        term_params.append(f"{p_name}='{token_name}'")
                                        local_grid_config[token_name] = feat_space[p_name]
                                    else:
                                        term_params.append(f"{p_name}='{p_val}'" if isinstance(p_val, str) else f"{p_name}={p_val}")
                                params_str = ", ".join(term_params)
                                tokenized_terms.append(f"{eff_type}({term_feat}, {params_str})" if params_str else f"{eff_type}({term_feat})")
                            
                            if local_grid_config:
                                tokenized_form = f"{ctx.target} ~ " + " + ".join(tokenized_terms)
                                m_grid_template = StaticTAM(formula=tokenized_form, group_col=ctx.group_col, date_col=ctx.date_col)
                                
                                m_grid = m_grid_template.grid_search_fit(cv_folds=ctx.cv_folds, grid_search_config=local_grid_config)
                                cv_score = self._evaluate_model_cv(m_grid, ctx.cv_folds, ctx.target, metric=opt_metric)
                                comp = ctx.estimate_complexity(tokenized_form)
                                pen_score = ctx.penalize_score(cv_score, tokenized_form, n_samples)
                                
                                name = f"Static_{prefix}_Grid"
                                candidate_pool[name] = {"type": "static", "model": m_grid, "island": island_name, "cv_rmse": cv_score}
                                self._log_test(test_log, name, "StaticTAM_grid", tokenized_form, str(local_grid_config), cv_score, pen_score, comp)
                        except Exception: 
                            pass

                    best_static_model, best_static_cv, min_penalized_cv = None, float('inf'), float('inf')
                    
                    for name, c_dict in candidate_pool.items():
                        if prefix in name:
                            raw_cv = c_dict.get("cv_rmse", float('inf'))
                            current_formula = getattr(c_dict["model"], 'formula_', base_form)
                            penalized_cv = ctx.penalize_score(raw_cv, str(current_formula), n_samples)
                            if penalized_cv < min_penalized_cv:
                                min_penalized_cv = penalized_cv
                                best_static_cv = raw_cv
                                best_static_model = c_dict["model"]

                    if not best_static_model: 
                        continue
                    
                    if best_static_cv < global_best_cv:
                        global_best_cv = best_static_cv
                        global_best_static = best_static_model

                    try:
                        d_meta = self._prepare_meta_learning_data(df_cont_all, best_static_model, ctx)

                        if self.expansions.get("kalman"):
                            base_effects = [c for c in best_static_model.decompose_prediction(df_fit_clean).columns if c.startswith("effect_")]
                            dynamic_formula = f"{ctx.target} ~ {' + '.join(base_effects)} - 1"
                            combined_formula = f"{dynamic_formula} + {getattr(best_static_model, 'formula_', '')}"
                            comp = ctx.estimate_complexity(combined_formula)
                            pen_score = ctx.penalize_score(best_static_cv, combined_formula, n_samples)
                            
                            obs_noise = getattr(ctx, "kalman_priors", {}).get("observation_noise_var", 0.1)
                            p_init = getattr(ctx, "kalman_priors", {}).get("P_init_diag", 1.0)
                            offset_boost = getattr(ctx, "kalman_priors", {}).get("offset_boost", 100.0)
                            
                            dynamic_rates = [0, obs_noise * 1e-4, obs_noise * 1e-2, obs_noise, obs_noise * 10, obs_noise * 100]
                            
                            for rate in dynamic_rates:
                                try:
                                    kalman = KalmanTAM(
                                        base_model=best_static_model, kalman_formula=dynamic_formula, date_col=ctx.date_col,
                                        horizon_steps=ctx.inferred_horizon, 
                                        offset_boost=offset_boost, 
                                        process_noise_var=rate, 
                                        observation_noise_var=obs_noise, 
                                        P_init_diag=p_init
                                    )
                                    params = {
                                        "offset_boost": offset_boost, 
                                        "process_noise_var": rate, 
                                        "observation_noise_var": obs_noise, 
                                        "P_init_diag": p_init
                                    }
                                    name_kalman = f"Kalman_{str(rate).replace('.', '_')}_{prefix}"
                                    candidate_pool[name_kalman] = {"type": "kalman", "model": kalman, "params": params, "island": island_name, "dynamic_formula": dynamic_formula, "cv_rmse": best_static_cv}
                                    self._log_test(test_log, name_kalman, "KalmanTAM", dynamic_formula, str(params), best_static_cv, pen_score, comp)
                                except Exception: 
                                    pass

                        if self.expansions.get("adaptive"):
                            lag_cols = [c for c in d_meta.columns if f"Residual{ctx.target}_lag_" in c]
                            effect_cols = [c for c in d_meta.columns if c.startswith("effect_")]
                            formulas_to_test = {}
                            if lag_cols: 
                                formulas_to_test["AR_Only"] = f"Residual{ctx.target} ~ " + " + ".join([f"l({c})" for c in lag_cols])
                            if effect_cols: 
                                formulas_to_test["Effects_Only"] = f"Residual{ctx.target} ~ " + " + ".join([f"l({c})" for c in effect_cols])
                            if lag_cols and effect_cols: 
                                formulas_to_test["Full_ECM"] = f"Residual{ctx.target} ~ " + " + ".join([f"l({c})" for c in lag_cols]) + " + " + " + ".join([f"l({c})" for c in effect_cols])

                            for variant_name, dyn_form in formulas_to_test.items():
                                combined_formula = f"{dyn_form} + {getattr(best_static_model, 'formula_', '')}"
                                comp = ctx.estimate_complexity(combined_formula)
                                pen_score = ctx.penalize_score(best_static_cv, combined_formula, n_samples)
                                
                                for periods in ctx.candidate_windows:
                                    try:
                                        adapt = AdaptiveTAM(
                                            base_model=best_static_model, adaptive_formula=dyn_form, update_interval_periods=1,
                                            training_window_periods=periods, steps_per_period=ctx.steps_per_period, horizon_steps=ctx.inferred_horizon
                                        )
                                        adapt._saved_base_model = best_static_model
                                        params = {"training_window_periods": periods, "architecture": variant_name}
                                        name_adapt = f"Adaptive_{variant_name}_{periods}p_{prefix}"
                                        candidate_pool[name_adapt] = {"type": "adaptive", "model": adapt, "params": params, "island": island_name, "dynamic_formula": dyn_form, "cv_rmse": best_static_cv}
                                        self._log_test(test_log, name_adapt, f"AdaptiveTAM_{variant_name}", dyn_form, str(params), best_static_cv, pen_score, comp)
                                    except Exception: 
                                        pass
                                    
                    except Exception:
                        continue

                except Exception:
                    continue

        if global_best_static and ctx.df_val is not None:
            try:
                terms_df = global_best_static.decompose_prediction(ctx.df_val)
                e_cols = [c for c in terms_df.columns if c.startswith("effect_")]
                pdp_stats = [{"Feature_Effect": c.replace("effect_", ""), "Mean_Contribution": float(terms_df[c].mean()), "Max_Contribution": float(terms_df[c].max()), "Variance": float(terms_df[c].var())} for c in e_cols]
                reporter.export_pdp_variance(pd.DataFrame(pdp_stats).sort_values(by="Variance", ascending=False), "Global_Static_Champion")
            except Exception: 
                pass

        return candidate_pool
#: </expert_expander_generate>

#: <expert_expander_prepare_data>
    def _prepare_meta_learning_data(self, df: pd.DataFrame, base_model, ctx: PipelineContext):
        """
        Calculates the pure mathematical components of the base model on the 
        continuous validation space to support residual tracking and state-space updates.
        """
        cols_to_drop = [c for c in df.columns if c.startswith("effect_") or c.startswith(f"Residual{ctx.target}")]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        terms = base_model.decompose_prediction(df.copy())
        e_cols = [c for c in terms.columns if c.startswith("effect_")]
        terms[f"Estimated{ctx.target}"] = terms[e_cols].sum(axis=1)
        d_meta = pd.concat([df, terms[e_cols]], axis=1)
        d_meta[f'Residual{ctx.target}_calc'] = d_meta[ctx.target] - terms[f"Estimated{ctx.target}"].values
        
        ctx.steps_per_period = 1
        if ctx.date_col and ctx.date_col in d_meta.columns:
            temp_df = d_meta.sort_values(by=[ctx.group_col, ctx.date_col]) if ctx.group_col else d_meta.sort_values(by=ctx.date_col)
            deltas = temp_df.groupby(ctx.group_col)[ctx.date_col].diff().dropna() if ctx.group_col else temp_df[ctx.date_col].diff().dropna()
            if not deltas.empty:
                median_delta = deltas.median()
                if pd.notnull(median_delta) and median_delta.total_seconds() > 0 and median_delta < pd.Timedelta(hours=23):
                    ctx.steps_per_period = max(1, int(round(pd.Timedelta(days=1) / median_delta)))

        ctx.adaptive_lags = ctx.lags if ctx.lags else [1 * ctx.steps_per_period, 7 * ctx.steps_per_period]
        ctx.inferred_horizon = min(ctx.adaptive_lags) if ctx.adaptive_lags else 1

        for lag in ctx.adaptive_lags:
            col_name = f'Residual{ctx.target}_lag_{lag}'
            if ctx.group_col and ctx.group_col in d_meta.columns:
                d_meta[col_name] = d_meta.groupby(ctx.group_col)[f'Residual{ctx.target}_calc'].shift(lag)
            else:
                d_meta[col_name] = d_meta[f'Residual{ctx.target}_calc'].shift(lag)
                
        d_meta[f'Residual{ctx.target}'] = d_meta[f'Residual{ctx.target}_calc']

        est_train_len = (len(d_meta) * 2) // 3
        
        scaled_windows = [
            7 * ctx.steps_per_period, 
            30 * ctx.steps_per_period, 
            90 * ctx.steps_per_period, 
            365 * ctx.steps_per_period
        ]
        base_windows = sorted(list(set([w for w in scaled_windows if w > 0])))
        
        max_allowed = max(5, est_train_len // 2)
        ctx.candidate_windows = [w for w in base_windows if w <= max_allowed] or [max_allowed]
        return d_meta
#: </expert_expander_prepare_data>

#: <expert_expander_log>
    def _log_test(self, test_log: list, name: str, m_type: str, formula: str, params: str, cv_score: float, penalized_score: float = None, complexity: int = None):
        test_log.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Phase": "Expert_Expansion", "Model_Name": name, "Model_Type": m_type, 
            "Validation_RMSE": cv_score, "Penalized_Score": penalized_score, "Complexity": complexity,
            "Formula": formula, "Hyperparameters": params
        })
#: </expert_expander_log>
#: </expert_expander_class>