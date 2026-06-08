#: <dragtam_module_doc>
"""
Evolutionary Optimizer (DragTAM) for TAM.

This module orchestrates the distributed evolutionary search across distinct 
mathematical islands. It acts as an Estimation of Distribution Algorithm (EDA), 
utilizing a Bayesian Knowledge Graph to track the empirical success of terms, 
inform component sampling, and execute parsimonious pruning of bloated genomes.
"""
#: </dragtam_module_doc>

#: <dragtam_imports>
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
import re

from .knowledge_graph import KnowledgeGraph
from tam.common.utils import parse_formula_to_terms
from tam.model.additive import StaticTAM
#: </dragtam_imports>

#: <dragtam_class>
class DragTAM:
    """
    Evolutionary orchestrator for autonomous Generalized Additive Model (GAM) selection.
    
    Instead of a brute-force grid search across all possible mathematical topologies,
    DragTAM evolves a population of formulas over multiple generations. It iteratively 
    trains candidates, prunes statistically insignificant terms via variance decomposition, 
    and breeds new formulas using the probabilistic weights stored in the Knowledge Graph.
    """

#: <dragtam_init>
    def __init__(
        self, 
        target_col: str,
        population_size: int = 50,
        n_generations: int = 20,
        survival_rate: float = 0.5,
        exploration_rate: float = 0.2,
        alpha_complexity: float = 0.01,
        temperature: float = 1.0,
        max_complexity_multiplier: float = 5.0
    ):
        """
        Initializes the Evolutionary Engine.
        """
        self.target_col = target_col
        self.population_size = population_size
        self.n_generations = n_generations
        self.survival_rate = survival_rate
        self.alpha_complexity = alpha_complexity
        self.temperature = temperature
        self.max_complexity_multiplier = max_complexity_multiplier
        
        self.kg = KnowledgeGraph(exploration_rate=exploration_rate, temperature=self.temperature)
        self.population: List[str] = []
        self.history: List[Dict[str, Any]] = []

        self.evaluation_cache: Dict[str, Tuple[float, Any, List[Dict[str, Any]], Dict[str, float]]] = {}
#: </dragtam_init>

    def _build_canonical_formula(self, rhs: str) -> str:
        """
        Sorts the Right-Hand Side (RHS) terms alphabetically to ensure 
        formulas are canonicalized upon creation, preventing duplicate evaluations.
        """
        if not rhs or rhs == "1":
            return f"{self.target_col} ~ "
            
        terms = sorted([t.strip() for t in rhs.split("+") if t.strip() and t.strip() != "1"])
        
        rhs_clean = " + ".join(terms)
        return f"{self.target_col} ~ {rhs_clean}" if rhs_clean else f"{self.target_col} ~ "

#: <dragtam_metric_calculator>
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
#: </dragtam_metric_calculator>

#: <dragtam_evaluate>
    def _evaluate_candidate(self, candidate_formula: str, cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]], metric: str = 'rmse') -> Tuple[float, Any, List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluates a formula across all temporal cross-validation folds to prevent overfitting.
        """
        rmses = []
        
        for fold_train, fold_val in cv_folds:
            try:
                model = StaticTAM(candidate_formula)
                
                req_cols = list(set([c for c in re.findall(r'[a-zA-Z0-9_]+', candidate_formula) if c in fold_train.columns]))
                if self.target_col not in req_cols: 
                    req_cols.append(self.target_col)
                
                train_clean = fold_train[req_cols].copy()
                val_clean = fold_val[req_cols].copy()
                
                train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                val_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                train_clean.dropna(inplace=True)
                val_clean.dropna(inplace=True)

                rhs = candidate_formula.split("~")[1] if "~" in candidate_formula else candidate_formula
                num_terms = len([t for t in rhs.split("+") if t.strip()]) + 1 
                
                min_train_rows = max(5, num_terms * 2) 
                min_val_rows = 2

                if len(train_clean) < min_train_rows or len(val_clean) < min_val_rows:
                    continue

                model.fit(train_clean)
                preds = model.predict(val_clean)
                
                y_true = val_clean[self.target_col].values
                y_pred = preds[f"Estimated{self.target_col}"].values
                
                error_val = self._calculate_error(y_true, y_pred, metric)
                rmses.append(error_val)
                    
            except Exception:
                continue

        mean_rmse = np.mean(rmses) if rmses else float('inf')
        
        final_model = None
        parsed_terms = []
        penalties = {}
        
        if mean_rmse != float('inf'):
            try:
                last_train = cv_folds[-1][0]
                req_cols = list(set([c for c in re.findall(r'[a-zA-Z0-9_]+', candidate_formula) if c in last_train.columns]))
                if self.target_col not in req_cols: 
                    req_cols.append(self.target_col)
                    
                train_clean = last_train[req_cols].copy()
                train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                train_clean.dropna(inplace=True)
                
                if len(train_clean) >= min_train_rows:
                    final_model = StaticTAM(candidate_formula)
                    final_model.fit(train_clean)
                    parsed_terms = getattr(final_model, 'parsed_terms_', [])
                    penalties = getattr(final_model, 'component_penalties_', {})
            except Exception:
                mean_rmse = float('inf')

        return mean_rmse, final_model, parsed_terms, penalties
#: </dragtam_evaluate>

#: <dragtam_optimize>
    def optimize(
        self, 
        cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]], 
        island_generators: List[Callable],
        search_space: Dict[str, Any],
        metric: str = 'rmse'
    ) -> str:
        """
        Executes the main evolutionary loop guided by the Knowledge Graph.
        """
        if not cv_folds:
            raise ValueError("cv_folds cannot be empty.")
            
        available_feats = [f for f in cv_folds[0][0].columns if f in search_space]
        self.population = []
        
        target_std = float(cv_folds[0][0][self.target_col].std())
        if pd.isna(target_std) or target_std == 0.0:
            target_std = 1.0
            
        val_df_combined = pd.concat([fold[1] for fold in cv_folds]).drop_duplicates()
        
        # Generation 0: Strictly enforce complexity caps to establish simple baselines
        for gen in np.random.choice(island_generators, self.population_size):
            rhs = gen(self.kg, available_feats, search_space, complexity_cap=True)
            rhs = rhs if rhs != "1" else ""
            self.population.append(self._build_canonical_formula(rhs))

        for generation in range(self.n_generations):
            evaluated_population = []
            
            # Dynamic Annealing: Linearly scale the complexity penalty multiplier
            # from 1.0 at generation 0 to max_complexity_multiplier at the final generation.
            progress_ratio = generation / max(1, self.n_generations - 1)
            current_multiplier = 1.0 + (self.max_complexity_multiplier - 1.0) * progress_ratio
            active_alpha = self.alpha_complexity * current_multiplier

            for formula in self.population:
                if formula in self.evaluation_cache:
                    rmse, model, parsed_terms, penalties = self.evaluation_cache[formula]
                else:
                    rmse, model, parsed_terms, penalties = self._evaluate_candidate(formula, cv_folds, metric)
                    self.evaluation_cache[formula] = (rmse, model, parsed_terms, penalties)
                
                pruned_terms = self.kg.update_and_prune(
                    parsed_terms=parsed_terms,
                    model=model,
                    df=val_df_combined,
                    target_col=self.target_col,
                    global_rmse=rmse,
                    target_std=target_std,
                    component_penalties=penalties
                )
                
                total_penalty = sum(penalties.values()) if penalties else 0.0

                evaluated_population.append({
                    'original_formula': formula,
                    'pruned_terms': pruned_terms,
                    'rmse': rmse,  
                    'total_penalty': total_penalty
                })

            # Sort the population using the dynamically annealed alpha
            evaluated_population.sort(key=lambda x: x['rmse'] * (1.0 + (active_alpha * x['total_penalty'])))
            
            best_gen = evaluated_population[0]
            self.history.append({
                "Generation": generation + 1,
                "Budget": ((generation + 1) / self.n_generations) * 100.0,
                "Validation_RMSE": best_gen["rmse"],  
                "Formula": best_gen["original_formula"]
            })
            
            n_survivors = max(1, int(self.population_size * self.survival_rate))
            survivors = evaluated_population[:n_survivors]
            
            for survivor in survivors:
                self.kg.update_survival(survivor['pruned_terms'], survived=True)

            self.population = []
            for survivor in survivors:
                reconstructed = self._reconstruct_formula(survivor['pruned_terms'])
                if reconstructed:
                    self.population.append(reconstructed)
                    
                if len(survivor['pruned_terms']) > 1:
                    sorted_terms = sorted(
                        survivor['pruned_terms'],
                        key=lambda t: self.kg.features[t['feature']]['avg_variance']
                    )
                    ablated_terms = sorted_terms[1:]
                    ablated_formula = self._reconstruct_formula(ablated_terms)
                    if ablated_formula and ablated_formula not in self.population:
                        self.population.append(ablated_formula)

            # Repopulation
            while len(self.population) < self.population_size:
                generator = np.random.choice(island_generators)
                apply_cap = np.random.rand() < 0.20
                rhs = generator(self.kg, available_feats, search_space, complexity_cap=apply_cap)
                self.population.append(self._build_canonical_formula(rhs))

        best_candidate = evaluated_population[0]
        return self._reconstruct_formula(best_candidate['pruned_terms'])
#: </dragtam_optimize>

#: <dragtam_reconstruct>
    def _reconstruct_formula(self, parsed_terms: List[Dict[str, Any]]) -> str:
        """
        Reconstructs a valid, native StaticTAM formula string from a list of term dictionaries.
        Used to translate the pruned genome back into a parsable model state.
        """
        terms = []
        for t in parsed_terms:
            eff = t['type']
            feat = t['feature']
            params = t.get('params', {})
            
            if eff == 'te':
                sub_terms = list(params.keys())
                terms.append(f"te({', '.join(sub_terms)})")
            else:
                param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
                if param_str:
                    terms.append(f"{eff}({feat}, {param_str})")
                else:
                    terms.append(f"{eff}({feat})")
                
        terms.sort()
        rhs = " + ".join(terms)
        return f"{self.target_col} ~ {rhs}" if rhs else f"{self.target_col} ~ "
#: </dragtam_reconstruct>
#: </dragtam_class>