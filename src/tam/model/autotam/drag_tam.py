# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Evolutionary Optimizer (DragTAM) for TAM.

This module orchestrates the distributed evolutionary search across distinct 
mathematical islands. It acts as an Estimation of Distribution Algorithm (EDA), 
utilizing a Bayesian Knowledge Graph to track the empirical success of terms, 
inform component sampling, and execute parsimonious pruning of bloated genomes.
"""

#: <dragtam_imports>
import pandas as pd
import numpy as np
import random
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
        max_complexity_multiplier: float = 5.0,
        group_col: str = None,
        date_col: str = None,
        fresh_fraction: float = 0.25
    ):
        """
        Initializes the Evolutionary Engine.
        """
        self.target_col = target_col
        # Preserve group_col during evaluation to prevent structural collapse on multi-level panel data.
        self.group_col = group_col
        self.date_col = date_col
        self.population_size = population_size
        self.n_generations = n_generations
        self.survival_rate = survival_rate
        self.alpha_complexity = alpha_complexity
        self.temperature = temperature
        self.max_complexity_multiplier = max_complexity_multiplier
        # Share of every generation reserved for fresh Mutant-UCB spawns (see optimize).
        self.fresh_fraction = fresh_fraction
        self.ucb_E = 2.0   # Mutant-UCB exploration parameter E in mu_k + sqrt(E / N_k)
        # Enforce minimum pulls and epsilon-greedy exploration to prevent premature UCB convergence on a single island.
        self.ucb_min_pulls = 3
        self.ucb_epsilon = 0.15
        
        self.kg = KnowledgeGraph(exploration_rate=exploration_rate, temperature=self.temperature)
        self.population: List[str] = []
        self.history: List[Dict[str, Any]] = []

        self.evaluation_cache: Dict[str, Tuple[float, Any, List[Dict[str, Any]], Dict[str, float]]] = {}
        # Pruned genome per evaluated formula, so a cached elite is not re-pruned (and re-credited).
        self.pruned_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Formulas whose survival has already been credited to the Knowledge Graph.
        self.survival_credited: set = set()
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

    def _required_columns(self, candidate_formula: str, available) -> List[str]:
        """Columns a candidate fit needs: formula tokens + target + group/date keys.

        The grouping and date keys are rarely named inside the formula, so they
        must be added explicitly or the grouped StaticTAM loses them after the
        column selection/dropna.
        """
        req = [c for c in re.findall(r'[a-zA-Z0-9_]+', candidate_formula) if c in available]
        for extra in (self.target_col, self.group_col, self.date_col):
            if extra and extra in available and extra not in req:
                req.append(extra)
        return list(dict.fromkeys(req))

#: <dragtam_evaluate>
    def _evaluate_candidate(self, candidate_formula: str, cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]], metric: str = 'rmse') -> Tuple[float, Any, List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluates a formula across all temporal cross-validation folds to prevent overfitting.
        """
        rmses = []
        
        for fold_train, fold_val in cv_folds:
            try:
                model = StaticTAM(candidate_formula, group_col=self.group_col, date_col=self.date_col)

                req_cols = self._required_columns(candidate_formula, fold_train.columns)

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
                req_cols = self._required_columns(candidate_formula, last_train.columns)

                train_clean = last_train[req_cols].copy()
                train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                train_clean.dropna(inplace=True)

                if len(train_clean) >= min_train_rows:
                    final_model = StaticTAM(candidate_formula, group_col=self.group_col, date_col=self.date_col)
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
        islands: List[Any],
        search_space: Dict[str, Any],
        metric: str = 'rmse'
    ) -> str:
        """
        Mutant-UCB evolutionary loop (Island-bandit + Quality-Diversity).

        Islands are the bandit arms. Each round the orchestrator picks an Island via UCB-E
        (mu_k + sqrt(E / N_k)); with the budget probability p_t = 1 - g/G it performs an
        evolutionary jump (mutate the Island's archived champion) instead of generating
        fresh from the KnowledgeGraph. Surviving genomes feed the TED-diverse Champion Archive.
        """
        if not cv_folds:
            raise ValueError("cv_folds cannot be empty.")

        available_feats = [f for f in cv_folds[0][0].columns if f in search_space]

        # Rely on the selector grid for capacity; leave basis family choices to the UCB bandit to maintain OPERA diversity.

        target_std = float(cv_folds[0][0][self.target_col].std())
        if pd.isna(target_std) or target_std == 0.0:
            target_std = 1.0

        val_df_combined = pd.concat([fold[1] for fold in cv_folds]).drop_duplicates()

        # An Island that reports it cannot build a candidate on these features is left out of the
        # bandit: its minimum budget would be spent on empty candidates.
        islands = [isl for isl in islands
                   if getattr(isl, 'is_viable', None) is None or isl.is_viable(available_feats, search_space)]
        island_by_name = {isl.name: isl for isl in islands}
        island_names = list(island_by_name.keys())
        G = max(1, self.n_generations)

        def _spawn(island_name: str, p_t: float) -> Tuple[str, str]:
            """Generate-or-mutate one candidate from the chosen Island; returns (formula, island)."""
            isl = island_by_name[island_name]
            if random.random() < p_t and isl.champion_genome:
                rhs = isl.mutate(self.kg, available_feats, search_space)       # evolutionary jump
            else:
                rhs = isl.generate(self.kg, available_feats, search_space,
                                   complexity_cap=(random.random() < 0.20))
            rhs = rhs if rhs != "1" else ""
            return self._build_canonical_formula(rhs), island_name

        # Generation 0: seed via UCB (initially explores every island once) under strict parsimony.
        self.population = []
        for _ in range(self.population_size):
            name = self.kg.select_island_ucb(island_names, self.ucb_E, self.ucb_epsilon, self.ucb_min_pulls)
            isl = island_by_name[name]
            rhs = isl.generate(self.kg, available_feats, search_space, complexity_cap=True)
            rhs = rhs if rhs != "1" else ""
            self.population.append((self._build_canonical_formula(rhs), name))

        evaluated_population = []
        for generation in range(G):
            evaluated_population = []

            progress_ratio = generation / max(1, G - 1)
            current_multiplier = 1.0 + (self.max_complexity_multiplier - 1.0) * progress_ratio
            active_alpha = self.alpha_complexity * current_multiplier
            p_t = 1.0 - (generation / G)   # Mutant-UCB budget: high mutation early, exploit late

            for formula, island_name in self.population:
                is_new = formula not in self.evaluation_cache
                if is_new:
                    self.evaluation_cache[formula] = self._evaluate_candidate(formula, cv_folds, metric)
                rmse, model, parsed_terms, penalties = self.evaluation_cache[formula]

                total_penalty = sum(penalties.values()) if penalties else 0.0
                penalized = rmse * (1.0 + (active_alpha * total_penalty))

                # Credit evidence only on first evaluation to prevent cached elites from artificially inflating their island's UCB reward.
                if is_new:
                    # Island-UCB reward (higher = better): inverse scale-normalized error.
                    # A failed fit is recorded as a zero-reward pull (see update_island_reward).
                    reward = 1.0 / (rmse / target_std + 1e-6) if np.isfinite(rmse) else None
                    self.kg.update_island_reward(island_name, reward)

                    self.pruned_cache[formula] = self.kg.update_and_prune(
                        parsed_terms=parsed_terms,
                        model=model,
                        df=val_df_combined,
                        target_col=self.target_col,
                        global_rmse=rmse,
                        target_std=target_std,
                        component_penalties=penalties
                    )

                    # Quality-Diversity admission keyed on the penalized score.
                    if np.isfinite(rmse) and self.pruned_cache[formula]:
                        self.kg.try_add_champion(self.pruned_cache[formula], penalized, island_name)

                pruned_terms = self.pruned_cache.get(formula, parsed_terms)

                evaluated_population.append({
                    'original_formula': formula,
                    'island': island_name,
                    'pruned_terms': pruned_terms,
                    'rmse': rmse,
                    'total_penalty': total_penalty
                })

            evaluated_population.sort(key=lambda x: x['rmse'] * (1.0 + (active_alpha * x['total_penalty'])))

            best_gen = evaluated_population[0]
            self.history.append({
                "Generation": generation + 1,
                "Budget": ((generation + 1) / G) * 100.0,
                "Validation_RMSE": best_gen["rmse"],
                "Formula": best_gen["original_formula"]
            })

            n_survivors = max(1, int(self.population_size * self.survival_rate))
            survivors = evaluated_population[:n_survivors]
            for survivor in survivors:
                # Credited once per formula for the same reason as the reward above; otherwise
                # survival_count outgrows usage_count and the score saturates on the elites.
                if survivor['original_formula'] not in self.survival_credited:
                    self.kg.update_survival(survivor['pruned_terms'], survived=True)
                    self.survival_credited.add(survivor['original_formula'])

            # Refresh each Island's reigning champion from the diverse archive.
            for isl in islands:
                champ = self.kg.get_island_champion(isl.name)
                if champ:
                    isl.set_champion(champ)

            # Reserve a strict budget for fresh spawns to prevent cached elites and ablations from stalling the search.
            n_fresh = max(1, int(round(self.population_size * self.fresh_fraction)))
            carry_limit = max(1, self.population_size - n_fresh)

            self.population = []
            existing = set()
            for survivor in survivors:
                reconstructed = self._reconstruct_formula(survivor['pruned_terms'])
                if reconstructed and reconstructed not in existing and len(self.population) < carry_limit:
                    self.population.append((reconstructed, survivor['island']))
                    existing.add(reconstructed)

            for survivor in survivors:
                if len(self.population) >= carry_limit:
                    break
                if len(survivor['pruned_terms']) > 1:
                    sorted_terms = sorted(survivor['pruned_terms'], key=self.kg.term_avg_variance)
                    ablated_formula = self._reconstruct_formula(sorted_terms[1:])
                    # An ablation that was already evaluated carries no new information.
                    if (ablated_formula and ablated_formula not in existing
                            and ablated_formula not in self.evaluation_cache):
                        self.population.append((ablated_formula, survivor['island']))
                        existing.add(ablated_formula)

            # Repopulate via Mutant-UCB: UCB-E picks the Island, p_t decides mutate vs. generate.
            # A spawn duplicating an already-evaluated formula is redrawn, a bounded number of times.
            redraws = 0
            while len(self.population) < self.population_size:
                name = self.kg.select_island_ucb(island_names, self.ucb_E, self.ucb_epsilon, self.ucb_min_pulls)
                candidate = _spawn(name, p_t)
                if ((candidate[0] in existing or candidate[0] in self.evaluation_cache)
                        and redraws < 5 * self.population_size):
                    redraws += 1
                    continue
                self.population.append(candidate)
                existing.add(candidate[0])

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
                # keep only real sub-term strings, e.g. "s(x1, k=10)"; drop the parser's
                # internal "__sub_*" bookkeeping keys that would otherwise corrupt the formula.
                sub_terms = [k for k in params.keys() if re.match(r'^\s*[a-zA-Z]{1,4}\s*\(', str(k))]
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