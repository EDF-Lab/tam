# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Bayesian Knowledge Graph for Auto-ML Component Tracking.

This module acts as the statistical brain of the framework. It tracks the empirical 
success of mathematical terms, explicitly prunes bloated genomes via variance 
decomposition and collinearity checks, and maintains statistical distributions 
of hyperparameters for optimal generative sampling.

Instead of blind random mutation, this graph enables the evolutionary engine 
to learn which feature-effect combinations yield the highest predictive power 
with the lowest complexity penalties.
"""

#: <knowledge_graph_imports>
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
#: </knowledge_graph_imports>

#: <knowledge_graph_class>
class KnowledgeGraph:
    """
    Tracks and guides the evolutionary generation of formula components.
    
    It maintains a bipartite-like graph mapping features to mathematical effects 
    (and their synergies). It uses an Exploration vs. Exploitation paradigm to 
    sample high-performing architectures while continuing to search for novel combinations.
    """
    
#: <knowledge_graph_init>
    def __init__(
        self, 
        exploration_rate: float = 0.2, 
        temperature: float = 1.0,
        prune_threshold: float = 0.01, 
        max_collinearity: float = 0.98
    ):
        """
        Initializes the Knowledge Graph with hyperparameters governing the evolutionary search.

        Args:
            exploration_rate (float): Probability of uniform random sampling (Epsilon-greedy).
                                      Ensures the algorithm doesn't get stuck in local optima.
            temperature (float): Softmax temperature for probabilistic sampling. Higher values 
                                 make selection more uniform; lower values make it greedier.
            prune_threshold (float): Minimum variance fraction to retain a term. If a term 
                                     explains less than this fraction of the total prediction variance, it is dropped.
            max_collinearity (float): Maximum allowed Pearson correlation before pruning.
                                      Prevents redundant terms from destabilizing the Conjugate Gradient solver.
        """
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.prune_threshold = prune_threshold
        self.max_collinearity = max_collinearity
        
        self.features: Dict[str, Dict[str, Any]] = defaultdict(self._default_metrics)
        self.effects: Dict[str, Dict[str, Any]] = defaultdict(self._default_metrics)
        self.feature_effect_edges: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(self._default_metrics)
        self.interaction_edges: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(self._default_metrics)
#: </knowledge_graph_init>

#: <knowledge_graph_default>
    def _default_metrics(self) -> Dict[str, Any]:
        """
        Initializes the base tracking metrics for graph nodes and edges.
        Tracks success (reward), structural complexity (penalty), and predictive power (variance).
        """
        return {
            'usage_count': 0.0,
            'survival_count': 0.0,
            'total_reward': 0.0,
            'avg_penalty': 0.0,
            'avg_variance': 0.0,
            'params_history': defaultdict(list)
        }
#: </knowledge_graph_default>

#: <knowledge_graph_update_prune>
    def update_and_prune(
        self, 
        parsed_terms: List[Dict[str, Any]], 
        model: Any, 
        df: pd.DataFrame, 
        target_col: str,
        global_rmse: float,
        target_std: float,
        component_penalties: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a genome, prunes redundant terms, and updates the knowledge graph.

        This is the core regularization mechanism. It decomposes the predictions of the GAM 
        into individual term contributions. Terms that explain negligible variance or are highly 
        collinear with existing terms are pruned to enforce parsimony.

        Args:
            parsed_terms: List of term dictionaries comprising the current formula.
            model: The fitted base model exposing a decompose_prediction method.
            df: Validation dataset.
            target_col: Target column name.
            global_rmse: Global validation RMSE of the model.
            target_std: Standard deviation of the target variable to ensure scale-invariant rewards.
            component_penalties: Dictionary mapping term signatures to their active penalty.

        Returns:
            List[Dict[str, Any]]: The strictly pruned list of formula terms.
        """
        component_penalties = component_penalties or {}
        
        try:
            contributions = model.decompose_prediction(df)
            preds_df = model.predict(df)
            pred_col = f"Estimated{target_col}" if f"Estimated{target_col}" in preds_df.columns else preds_df.columns[0]
            total_pred_var = np.var(preds_df[pred_col].values)
        except Exception:
            return parsed_terms

        pruned_terms = []
        active_effects = []
        
        normalized_rmse = global_rmse / (target_std + 1e-9)
        base_reward = 1.0 / (normalized_rmse + 1e-6)

        for term in parsed_terms:
            term_id = f"{term['type']}({term['feature']})"
            effect_values = contributions.get(term_id)

            if effect_values is None:
                pruned_terms.append(term)
                continue

            importance = np.var(effect_values) / (total_pred_var + 1e-9)
            term_std = np.std(effect_values)
            is_redundant = False

            if term_std > 0:
                for prev_effect in active_effects:
                    if np.std(prev_effect) > 0:
                        corr = np.abs(np.corrcoef(effect_values, prev_effect)[0, 1])
                        if corr > self.max_collinearity:
                            is_redundant = True
                            break

            if importance > self.prune_threshold and not is_redundant:
                pruned_terms.append(term)
                active_effects.append(effect_values)
                
                penalty = component_penalties.get(term_id, 0.0)
                composite_reward = base_reward * (1.0 + importance) / (1.0 + penalty)
                
                self._register_success(term, composite_reward, penalty, importance)

        return pruned_terms
#: </knowledge_graph_update_prune>

#: <knowledge_graph_register>
    def _register_success(self, term: Dict[str, Any], reward: float, penalty: float, variance: float) -> None:
        """
        Logs successful term metrics and parameters into the probabilistic hierarchy.
        Updates the global feature nodes, effect nodes, and the specific edges between them.
        """
        feat = term['feature']
        eff = term['type']
        params = term.get('params', {})

        interacting_feats = []
        if 'others' in params:
            others_str = params['others']
            interacting_feats = [s.strip() for s in str(others_str).split('|') if s.strip()]

        self._update_node(self.features[feat], reward, penalty, variance, params)
        self._update_node(self.effects[eff], reward, penalty, variance, params)
        self._update_node(self.feature_effect_edges[(feat, eff)], reward, penalty, variance, params)

        for interact_feat in interacting_feats:
            pair = tuple(sorted([feat, interact_feat]))
            self._update_node(self.interaction_edges[pair], reward, penalty, variance, {})
#: </knowledge_graph_register>

#: <knowledge_graph_update_node>
    def _update_node(
        self, 
        node: Dict[str, Any], 
        reward: float, 
        penalty: float, 
        variance: float, 
        params: Dict[str, Any]
    ) -> None:
        """
        Aggregates metrics and hyperparameter occurrences for a specific graph node.
        Uses numerically stable moving averages for penalty and variance tracking.
        """
        node['usage_count'] += 1
        node['total_reward'] += reward
        
        n = node['usage_count']
        node['avg_penalty'] += (penalty - node['avg_penalty']) / n
        node['avg_variance'] += (variance - node['avg_variance']) / n

        for k, v in params.items():
            if isinstance(v, (int, float)):
                node['params_history'][k].append(v)
#: </knowledge_graph_update_node>

#: <knowledge_graph_survival>
    def update_survival(self, parsed_terms: List[Dict[str, Any]], survived: bool) -> None:
        """
        Increments the survival count for components of algorithms retained post-pruning.
        This provides a strong evolutionary signal: terms that survive are heavily favored.
        """
        if not survived:
            return
            
        for term in parsed_terms:
            feat = term['feature']
            eff = term['type']
            
            self.features[feat]['survival_count'] += 1
            self.effects[eff]['survival_count'] += 1
            self.feature_effect_edges[(feat, eff)]['survival_count'] += 1
#: </knowledge_graph_survival>

#: <knowledge_graph_scoring>
    def _calculate_score(self, node: Dict[str, Any]) -> float:
        """
        Computes the probabilistic selection score based on historical performance.
        
        Score balances:
        - Reward (low RMSE / high variance explained)
        - Survival (proven resilience against pruning)
        - Penalty (structural complexity cost)
        """
        if node['usage_count'] == 0:
            return 1.0 
            
        avg_reward = node['total_reward'] / node['usage_count']
        survival_rate = node['survival_count'] / node['usage_count']
        
        score = (avg_reward * (1.0 + survival_rate)) / (1.0 + node['avg_penalty'])
        return score
#: </knowledge_graph_scoring>

#: <knowledge_graph_sampling>
    def suggest_effect_for_feature(self, feature: str, valid_effects: List[str]) -> str:
        """
        Samples an optimal mathematical effect for a given feature.
        Balances epsilon-greedy exploration with temperature-scaled exploitation.
        """
        if random.random() < self.exploration_rate:
            return random.choice(valid_effects)

        scores = {}
        for eff in valid_effects:
            edge_stats = self.feature_effect_edges.get((feature, eff), self._default_metrics())
            scores[eff] = self._calculate_score(edge_stats)

        return self._softmax_sample(scores)

    def suggest_parameters(self, feature: str, effect: str) -> Dict[str, Any]:
        """
        Returns the median consensus of hyperparameters for a specific term combination.
        Extracts the wisdom of the crowd from all surviving models.
        """
        node = self.feature_effect_edges.get((feature, effect))
        if not node or not node['params_history']:
            return {}

        consensus = {}
        for k, values in node['params_history'].items():
            if not values: continue
            median_val = np.median(values)
            consensus[k] = int(median_val) if isinstance(values[0], int) else float(median_val)
            
        return consensus

    def suggest_interaction(self, base_feature: str, available_features: List[str]) -> Optional[str]:
        """
        Samples an optimal interacting feature based on historical synergy.
        Used primarily by Deep Islands (Tree, RBF, Neural) to build interaction graphs.
        """
        if random.random() < self.exploration_rate:
             return random.choice(available_features) if available_features else None
             
        scores = {}
        for feat in available_features:
            pair = tuple(sorted([base_feature, feat]))
            edge_stats = self.interaction_edges.get(pair, self._default_metrics())
            scores[feat] = self._calculate_score(edge_stats)
            
        if not scores: return None
        return self._softmax_sample(scores)

    def _softmax_sample(self, scores_dict: Dict[str, float]) -> str:
        """
        Executes temperature-scaled Softmax selection.
        Converts raw historical scores into a probability distribution.
        """
        keys = list(scores_dict.keys())
        raw_scores = [scores_dict[k] for k in keys]
        max_score = max(raw_scores) if raw_scores else 0
        
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in raw_scores]
        total_exp = sum(exp_scores)
        
        if total_exp == 0:
            return random.choice(keys)
            
        probs = [e / total_exp for e in exp_scores]
        r = random.random()
        cumulative = 0.0
        
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return keys[i]
                
        return keys[-1]
#: </knowledge_graph_sampling>
#: </knowledge_graph_class>