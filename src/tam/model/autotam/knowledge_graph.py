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
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

from tam.common.utils import parse_formula_to_terms
from tam.model._math import decomposition_names
from tam.model.spectrum import OffsetEffect
from .parser import terms_are_equivalent, term_subsumes
#: </knowledge_graph_imports>

#: <knowledge_graph_term_identity>
_SUB_TERM = re.compile(r'^\s*[a-zA-Z]{1,4}\s*\(')


def term_members(term: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(feature, basis) pairs a formula term is built from.

    A marginal term is its own single member. The formula parser names every tensor product's
    feature 'interaction' and keeps its sub-terms only as parameter keys, so they are recovered
    here; otherwise the graph files every te() under one pseudo-feature and credits none of the
    real features it crosses.
    """
    if term.get('type') != 'te':
        return [(term.get('feature'), term.get('type'))]
    sub_terms = [str(key).strip() for key in term.get('params', {}) if _SUB_TERM.match(str(key))]
    if not sub_terms:
        return []
    try:
        _, parsed = parse_formula_to_terms("DUMMY ~ " + " + ".join(sub_terms))
    except ValueError:
        return []
    return [(sub['feature'], sub['type']) for sub in parsed]


def term_signature(term: Dict[str, Any]) -> str:
    """Stable identity of a term, e.g. 'f(toy)' or 'te(f(toy),c(day_type_week))'."""
    if term.get('type') != 'te':
        return f"{term.get('type')}({term.get('feature')})"
    return "te(" + ",".join(f"{effect}({feature})" for feature, effect in term_members(term)) + ")"
#: </knowledge_graph_term_identity>

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
        prune_threshold: float = 0.005,
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

        # Pre-search basis diagnosis: feature -> (effect, strength). Decays with evidence.
        self.basis_prior: Dict[str, Tuple[str, float]] = {}

        # --- Step 3: Mutant-UCB island bandit + Quality-Diversity Champion Archive ---
        # Island-UCB reward stats: empirical mean fitness mu_k and pull count N_k per island.
        self.island_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {'N': 0.0, 'reward_sum': 0.0})
        # Champion Archive (QD): structurally-distinct elite genomes, kept diverse via TED.
        self.champion_archive: List[Dict[str, Any]] = []
        self.archive_max_size: int = 50
        self.niche_radius: float = 0.15            # D_min in normalized TED units [0, 1]
        self.ted_weights: Tuple[float, float, float] = (1.0, 0.5, 0.25)  # (feat, topology, hyperparam)
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
        component_penalties: Optional[Dict[str, float]] = None,
        mandatory_terms: Optional[List[str]] = None,
        mandatory_variables: Optional[List[str]] = None
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
            mandatory_terms: List of term strings that must not be pruned.

        Returns:
            List[Dict[str, Any]]: The strictly pruned list of formula terms.
        """
        component_penalties = component_penalties or {}
        mandatory_terms = list(mandatory_terms or [])
        mandatory_variables = set(mandatory_variables or [])
        
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

        column_of = self._term_columns(parsed_terms, model)

        def _contribution_for(term: Dict[str, Any]):
            """Locate a term's decomposed contribution.

            When the model exposes its effects the column is resolved exactly (_term_columns).
            The name-based fallback serves models without an effects list; it once built
            '<type>(<feature>)', which matched no column, so nothing was ever pruned and
            _register_success never fired. A te() cannot be resolved by name at all: its parsed
            feature is the placeholder 'interaction'.
            """
            if column_of:
                key = column_of.get(id(term))
                return contributions[key] if key is not None and key in contributions else None
            if term.get('type') == 'te':
                return None
            feature = term.get("feature")
            for key in (f"effect_{feature}", f"{term['type']}({feature})"):
                if key in contributions:
                    return contributions[key]
            return None

        # Prune collinear terms strongest-first so interactions claim variance before their marginals.
        contributions_by_term = {id(term): _contribution_for(term) for term in parsed_terms}

        def _importance(term: Dict[str, Any]) -> float:
            values = contributions_by_term[id(term)]
            if values is None:
                return -np.inf
            share = float(np.var(values) / (total_pred_var + 1e-9))
            return share if np.isfinite(share) else -np.inf

        importances = {id(term): _importance(term) for term in parsed_terms}
        kept_ids = set()
        
        # Protect mandatory variables: ensure every mandatory variable has at least one term preserved
        if mandatory_variables:
            for var in mandatory_variables:
                terms_with_var = [term for term in parsed_terms if var in {m[0] for m in term_members(term)}]
                if terms_with_var:
                    best_term_for_var = max(terms_with_var, key=lambda t: importances[id(t)])
                    kept_ids.add(id(best_term_for_var))


        for term in sorted(parsed_terms, key=lambda t: importances[id(t)], reverse=True):
            is_mandatory_term = any(term_subsumes(term, mt) for mt in mandatory_terms)
            is_pre_protected = id(term) in kept_ids

            effect_values = contributions_by_term[id(term)]
            if effect_values is None:
                kept_ids.add(id(term))
                continue

            importance = importances[id(term)]
            is_redundant = False
            if np.std(effect_values) > 0:
                for prev_effect in active_effects:
                    if np.std(prev_effect) > 0:
                        corr = np.abs(np.corrcoef(effect_values, prev_effect)[0, 1])
                        if corr > self.max_collinearity:
                            is_redundant = True
                            break

            if is_mandatory_term or is_pre_protected or (importance > self.prune_threshold and not is_redundant):
                if id(term) not in kept_ids:
                    kept_ids.add(id(term))
                active_effects.append(effect_values)

                term_id = term_signature(term)
                penalty = component_penalties.get(term_id, 0.0)
                composite_reward = base_reward * (1.0 + importance) / (1.0 + penalty)
                self._register_success(term, composite_reward, penalty, importance)

        pruned_terms = [term for term in parsed_terms if id(term) in kept_ids]
        return pruned_terms

    @staticmethod
    def _term_columns(parsed_terms: List[Dict[str, Any]], model: Any) -> Dict[int, str]:
        """Maps each formula term (by object identity) to its decompose_prediction column.

        The effect factory builds exactly one effect per parsed term after the intercept, so the
        effects list without its offset is aligned with parsed_terms, and decomposition_names
        gives each effect's exact column. Returns {} when the model exposes no effects or the two
        lists disagree, leaving the caller on its name-based fallback.
        """
        effects = getattr(model, 'effects_list_', None)
        if not effects:
            return {}
        named = [(effect, name) for effect, name in zip(effects, decomposition_names(effects))
                 if not isinstance(effect, OffsetEffect)]
        if len(named) != len(parsed_terms):
            return {}
        return {id(term): f"effect_{name}" for term, (_, name) in zip(parsed_terms, named)}
#: </knowledge_graph_update_prune>

#: <knowledge_graph_register>
    def _register_success(self, term: Dict[str, Any], reward: float, penalty: float, variance: float) -> None:
        """
        Logs successful term metrics and parameters into the probabilistic hierarchy.
        Updates the global feature nodes, effect nodes, and the specific edges between them.

        Credit tensor products to all crossed features to guide future interaction sampling.
        """
        eff = term['type']
        if eff == 'te':
            members = [feature for feature, _ in term_members(term)]
            self._update_node(self.effects[eff], reward, penalty, variance, {})
            for feature in members:
                self._update_node(self.features[feature], reward, penalty, variance, {})
                self._update_node(self.feature_effect_edges[(feature, eff)], reward, penalty, variance, {})
            for i, first in enumerate(members):
                for second in members[i + 1:]:
                    pair = tuple(sorted([first, second]))
                    self._update_node(self.interaction_edges[pair], reward, penalty, variance, {})
            return

        feat = term['feature']
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
            eff = term['type']
            self.effects[eff]['survival_count'] += 1
            # A te() survives on behalf of every feature it crosses (see _register_success).
            for feat, _ in term_members(term):
                self.features[feat]['survival_count'] += 1
                self.feature_effect_edges[(feat, eff)]['survival_count'] += 1

    def term_avg_variance(self, term: Dict[str, Any]) -> float:
        """Variance share the graph has recorded for a term; DragTAM ablates the lowest first.

        A marginal reads its feature node. A te() has no feature of its own, so it reads the mean
        of its (feature, 'te') edges; reading the placeholder feature returned zero, which made
        every tensor product the first term stripped by ablation.
        """
        if term.get('type') != 'te':
            return self.features[term['feature']]['avg_variance']
        members = term_members(term)
        if not members:
            return 0.0
        return float(np.mean([self.feature_effect_edges[(feat, 'te')]['avg_variance'] for feat, _ in members]))
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
    def set_basis_prior(self, feature: str, effect: str, strength: float = 0.0) -> None:
        """Registers a basis-family prior for a feature. DISABLED by default (strength 0).

        Basis-family prior disabled by default: forcing a single family starves diversity and degrades OPERA ensemble performance.
        """
        self.basis_prior[feature] = (effect, float(strength))

    def _prior_probability(self, feature: str) -> float:
        """Prior weight for a feature, decaying as observations accumulate."""
        entry = self.basis_prior.get(feature)
        if not entry:
            return 0.0
        observations = sum(self.feature_effect_edges[(f, e)]['usage_count']
                           for (f, e) in self.feature_effect_edges if f == feature)
        return float(entry[1]) / (1.0 + observations / 10.0)

    def suggest_effect_for_feature(self, feature: str, valid_effects: List[str]) -> str:
        """
        Samples an optimal mathematical effect for a given feature.
        Balances epsilon-greedy exploration with temperature-scaled exploitation, after a
        decaying prior drawn from the pre-search basis diagnosis.
        """
        prior = self.basis_prior.get(feature)
        if prior and prior[0] in valid_effects and random.random() < self._prior_probability(feature):
            return prior[0]

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

#: <knowledge_graph_island_ucb>
    def update_island_reward(self, island_name: str, reward: Optional[float]) -> None:
        """Records one pull of an Island (Mutant-UCB arm) and its fitness reward (higher = better).

        A candidate that failed to fit (reward None or non-finite) is a pull with zero reward. Left
        uncounted, an Island whose candidates always fail would stay below the minimum budget of
        select_island_ucb and be chosen on every draw.
        """
        st = self.island_stats[island_name]
        st['N'] += 1.0
        if reward is not None and np.isfinite(reward):
            st['reward_sum'] += float(reward)

    def select_island_ucb(self, island_names: List[str], exploration: float = 2.0,
                          epsilon: float = 0.0, min_pulls: int = 1) -> str:
        """
        Selects the next Island to breed.

        1. Minimum budget: an Island pulled fewer than ``min_pulls`` times is chosen first, uniformly
           among such Islands, so every topology is tried before any is exploited.
        2. Epsilon-greedy: otherwise, with probability ``epsilon``, an Island is drawn uniformly.
        3. UCB-E: otherwise I_t in argmax_k { mu_k + sqrt(E / N_k) }.

        The defaults (min_pulls=1, epsilon=0) give plain UCB-E with unqueried Islands first.
        """
        if not island_names:
            raise ValueError("select_island_ucb requires at least one island name.")

        budget = max(1, int(min_pulls))           # at least one pull, so N_k > 0 in the UCB term
        under_budget = [n for n in island_names if self.island_stats[n]['N'] < budget]
        if under_budget:
            return random.choice(under_budget)
        if epsilon > 0.0 and random.random() < epsilon:
            return random.choice(island_names)

        best_name, best_ucb = island_names[0], -float('inf')
        for name in island_names:
            st = self.island_stats[name]
            mu = st['reward_sum'] / st['N']
            ucb = mu + math.sqrt(exploration / st['N'])
            if ucb > best_ucb:
                best_ucb, best_name = ucb, name
        return best_name
#: </knowledge_graph_island_ucb>

#: <knowledge_graph_ted>
    @staticmethod
    def _term_key(term: Dict[str, Any]) -> Tuple[str, str]:
        # Every te() parses to the feature 'interaction'; keying on it made all tensor products
        # topologically identical, so distinct interactions competed for a single archive niche.
        if term.get('type') == 'te':
            return (term_signature(term), 'te')
        return (term.get('feature'), term.get('type'))

    @staticmethod
    def _genome_features(genome: List[Dict[str, Any]]) -> set:
        return {feat for term in genome for feat, _ in term_members(term)}

    def topological_edit_distance(self, genome_a: List[Dict[str, Any]], genome_b: List[Dict[str, Any]]) -> float:
        """
        Topological Edit Distance (TED) between two formula DAGs, normalized to [0, 1]:

            TED = (w1 * d_feat + w2 * d_top + w3 * d_hyp) / (w1 + w2 + w3)

        * d_feat: Jaccard distance of active input features (highest weight).
        * d_top : Jaccard distance of applied (feature, basis) pairs (e.g. s() -> w()).
        * d_hyp : mean normalized hyperparameter distance over shared (feature, basis) terms.
        """
        if not genome_a and not genome_b:
            return 0.0

        feats_a = self._genome_features(genome_a)
        feats_b = self._genome_features(genome_b)
        union_f = feats_a | feats_b
        d_feat = len(feats_a ^ feats_b) / len(union_f) if union_f else 0.0

        keys_a = {self._term_key(t) for t in genome_a}
        keys_b = {self._term_key(t) for t in genome_b}
        union_k = keys_a | keys_b
        d_top = len(keys_a ^ keys_b) / len(union_k) if union_k else 0.0

        shared = keys_a & keys_b
        d_hyp = 0.0
        if shared:
            a_map = {self._term_key(t): t.get('params', {}) for t in genome_a}
            b_map = {self._term_key(t): t.get('params', {}) for t in genome_b}
            diffs = []
            for k in shared:
                pa, pb = a_map[k], b_map[k]
                for p in set(pa) & set(pb):
                    va, vb = pa[p], pb[p]
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        denom = max(abs(va), abs(vb), 1e-9)
                        diffs.append(min(1.0, abs(va - vb) / denom))
            d_hyp = sum(diffs) / len(diffs) if diffs else 0.0

        w1, w2, w3 = self.ted_weights
        return (w1 * d_feat + w2 * d_top + w3 * d_hyp) / (w1 + w2 + w3)
#: </knowledge_graph_ted>

#: <knowledge_graph_archive>
    def try_add_champion(self, genome: List[Dict[str, Any]], score: float, island_name: str) -> bool:
        """
        Quality-Diversity admission into the Champion Archive.

        A candidate enters only if it is structurally distinct (min TED >= D_min) from every
        existing champion, OR if it occupies the same topological niche but has a strictly
        lower penalized score. Keeps the archive both elite and diverse.
        """
        if not genome or score is None or not np.isfinite(score):
            return False

        entry = {
            'genome': [{'type': t['type'], 'feature': t['feature'], 'params': dict(t.get('params', {}))}
                       for t in genome],
            'score': float(score),
            'island': island_name,
        }

        if not self.champion_archive:
            self.champion_archive.append(entry)
            return True

        dists = [(self.topological_edit_distance(genome, c['genome']), i)
                 for i, c in enumerate(self.champion_archive)]
        min_d, min_i = min(dists, key=lambda x: x[0])

        if min_d >= self.niche_radius:
            self.champion_archive.append(entry)
        elif entry['score'] < self.champion_archive[min_i]['score']:
            self.champion_archive[min_i] = entry        # same niche, strictly better -> overwrite
        else:
            return False

        if len(self.champion_archive) > self.archive_max_size:
            self.champion_archive.sort(key=lambda c: c['score'])
            self.champion_archive = self.champion_archive[:self.archive_max_size]
        return True

    def get_island_champion(self, island_name: str) -> Optional[List[Dict[str, Any]]]:
        """Returns the best-scoring archived genome spawned by the given Island, if any."""
        cands = [c for c in self.champion_archive if c['island'] == island_name]
        if not cands:
            return None
        return min(cands, key=lambda c: c['score'])['genome']

    def get_champions(self) -> List[Dict[str, Any]]:
        """Returns the current Champion Archive sorted best-first."""
        return sorted(self.champion_archive, key=lambda c: c['score'])
#: </knowledge_graph_archive>
#: </knowledge_graph_class>