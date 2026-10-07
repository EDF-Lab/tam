# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Effect Selector for Automated TAM (AutoTAM).

This module bridges the data engineering pipeline and the mathematical primal solver.
It guarantees strict isolation against Target Leakage by ensuring only user-defined 
features and mathematically safe temporal lags enter the evolutionary search space.

Crucially, it implements the "Strict Covariate Lock", a theoretical upgrade that limits 
the number of concurrent mathematical representations (effects) a single feature can 
have in a formula. This prevents formula bloat, maintains Partial Dependence Plot (PDP) 
interpretability, and avoids design matrix singularities during Conjugate Gradient descent.

"""

#: <effect_selector_imports>
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List
from .feature_profiler import FeatureProfiler
#: </effect_selector_imports>

#: <effect_selector_class>
class EffectSelector:
    """
    Analyzes feature topologies to build a safe, mathematically restricted 
    Search Space for the Evolutionary Pipeline Search (DragTAM).
    
    This ensures the evolutionary engine does not waste computational resources evaluating 
    mathematically invalid topologies (e.g., applying continuous Fourier series to discrete 
    categorical data or highly sparse distributions).
    """
    
#: <effect_selector_init>
    def __init__(self, categorical_threshold: int = 15, sparsity_threshold: float = 0.80, max_active_effects: int = 2):
        """
        Initializes the EffectSelector with strict empirical and structural thresholds.

        Args:
            categorical_threshold (int): If a feature contains this many or fewer unique 
                values, it is mathematically classified as discrete.
            sparsity_threshold (float): If a continuous feature exhibits a zero-ratio greater 
                than or equal to this threshold, it is classified as highly sparse.
            max_active_effects (int): Strict Covariate Lock. The maximum number of different 
                mathematical term types (e.g., Spline, Tree, Fourier) a single feature 
                can simultaneously hold in a generated formula. Prevents feature deduplication bloat.
        """
        self.cat_threshold = categorical_threshold
        self.sparsity_threshold = sparsity_threshold
        self.max_active_effects = max_active_effects
        self.profiler = FeatureProfiler()
#: </effect_selector_init>

#: <build_search_space>
    def build_search_space(self, df: pd.DataFrame, config: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs the valid parameter grid by evaluating uniqueness and sparsity thresholds.
        Dynamically scales mathematical bounds (knots, trees, penalties) to the dataset's geometry.

        Args:
            df (pd.DataFrame): The preprocessed dataset.
            config (Dict[str, Any]): The parsed formula configuration dictating allowed features.
            metadata (Dict[str, Any]): Profiler metadata containing 'date_col' and 'group_col'.

        Returns:
            Dict[str, Any]: A nested dictionary mapping each viable feature to its allowable 
                functional effects, hyperparameter grids, and covariate lock constraints.
        """
        targets = config.get("targets", [])
        explicit_features = config.get("features", [])
        date_col = metadata.get("date_col", None)
        group_col = metadata.get("group_col", None)
        
        lower_targets = [t.lower() for t in targets]
        
        if explicit_features:
            allowed = set(explicit_features)
            for f in explicit_features:
                allowed.update([c for c in df.columns if c.startswith(f"{f}_")])
            for t in targets:
                allowed.update([c for c in df.columns if c.startswith(f"{t}_lag_")])
                
            features = [c for c in df.columns if c in allowed and c.lower() not in lower_targets and c not in [date_col, group_col]]
        else:
            features = [c for c in df.columns if c.lower() not in lower_targets and c not in [date_col, group_col]]
        
        # Calculate dynamic geometric constraints to prevent solver crashes
        dataset_size = len(df)
        target_col = targets[0] if targets else None
        
        if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            target_var = float(df[target_col].var())
        else:
            target_var = 1.0

        base_ap = round(float(np.log10(max(1e-5, target_var))), 1)
        ap_grid = [base_ap - 2.0, base_ap, base_ap + 2.0]
        strong_ap_grid = [base_ap - 4.0, base_ap - 2.0, base_ap]

        max_k = min(20, max(3, dataset_size // 10))
        # anchor on the factory default (k=10) plus data-scaled neighbours
        k_grid = sorted(list(set([min(10, max_k), max(3, max_k - 4), max(3, max_k - 2), max_k])))
        
        max_trees = min(50, max(10, dataset_size // 20))
        t_grid = sorted(list(set([10, max(10, max_trees // 2), max_trees])))
        
        max_centers = min(26, max(5, dataset_size // 15))
        rbf_grid = sorted(list(set([max(5, max_centers - 5), max(5, max_centers - 2), max_centers])))
        
        max_neurons = min(100, max(5, dataset_size // 10))
        n_grid = sorted(list(set([2, max(2, max_neurons // 4), max(5, max_neurons // 2), max_neurons])))

        # Profile once to lock feature capacity, preventing combinatorial explosion during structural search.
        profiles = self.profiler.profile(df, target_col, features, group_col)

        # PID terms are offered on the most recent autoregressive lag only (the smallest lag order).
        most_recent_lag = FeatureProfiler.most_recent_lag(features)
        search_space = {}

        for col in features:
            topology = self._analyze_topology(df[col])

            # The profiler only speaks for continuous features; discrete/sparse topologies are
            # already routed to categorical/tree bases, where a basis-capacity verdict is moot.
            prof = profiles.get(col) if topology == "continuous" else None
            profiled = prof is not None and not prof["linear"]
            prof_m = int(prof["m"]) if profiled and prof.get("m") else 6
            # Bound capacity by sample size, ignoring legacy cap.
            cap_k = max(10, min(50, dataset_size // 100))
            prof_k = int(np.clip(int(prof["k"]) if profiled and prof.get("k") else min(10, max_k),
                                 3, cap_k))

            def _ap(effect: str) -> List[float]:
                return [FeatureProfiler.penalty_for(effect, col, prof_m, prof_k)]

            feature_space = {
                "topology": topology,
                "profile": prof,
                "eligible_effects": ["l"],
                # Distribute identical capacity across all families to ensure OPERA diversity.
                "grids": {
                    "l": {"ap": _ap("l")},
                    "f": {"m": [prof_m], "s": [2], "ap": _ap("f")},
                    "p": {"deg": [int(np.clip(prof_m, 5, 20))], "s": [2], "ap": _ap("p")},
                    "s": {"k": [prof_k], "deg": [3], "p": [2], "ap": _ap("s")},
                    "w": {"n_scales": [4], "n_locations": [int(np.clip(prof_k // 3, 10, 16))],
                          "ap": _ap("w")},
                    # A tree is an oblivious binary tree (max_depth) or a flat histogram
                    # (max_leaves); _get_safe_params keeps one of the two per term.
                    "t": {"n_trees": [10, 25], "max_depth": [1, 2], "max_leaves": [50],
                          "split_strategy": ["quantile", "uniform"], "sp_alpha": [0.0, 1.0]},
                    "lt": {"max_depth": [3, 5], "max_leaves": [50], "split_strategy": ["quantile", "uniform"],
                           "ap": _ap("lt")},
                    # Exclude 'cos' activation: extrapolates poorly across regime shifts.
                    "n": {"n_neurons": [100], "n_hidden_layers": [1], "act": ["relu", "tanh"]},
                    "rbf": {"n_centers": [int(np.clip(prof_k, 10, 50))], "ap": _ap("rbf")},
                    "pid": {"w": [3, 7], "d_pen": [10.0], "ap": _ap("pid")}
                },
                "max_active_effects": self.max_active_effects
            }

            if prof is not None and prof["linear"]:
                # Hard lock: Prevent non-linear solvers from fitting noise on purely linear features.
                if col == most_recent_lag:
                    feature_space["eligible_effects"].append("pid")
                search_space[col] = feature_space
                continue

            # Topology only gates which effects are ELIGIBLE.
            if topology == "discrete":
                feature_space["eligible_effects"].extend(["c", "t"])
                feature_space["grids"]["c"] = {"n_cat": [df[col].nunique()], "topo": ["nominal"], "p_order": [1]}
            elif topology == "sparse":
                feature_space["eligible_effects"].extend(["t", "rbf", "n"])
            elif topology == "continuous":
                feature_space["eligible_effects"].extend(["s", "p", "w", "n", "rbf", "t", "f", "lt"])

            search_space[col] = feature_space

        return search_space
#: </build_search_space>

#: <analyze_topology>
    def _analyze_topology(self, series: pd.Series) -> str:
        """
        Mathematically categorizes a 1D vector into its inherent data topology.

        Args:
            series (pd.Series): The feature vector to analyze.

        Returns:
            str: The detected topology ('discrete', 'sparse', or 'continuous').
        """
        if not pd.api.types.is_numeric_dtype(series):
            return "discrete"
        
        if "_rolling_" in str(series.name).lower() or "_ewma_" in str(series.name).lower():
            return "continuous"
        
        if series.nunique() <= self.cat_threshold:
            return "discrete"
            
        if (series == 0).mean() >= self.sparsity_threshold: 
            return "sparse"
            
        return "continuous"
#: </analyze_topology>

#: <covariate_lock_validator>
    def validate_covariate_lock(self, genome: List[str]) -> bool:
        """
        Helper validation method for the Strict Covariate Lock.
        Can be called downstream by the genetic engine to verify a generated 
        formula does not violate the maximum active effects threshold.

        Args:
            genome (List[str]): The list of additive terms representing a model formula.

        Returns:
            bool: True if the genome complies with the Covariate Lock, False otherwise.
        """
        feature_counts = {}
        
        for term in genome:
            match = re.search(r'\A[a-z]{1,3}\s*\(\s*([A-Za-z0-9_\.]+)', term.strip())
            if match:
                feat = match.group(1).strip()
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
                
                if feature_counts[feat] > self.max_active_effects:
                    return False
        return True
#: </covariate_lock_validator>

#: </effect_selector_class>