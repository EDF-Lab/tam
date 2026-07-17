# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

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
        k_grid = sorted(list(set([max(3, max_k - 4), max(3, max_k - 2), max_k])))
        
        max_trees = min(50, max(10, dataset_size // 20))
        t_grid = sorted(list(set([10, max(10, max_trees // 2), max_trees])))
        
        max_centers = min(26, max(5, dataset_size // 15))
        rbf_grid = sorted(list(set([max(5, max_centers - 5), max(5, max_centers - 2), max_centers])))
        
        max_neurons = min(100, max(5, dataset_size // 10))
        n_grid = sorted(list(set([2, max(2, max_neurons // 4), max(5, max_neurons // 2), max_neurons])))

        search_space = {}        
        
        for col in features:
            topology = self._analyze_topology(df[col])
            
            feature_space = {
                "topology": topology,
                "eligible_effects": ["l"], 
                "grids": {
                    "l": {"ap": [base_ap - 5.0]}, 
                    "f": {"m": [4, 5, 6], "s": [1, 2], "ap": ap_grid},
                    "p": {"deg": [6, 8, 10], "s": [1, 2], "ap": ap_grid},
                    "s": {"k": k_grid, "deg": [3], "p": [1, 2], "ap": ap_grid},
                    "w": {"n_scales": [3, 4], "n_locations": [10, 12, 14], "ap": ap_grid},
                    "t": {"n_trees": t_grid, "max_depth": [1, 3], "ap": ap_grid},
                    "n": {"n_neurons": n_grid, "n_hidden_layers": [1], "act": ["relu"], "ap": strong_ap_grid},
                    "rbf": {"n_centers": rbf_grid, "ap": strong_ap_grid}
                },
                "max_active_effects": self.max_active_effects 
            }

            if topology == "discrete":
                feature_space["eligible_effects"].extend(["c", "t"])
                feature_space["grids"]["c"] = {"n_cat": [df[col].nunique()], "topo": ["fourier"], "ap": [base_ap - 8.0]}
                feature_space["grids"]["t"] = {"n_trees": t_grid, "max_depth": [1, 3, 6]}
                
            elif topology == "sparse":
                feature_space["eligible_effects"].extend(["t", "rbf", "n"])
                feature_space["grids"]["t"] = {"n_trees": t_grid, "max_depth": [1, 3, 6]}
                feature_space["grids"]["rbf"] = {"n_centers": rbf_grid}
                feature_space["grids"]["n"] = {"n_neurons": n_grid, "act": ["relu", "tanh", "cos"]}
                
            elif topology == "continuous":
                feature_space["eligible_effects"].extend(["s", "p", "w", "n", "rbf", "t", "f"])
                feature_space["grids"]["f"] = {"m": [4, 5, 6, 7, 8], "s": [1, 2, 3]}
                feature_space["grids"]["s"] = {"k": k_grid, "deg": [3], "p": [1, 2, 3]}
                feature_space["grids"]["p"] = {"deg": [8, 10, 12, 14, 16], "s": [1, 2, 3]}
                feature_space["grids"]["w"] = {"n_scales": [3, 4, 5], "n_locations": [10, 12, 14, 16, 18]}
                feature_space["grids"]["n"] = {"n_neurons": n_grid, "act": ["relu", "tanh", "cos"]}
                feature_space["grids"]["rbf"] = {"n_centers": rbf_grid}
                feature_space["grids"]["t"] = {"n_trees": t_grid, "max_depth": [1, 3, 6]}
                    
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