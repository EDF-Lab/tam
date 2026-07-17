# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Evolutionary Reporter & Diagnostics for TAM (AutoTAM).

This module handles all transparency, logging, and CSV artifact generation. 
It cleanly decouples the diagnostic tracking from the mathematical primal solvers,
providing total visibility into the Estimation of Distribution Algorithm (EDA), 
Successive Halving search, and sequential ensemble aggregation.

Exports:
    1. AutoTAM_history_*.csv: The Successive Halving evolutionary trail.
    2. AutoTAM_chronological_tests_*.csv: Every single model type tested across the pipeline.
    3. AutoTAM_feature_statistics_*.csv: Deep dive into feature survival and variance.
    4. AutoTAM_island_statistics_*.csv: Generative island activation frequencies.
    5. AutoTAM_final_architectures_*.csv: The exact configurations of the final ensemble.
    6. AutoTAM_collinearity_purge_log_*.csv: Tracks redundant features deleted to ensure matrix stability.
    7. AutoTAM_opera_weights_*.csv: Sequential trajectory of ensemble weights over time.
    8. AutoTAM_pdp_variance_*.csv: Global feature importance mapped via Partial Dependence.
"""

#: <evolution_reporter_imports>
import pandas as pd
import numpy as np
import datetime
import os
import re
from typing import Dict, List, Any
#: </evolution_reporter_imports>

#: <evolution_reporter_class>
class EvolutionReporter:
    """
    Dedicated diagnostic and summary logger for the AutoTAM framework.
    
    Acts as a telemetry module, securely dumping the internal states of the 
    Knowledge Graph, the DragTAM evolutionary engine, and the OPERA Minimax 
    aggregator into easily parseable CSVs for downstream MLOps tracking.
    """
    
#: <reporter_init>
    def __init__(self, export_dir: str = "AutoTAM_exports"):
        """
        Initializes the reporter and ensures the local export directory exists.
        
        Args:
            export_dir (str): Directory path where diagnostic CSVs will be saved.
        """
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#: </reporter_init>

#: <reporter_export_chronological>
    def export_chronological_tests(self, test_registry: List[Dict[str, Any]]) -> None:
        """
        Exports a chronological log of all tests made across the entire pipeline.
        This provides a transparent audit trail of evaluated models 
        (e.g., StaticTAM_priors, StaticTAM_gcv, KalmanTAM, AdaptiveTAM, OperaTAM).
        """
        if not test_registry:
            return
            
        filepath = os.path.join(self.export_dir, f"AutoTAM_chronological_tests_{self.run_timestamp}.csv")
        df = pd.DataFrame(test_registry)
        
        cols = ['Timestamp', 'Phase', 'Model_Type', 'Validation_RMSE', 'Formula', 'Hyperparameters']
        df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
        
        df.to_csv(filepath, index=False)
        print(f"Chronological test log exported to: {filepath}")
#: </reporter_export_chronological>

#: <reporter_export_feature_stats>
    def export_feature_statistics(self, draga_engine: Any) -> None:
        """
        Aggregates the Knowledge Graph into a feature-centric report.
        Shows which features dominated the evolutionary search and what 
        mathematical topologies (splines, wavelets, etc.) yielded the most variance.
        """
        if not hasattr(draga_engine, 'kg') or not hasattr(draga_engine.kg, 'graph'):
            return
            
        feature_stats = []
        for feature, effects in draga_engine.kg.graph.items():
            total_success = 0
            best_effect = None
            highest_variance = 0.0
            
            for effect, stats in effects.items():
                successes = len(stats["importance"])
                total_success += successes
                mean_var = np.mean(stats["importance"]) if stats["importance"] else 0.0
                
                if mean_var > highest_variance:
                    highest_variance = mean_var
                    best_effect = effect
                    
            feature_stats.append({
                "Feature": feature,
                "Total_Activations": total_success,
                "Highest_Variance_Explained": highest_variance,
                "Preferred_Mathematical_Topology": best_effect
            })
            
        if feature_stats:
            filepath = os.path.join(self.export_dir, f"AutoTAM_feature_statistics_{self.run_timestamp}.csv")
            df = pd.DataFrame(feature_stats).sort_values(by="Total_Activations", ascending=False)
            df.to_csv(filepath, index=False)
            print(f"Feature statistics exported to: {filepath}")
#: </reporter_export_feature_stats>

#: <reporter_export_island_stats>
    def export_island_statistics(self, draga_engine: Any) -> None:
        """
        Parses the evolutionary history to compute statistics on Island usage.
        Maps the raw formula string effects back to their theoretical 
        Generator Islands to show which mathematical families were most useful.
        """
        if not hasattr(draga_engine, 'history') or not draga_engine.history:
            return

        island_map = {
            'l': 'LinearIsland', 'c': 'LinearIsland (Categorical)',
            's': 'SplineIsland', 'f': 'FourierIsland', 'p': 'ChebyshevIsland',
            'w': 'WaveletIsland', 'n': 'NeuralIsland', 'rbf': 'RBFIsland',
            't': 'TreeIsland', 'te': 'CrossIsland (Tensor)'
        }
        
        effect_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(\s*([^,)]+)(.*?)\)")
        island_counts = {island: 0 for island in set(island_map.values())}
        
        for record in draga_engine.history:
            formula = record.get("Formula", "")
            if "~" not in formula: continue
            
            rhs_str = formula.split('~')[1]
            for part in rhs_str.split('+'):
                part = part.strip()
                if not part or part == '1': continue
                
                match = effect_pattern.match(part)
                if match:
                    eff = match.group(1).strip()
                    island_name = island_map.get(eff, f'UnknownIsland_{eff}')
                    island_counts[island_name] = island_counts.get(island_name, 0) + 1

        filepath = os.path.join(self.export_dir, f"AutoTAM_island_statistics_{self.run_timestamp}.csv")
        df = pd.DataFrame(list(island_counts.items()), columns=['Island_Name', 'Total_Proposals'])
        df = df[df['Total_Proposals'] > 0].sort_values(by='Total_Proposals', ascending=False)
        df.to_csv(filepath, index=False)
        print(f"Island statistics exported to: {filepath}")
#: </reporter_export_island_stats>

#: <reporter_export_advanced_mlops>
    def export_collinearity_purge_log(self, purge_log: List[Dict[str, Any]]) -> None:
        """
        Logs engineered features deleted by the FeatureEngineer to track redundancy.
        """
        if not purge_log: 
            return
        filepath = os.path.join(self.export_dir, f"AutoTAM_collinearity_purge_log_{self.run_timestamp}.csv")
        pd.DataFrame(purge_log).to_csv(filepath, index=False)
        print(f"Collinearity purge log exported to: {filepath}")

    def export_opera_weights_trajectory(self, weights_df: pd.DataFrame, league_name: str) -> None:
        """
        Exports the sequential online learning weights returned by the OPERA algorithm.
        """
        if weights_df.empty: 
            return
        filepath = os.path.join(self.export_dir, f"AutoTAM_opera_weights_{league_name}_{self.run_timestamp}.csv")
        weights_df.to_csv(filepath, index=False)
        print(f"OPERA weights trajectory ({league_name}) exported to: {filepath}")

    def export_pdp_variance(self, pdp_df: pd.DataFrame, model_name: str) -> None:
        """
        Exports the Partial Dependence values (variance explained) computed by decomposing 
        the predictions of the base Generalized Additive Model.
        """
        if pdp_df.empty: 
            return
        filepath = os.path.join(self.export_dir, f"AutoTAM_pdp_variance_{model_name}_{self.run_timestamp}.csv")
        pdp_df.to_csv(filepath, index=False)
        print(f"Partial Dependence (PDP) variance exported to: {filepath}")
#: </reporter_export_advanced_mlops>

#: <reporter_export_diagnostics>
    def export_evolutionary_diagnostics(self, draga_engine: Any) -> None:
        """
        Exports the evolutionary history and triggers generation of sub-reports.
        """
        if hasattr(draga_engine, 'history') and draga_engine.history:
            filepath = os.path.join(self.export_dir, f"AutoTAM_history_{self.run_timestamp}.csv")
            pd.DataFrame(draga_engine.history).to_csv(filepath, index=False)
            print(f"Evolutionary trail exported to: {filepath}")
            
        self.export_feature_statistics(draga_engine)
        self.export_island_statistics(draga_engine)
#: </reporter_export_diagnostics>

#: <reporter_export_architectures>
    def export_final_architectures(self, trained_experts: List[Dict[str, Any]]) -> None:
        """
        Saves the final, optimized formulas and explicit hyperparameters.
        Flattens nested state-space wrappers for clarity.
        """
        csv_data = []
        for exp in trained_experts:
            m_type = exp.get("type", "Unknown")
            
            if m_type == "static":
                final_formula = getattr(exp["model"], "formula_", "Unknown")
            elif m_type in ["kalman", "kalman_league_member", "adaptive"]:
                base = getattr(exp["model"], "base_model_", getattr(exp["model"], "_saved_base_model", None))
                final_formula = getattr(base, "formula_", "Unknown") if base else "Unknown"
            else:
                final_formula = "Unknown"
                
            csv_data.append({
                "Model_Name": exp.get("name", "Unknown"),
                "Type": m_type,
                "Validation_RMSE": exp.get("val_rmse", np.nan),
                "Optimized_Formula": final_formula,
                "Dynamic_Params": str(exp.get("params", ""))
            })
            
        filepath = os.path.join(self.export_dir, f"AutoTAM_final_architectures_{self.run_timestamp}.csv")
        pd.DataFrame(csv_data).to_csv(filepath, index=False)
        print(f"Final architecture CSV exported to: {filepath}")
#: </reporter_export_architectures>

#: <reporter_print_summary>
    def print_summary(self, trained_experts: List[Dict[str, Any]], weights_top10: Dict[str, float], weights_all: Dict[str, float]) -> None:
        """
        Prints the console summary report detailing the selected experts.
        """
        print("\n--- AutoTAM: Final Architecture Report ---")
        print("(Check AutoTAM_exports folder for full details)")
        selected_experts = set()
        
        if weights_top10:
            print("\n--- OPERA TOP 10 ENSEMBLE WEIGHTS ---")
            for exp_name, weight in sorted(weights_top10.items(), key=lambda x: x[1], reverse=True): 
                print(f"  * {exp_name}: {weight * 100:.2f}%")
                selected_experts.add(exp_name)
                
        if weights_all:
            print("\n--- OPERA ALL MODELS ENSEMBLE WEIGHTS ---")
            for exp_name, weight in sorted(weights_all.items(), key=lambda x: x[1], reverse=True): 
                print(f"  * {exp_name}: {weight * 100:.2f}%")
                selected_experts.add(exp_name)
                
        print("\n--- EXACT FORMULAS OF ALL SELECTED MODELS ---")
        for expert_dict in trained_experts:
            exp_name = expert_dict.get('name', 'Unknown')
            
            if exp_name in selected_experts:
                print(f"\n[Selected] {exp_name}")
                if expert_dict["type"] == "static":
                    form = getattr(expert_dict['model'], 'formula_', 'Unknown')
                    print(f"  Type: STATIC | Formula: {form}")
                elif expert_dict["type"] in ["kalman", "kalman_league_member"]:
                    base = getattr(expert_dict['model'], 'base_model_', None)
                    form = getattr(base, 'formula_', 'Unknown') if base else 'Unknown'
                    print(f"  Type: KALMAN | Params: {expert_dict.get('params', {})} \n  Base Formula: {form}")
                elif expert_dict["type"] == "adaptive":
                    base = getattr(expert_dict['model'], '_saved_base_model', None)
                    form = getattr(base, 'formula_', 'Unknown') if base else 'Unknown'
                    print(f"  Type: ADAPTIVE (ECM) | Params: {expert_dict.get('params', {})} \n  Base Formula: {form}")
#: </reporter_print_summary>
#: </evolution_reporter_class>