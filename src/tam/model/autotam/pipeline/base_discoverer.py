# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Base Discoverer for Automated TAM (AutoTAM).

Executes the evolutionary search process and extracts the top performing 
formulas per mathematical island.
"""

#: <base_discoverer_imports>
import pandas as pd
import re
from typing import Dict, Any, List, Tuple
from .context import PipelineContext
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.population_nodes import get_island_generators
#: </base_discoverer_imports>

#: <base_discoverer_class>
class BaseDiscoverer:
    """Executes the evolutionary search and extracts the top formulas per Island."""
    
#: <base_discoverer_init>
    def __init__(self, pop_size: int = 64, eta: float = 0.1):
        self.pop_size = pop_size
        self.eta = eta
#: </base_discoverer_init>

#: <base_discoverer_get_island>
    def _get_island_from_formula(self, formula: str) -> str:
        """
        Classifies a raw formula string into its originating mathematical Island 
        based on topological signatures and term counts.
        """
        island_map = {
            'l': 'LinearIsland', 'c': 'LinearIsland',
            's': 'SplineIsland', 'f': 'FourierIsland', 'p': 'ChebyshevIsland',
            'w': 'WaveletIsland', 'n': 'NeuralIsland', 'rbf': 'RBFIsland',
            't': 'TreeIsland', 'te': 'CrossIsland'
        }
        if "~" not in formula: 
            return "Continent"
        
        rhs = formula.split("~")[1]
        
        effect_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(")
        effects = set(effect_pattern.findall(rhs))
        effects.discard("1")
        
        terms = [t for t in rhs.split("+") if t.strip() and t.strip() != "1"]
        num_terms = len(terms)
        
        core_effects = effects - {'l', 'c'}
        
        if not core_effects: 
            return "LinearIsland"
            
        if len(core_effects) > 1:
            if 'te' in core_effects and num_terms <= 4: 
                return "CrossIsland"
            if num_terms <= 4:
                return "SmallContinent"
            return "Continent"
        
        eff = list(core_effects)[0]
        return island_map.get(eff, "Continent")
#: </base_discoverer_get_island>

#: <base_discoverer_search>
    def search(self, ctx: PipelineContext) -> Tuple[Dict[str, List[str]], Any]:
        """
        Runs the DragTAM optimizer and categorizes the best performing models.
        """
        print("BaseDiscoverer: Starting Evolutionary Search...")
        
        draga = DragTAM(target_col=ctx.target, population_size=self.pop_size)
        island_generators = get_island_generators()
        
        draga.optimize(
            cv_folds=ctx.cv_folds, 
            island_generators=island_generators, 
            search_space=ctx.search_space
        )
        
        island_champions = {}
        evaluated = []
        
        cache = None
        for attr in ['evaluation_cache', 'cache', 'history', '_cache', 'evaluated_models']:
            if hasattr(draga, attr):
                val = getattr(draga, attr)
                if isinstance(val, dict) and len(val) > 0:
                    cache = val
                    break
                    
        if cache is not None:
            for form, metrics in cache.items():
                island = self._get_island_from_formula(form)
                
                if isinstance(metrics, (tuple, list)): 
                    rmse = metrics[0] 
                elif isinstance(metrics, dict): 
                    rmse = metrics.get('rmse', metrics.get('RMSE', float('inf')))
                else:
                    try: 
                        rmse = float(metrics)
                    except ValueError: 
                        rmse = float('inf')
                    
                if pd.notnull(rmse) and rmse != float('inf'):
                    evaluated.append((island, form, rmse))
                    
        elif hasattr(draga, 'history_df') and isinstance(draga.history_df, pd.DataFrame):
            df_hist = draga.history_df
            if 'Formula' in df_hist.columns and 'RMSE' in df_hist.columns:
                for _, row in df_hist.iterrows():
                    if pd.notnull(row['RMSE']) and row['RMSE'] != float('inf'):
                        island = self._get_island_from_formula(row['Formula'])
                        evaluated.append((island, row['Formula'], row['RMSE']))

        if evaluated:
            df_eval = pd.DataFrame(evaluated, columns=["Island", "Formula", "RMSE"])
            for island, group in df_eval.groupby("Island"):
                top_5 = group.drop_duplicates(subset=["Formula"]).sort_values("RMSE").head(5)
                island_champions[island] = top_5["Formula"].tolist()
        else:
            print("Warning: No formulas survived the evolutionary evaluation. Check for hidden NaNs or Inf values.")

        print(f"BaseDiscoverer: Extracted {sum(len(v) for v in island_champions.values())} Top Formulas across {len(island_champions)} Islands.")
        return island_champions, draga
#: </base_discoverer_search>
#: </base_discoverer_class>