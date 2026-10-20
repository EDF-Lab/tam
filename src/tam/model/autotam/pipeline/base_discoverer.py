# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Base Discoverer for Automated TAM (AutoTAM).

Executes the evolutionary search process and extracts the top performing 
formulas per mathematical island.
"""

#: <base_discoverer_imports>
import pandas as pd
import re
from typing import Dict, Any, List, Optional, Tuple
from .context import PipelineContext
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.population_nodes import get_island_objects
from tam.model.autotam.guards import guard_violation, monotone_features
from tam.model.autotam.seeding import rank_key
from tam.model.autotam.search_preset import FULL_GENERATIONS
import logging
logger = logging.getLogger(__name__)
#: </base_discoverer_imports>

#: <base_discoverer_class>
class BaseDiscoverer:
    """Executes the evolutionary search and extracts the top formulas per Island."""
    
#: <base_discoverer_init>
    def __init__(self, pop_size: int = 64, eta: float = 0.1, mandatory_terms: Optional[List[str]] = None, mandatory_variables: Optional[List[str]] = None,
                 islands: Optional[List[str]] = None, n_generations: int = FULL_GENERATIONS, extended: bool = False,
                 crossover: bool = False, bounded_rewards: bool = False):
        self.islands = list(islands) if islands else None  # names of the islands searched, None for all
        self.n_generations = n_generations
        self.extended = bool(extended)
        self.crossover = bool(crossover)
        self.bounded_rewards = bool(bounded_rewards)
        self.pop_size = pop_size
        self.eta = eta
        self.mandatory_terms = mandatory_terms or []
        self.mandatory_variables = mandatory_variables or []
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
            't': 'TreeIsland', 'lt': 'TreeIsland'
        }
        if "~" not in formula: 
            return "Continent"
        
        rhs = formula.split("~")[1]
        
        effect_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(")
        effects = set(effect_pattern.findall(rhs))
        effects.discard("1")
        
        terms = [t for t in rhs.split("+") if t.strip() and t.strip() != "1"]
        num_terms = len(terms)
        
        # te and pid are open to every Island, so a formula is attributed by its bases,
        # including the ones inside a tensor product.
        core_effects = effects - {'l', 'c', 'pid', 'te'}
        
        if not core_effects: 
            return "LinearIsland"
            
        if len(core_effects) > 1:
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
        logger.info("BaseDiscoverer: Starting Evolutionary Search...")
        
        draga = DragTAM(target_col=ctx.target, population_size=self.pop_size, n_generations=self.n_generations,
                        group_col=ctx.group_col, date_col=ctx.date_col,
                        mandatory_terms=getattr(ctx, 'mandatory_terms', None) or self.mandatory_terms,
                        mandatory_variables=getattr(ctx, 'mandatory_variables', None) or self.mandatory_variables,
                        bounded_rewards=self.bounded_rewards)
        islands = [i for i in get_island_objects() if self.islands is None or type(i).__name__ in self.islands]
        for island in islands:
            island.extended_mutations = self.extended
            island.crossover = self.crossover

        draga.optimize(
            cv_folds=ctx.cv_folds,
            islands=islands,
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

        if evaluated and getattr(ctx, "extrapolation_guards", False):
            monotone = monotone_features(ctx.df_fit, [c for c in ctx.df_fit.columns if c != ctx.target])
            n_rows = len(ctx.df_fit)
            kept = []
            for island, form, rmse in evaluated:
                reason = guard_violation(form, n_rows, monotone, ctx.estimate_complexity)
                if reason is None:
                    kept.append((island, form, rmse))
                else:
                    code = reason.split(":")[0]
                    ctx.guard_rejections[code] = ctx.guard_rejections.get(code, 0) + 1
            evaluated = kept

        if evaluated:
            df_eval = pd.DataFrame(evaluated, columns=["Island", "Formula", "RMSE"])
            for island, group in df_eval.groupby("Island"):
                unique = group.drop_duplicates(subset=["Formula"])
                order = sorted(range(len(unique)), key=lambda i: rank_key(float(unique["RMSE"].iloc[i]), str(unique["Formula"].iloc[i])))
                top_5 = unique.iloc[order].head(5)
                island_champions[island] = top_5["Formula"].tolist()
        else:
            logger.warning("No formulas survived the evolutionary evaluation. Check for hidden NaNs or Inf values.")

        logger.info(f"BaseDiscoverer: Extracted {sum(len(v) for v in island_champions.values())} Top Formulas across {len(island_champions)} Islands.")
        return island_champions, draga
#: </base_discoverer_search>
#: </base_discoverer_class>