# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Evolutionary Islands (Population Nodes) for AutoTAM.

Defines specialized generative islands that propose parsimonious formula candidates 
by querying the Bayesian Knowledge Graph. Each island restricts its search space 
to a specific family of mathematical effects (e.g., spectral, tree-based, neural).

This distributed "Island Model" approach ensures that highly diverse mathematical 
topologies are explored simultaneously during the evolutionary search, preventing 
premature convergence on a single family of functions.
"""

#: <population_nodes_imports>
import random
from typing import List, Dict, Any, Callable
from .knowledge_graph import KnowledgeGraph
#: </population_nodes_imports>

#: <population_nodes_base>
class BaseIsland:
    """
    Base configuration for evolutionary formula generation islands.
    
    Every island enforces a strict limit on the number of terms it can generate 
    (`max_terms`) to encourage parsimonious models and prevent formula bloat.
    """
    def __init__(self, name: str, valid_effects: List[str], max_terms: int = 10):
        self.name = name
        self.valid_effects = valid_effects
        self.max_terms = max_terms

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        """
        Generates a formula string based on the island's constraints and the Knowledge Graph.
        
        Args:
            kg (KnowledgeGraph): The Bayesian tracker providing probabilistic weights.
            available_features (List[str]): Features allowed by the covariate lock.
            search_space (Dict[str, Any]): The restricted dictionary of safe parameters.
            complexity_cap (bool): If True, strictly forces the minimum capacity hyperparameter 
                                   configurations from the search space to enforce parsimony.
            
        Returns:
            str: A valid right-hand side (RHS) GAM formula string.
        """
        raise NotImplementedError
#: </population_nodes_base>

#: <standard_islands>
class LinearIsland(BaseIsland):
    """Proposes strictly linear ('l') and categorical ('c') terms for baseline effects."""
    def __init__(self, max_terms: int = 10):
        super().__init__("LinearIsland", ['l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class SplineIsland(BaseIsland):
    """Proposes smooth, localized non-linearities via penalized splines ('s')."""
    def __init__(self, max_terms: int = 10):
        super().__init__("SplineIsland", ['s', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class FourierIsland(BaseIsland):
    """Proposes global periodic bases ('f') ideal for capturing strict seasonalities."""
    def __init__(self, max_terms: int = 10):
        super().__init__("FourierIsland", ['f', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class ChebyshevIsland(BaseIsland):
    """Proposes orthogonal polynomial expansions ('p') for continuous trends."""
    def __init__(self, max_terms: int = 10):
        super().__init__("ChebyshevIsland", ['p', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class WaveletIsland(BaseIsland):
    """Proposes highly localized wavelet bases ('w') for sharp structural breaks or spikes."""
    def __init__(self, max_terms: int = 10):
        super().__init__("WaveletIsland", ['w', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)
#: </standard_islands>

#: <interaction_islands>
class NeuralIsland(BaseIsland):
    """Proposes shallow neural network components ('n') capable of dense feature interactions."""
    def __init__(self, max_terms: int = 5):
        super().__init__("NeuralIsland", ['n', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect='n', complexity_cap=complexity_cap)


class RBFIsland(BaseIsland):
    """Proposes Radial Basis Functions ('rbf') for distance-based spatial/temporal modeling."""
    def __init__(self, max_terms: int = 5):
        super().__init__("RBFIsland", ['rbf', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect='rbf', complexity_cap=complexity_cap)


class TreeIsland(BaseIsland):
    """Proposes Gradient Boosted Tree components ('t') for jagged, high-frequency signals."""
    def __init__(self, max_terms: int = 5):
        super().__init__("TreeIsland", ['t', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect='t', complexity_cap=complexity_cap)
#: </interaction_islands>

#: <cross_island>
class CrossIsland(BaseIsland):
    """
    Generates low-dimensional tensor products ('te') combined with simple effects.
    Critical for capturing synergistic interactions between continuous variables.
    """
    def __init__(self, max_terms: int = 4):
        super().__init__("CrossIsland", ['te', 'l', 'c'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        if not available_features: 
            return "1"
        
        n_terms = min(len(available_features), random.randint(1, self.max_terms))
        terms = []
        available = available_features.copy()
        
        for _ in range(n_terms):
            if not available: 
                break
            
            target_feat = random.choice(available)
            allowed_by_selector = search_space.get(target_feat, {}).get("eligible_effects", [])
            
            valid_top_level = [e for e in self.valid_effects if e in allowed_by_selector or e == 'te']
            if not valid_top_level: 
                valid_top_level = ['l']

            eff = kg.suggest_effect_for_feature(target_feat, valid_top_level)
            
            if eff == 'te':
                if len(available) < 2: 
                    break
                
                base_feat = target_feat
                available.remove(base_feat)
                
                interact_feat = kg.suggest_interaction(base_feat, available)
                if not interact_feat: 
                    continue
                available.remove(interact_feat)
                
                allowed_base = search_space.get(base_feat, {}).get("eligible_effects", [])
                allowed_interact = search_space.get(interact_feat, {}).get("eligible_effects", [])

                eff1 = kg.suggest_effect_for_feature(base_feat, [e for e in ['l', 'c', 's'] if e in allowed_base] or ['l'])
                eff2 = kg.suggest_effect_for_feature(interact_feat, [e for e in ['l', 'c', 's'] if e in allowed_interact] or ['l'])
                
                params1 = _get_safe_params(kg, search_space, base_feat, eff1, complexity_cap)
                params2 = _get_safe_params(kg, search_space, interact_feat, eff2, complexity_cap)
                
                p1_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params1.items()])
                p2_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params2.items()])
                
                t1 = f"{eff1}({base_feat}, {p1_str})" if p1_str else f"{eff1}({base_feat})"
                t2 = f"{eff2}({interact_feat}, {p2_str})" if p2_str else f"{eff2}({interact_feat})"
                
                terms.append(f"te({t1}, {t2})")
            else:
                available.remove(target_feat)
                params = _get_safe_params(kg, search_space, target_feat, eff, complexity_cap)
                param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
                terms.append(f"{eff}({target_feat}, {param_str})" if param_str else f"{eff}({target_feat})")

        return _clean_and_join_terms(terms)
#: </cross_island>

#: <smallcontinent>
class SmallContinent(BaseIsland):
    """
    The 'Simple Meta' Island. 
    Allows all mathematical topologies but strictly enforces low-capacity 
    hyperparameters to prevent high-variance overfitting while maintaining diversity.
    """
    def __init__(self, max_terms: int = 4):
        all_effects = ['l', 'c', 's', 'f', 'p', 'w', 'n', 'rbf', 't', 'te']
        super().__init__("SmallContinent", all_effects, max_terms)
        
        self.specialized_islands = [
            LinearIsland(max_terms=1), SplineIsland(max_terms=1),
            FourierIsland(max_terms=1), ChebyshevIsland(max_terms=1),
            WaveletIsland(max_terms=1), NeuralIsland(max_terms=1),
            RBFIsland(max_terms=1), TreeIsland(max_terms=1)
        ]

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        num_islands_to_query = random.randint(1, self.max_terms)
        chosen_islands = random.sample(self.specialized_islands, num_islands_to_query)
        
        composite_terms = []
        for island in chosen_islands:
            island_formula = island.generate(kg, available_features, search_space, complexity_cap=True)
            if island_formula != "1":
                composite_terms.append(island_formula)
                
        return _clean_and_join_terms(composite_terms)
#: </smallcontinent>  

#: <continent>
class Continent(BaseIsland):
    """
    The Meta-Island. Accepts all effects and leverages the specialized islands 
    to sample highly heterogeneous composite mega-formulas.
    """
    def __init__(self, max_terms: int = 15):
        all_effects = ['l', 'c', 's', 'f', 'p', 'w', 'n', 'rbf', 't', 'te']
        super().__init__("Continent", all_effects, max_terms)
        
        self.specialized_islands = [
            LinearIsland(max_terms=2), SplineIsland(max_terms=2),
            FourierIsland(max_terms=2), ChebyshevIsland(max_terms=2),
            WaveletIsland(max_terms=2), NeuralIsland(max_terms=2),
            RBFIsland(max_terms=2), TreeIsland(max_terms=2), 
            CrossIsland(max_terms=2)
        ]

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        num_islands_to_query = random.randint(2, 4)
        chosen_islands = random.sample(self.specialized_islands, num_islands_to_query)
        
        composite_terms = []
        for island in chosen_islands:
            island_formula = island.generate(kg, available_features, search_space, complexity_cap)
            if island_formula != "1":
                composite_terms.append(island_formula)
                
        return _clean_and_join_terms(composite_terms)
#: </continent>


#: <helper_functions>
def _clean_and_join_terms(term_list: List[str]) -> str:
    """
    Safely flattens, deduplicates, and joins additive formula terms.
    Prevents identical terms from compounding and causing design matrix singularities.
    """
    flat_terms = []
    for item in term_list:
        if not item or item == "1":
            continue
        for sub_term in item.split(" + "):
            cleaned = sub_term.strip()
            if cleaned and cleaned not in flat_terms:
                flat_terms.append(cleaned)
                
    return " + ".join(flat_terms) if flat_terms else "1"

def _get_safe_params(kg: KnowledgeGraph, search_space: Dict[str, Any], feat: str, eff: str, complexity_cap: bool = False) -> Dict[str, Any]:
    """
    Helper to fetch learned params from the Knowledge Graph or safely fallback 
    to randomly sampling the mathematically safe grid defined by the EffectSelector.
    If complexity_cap is True, strictly selects the lowest capacity hyperparameter bound.
    """
    grid = search_space.get(feat, {}).get("grids", {}).get(eff, {})
    
    if complexity_cap:
        params = {}
        for k, v in grid.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], (int, float)):
                    params[k] = min(v)
                else:
                    params[k] = v[0]
        return params
        
    params = kg.suggest_parameters(feat, eff)
    if not params:
        params = {k: random.choice(v) for k, v in grid.items() if isinstance(v, list)}
    return params

def _standard_generate(island: BaseIsland, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
    """
    Standard generation loop for univariate and simple additive islands.
    Verifies choices against the strict covariate locks in the search space.
    """
    if not available_features: 
        return "1"
    
    selected_features = random.sample(
        available_features, 
        min(len(available_features), random.randint(1, island.max_terms))
    )
    terms = []

    for feat in selected_features:
        allowed_by_selector = search_space.get(feat, {}).get("eligible_effects", [])
        valid_choices = [e for e in island.valid_effects if e in allowed_by_selector]
        
        if not valid_choices:
            valid_choices = ['l']
        
        eff = kg.suggest_effect_for_feature(feat, valid_choices)
        params = _get_safe_params(kg, search_space, feat, eff, complexity_cap)
            
        param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
        term = f"{eff}({feat}, {param_str})" if param_str else f"{eff}({feat})"
        terms.append(term)

    return _clean_and_join_terms(terms)

def _interaction_generate(island: BaseIsland, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], interaction_effect: str, complexity_cap: bool = False) -> str:
    """
    Specialized generation loop for Deep Islands (Neural, RBF, Tree) that 
    support passing interacting covariates via the 'others' parameter.
    """
    if not available_features: 
        return "1"
    
    selected_features = random.sample(
        available_features, 
        min(len(available_features), random.randint(1, island.max_terms))
    )
    terms = []

    for feat in selected_features:
        valid_effects = search_space.get(feat, {}).get("eligible_effects", island.valid_effects)
        valid_choices = [e for e in island.valid_effects if e in valid_effects] or ['l']
        
        topology = search_space.get(feat, {}).get("topology", "continuous")
        if topology == "continuous" and 'c' in valid_choices:
            valid_choices.remove('c')

        if not valid_choices: 
            valid_choices = ['l']

        eff = kg.suggest_effect_for_feature(feat, valid_choices)
        params = _get_safe_params(kg, search_space, feat, eff, complexity_cap)

        if eff == interaction_effect:
            interacting_feat = kg.suggest_interaction(feat, [f for f in available_features if f != feat])
            if interacting_feat:
                params['others'] = interacting_feat

        param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
        term = f"{eff}({feat}, {param_str})" if param_str else f"{eff}({feat})"
        terms.append(term)

    return _clean_and_join_terms(terms)
#: </helper_functions>

#: <population_nodes_registry>
def get_island_generators() -> List[Callable[[KnowledgeGraph, List[str], Dict[str, Any]], str]]:
    """
    Returns the list of instantiated island generation functions, 
    including the Meta-Continent.
    
    These callables are injected directly into the DragTAM optimizer to 
    initialize and repopulate the evolutionary generations.
    """
    islands = [
        LinearIsland(),
        SplineIsland(),
        FourierIsland(),
        ChebyshevIsland(),
        WaveletIsland(),
        NeuralIsland(),
        RBFIsland(),
        TreeIsland(),
        CrossIsland(),
        SmallContinent(),
        Continent()
    ]
    return [island.generate for island in islands]
#: </population_nodes_registry>