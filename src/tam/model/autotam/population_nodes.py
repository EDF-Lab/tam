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
import re
import random
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from .knowledge_graph import KnowledgeGraph

# Strict Covariate Lock (spec I): a single feature may carry at most this many
# active bases (e.g. s(Temp) + f(Temp) is the maximum allowed for Temp).
MAX_ACTIVE_EFFECTS_PER_FEATURE = 2
# Deep Islands condition one term on at most this many categorical partners (others='a|b|...').
# The reference hybrids condition on five day-type indicators at once.
MAX_INTERACTION_PARTNERS = 5
# Share of deep-basis terms (n, rbf, t) drawn without partners, so plain terms stay reachable:
# otherwise others= is attached whenever a categorical exists, and only a mutation can drop it.
PLAIN_DEEP_TERM_PROBABILITY = 0.25
#: </population_nodes_imports>

#: <population_nodes_base>
class BaseIsland:
    """
    Base configuration for evolutionary formula generation islands.
    
    Every island enforces a strict limit on the number of terms it can generate 
    (`max_terms`) to encourage parsimonious models and prevent formula bloat.
    """
    # The Island's own basis inside the tensor products it builds (None: the Knowledge Graph chooses).
    tensor_family: Optional[str] = None

    def __init__(self, name: str, valid_effects: List[str], max_terms: int = 10):
        self.name = name
        self.valid_effects = valid_effects
        self.max_terms = max_terms
        # Mutant-UCB: the reigning champion genome (list of parsed-term dicts) for this Island,
        # set by the engine whenever one of this Island's formulas enters the Champion Archive.
        self.champion_genome: List[Dict[str, Any]] = None

    def set_champion(self, genome: List[Dict[str, Any]]) -> None:
        """Stores the Island's reigning champion genome, used as the seed for mutate()."""
        self.champion_genome = genome

    def is_viable(self, available_features: List[str], search_space: Dict[str, Any]) -> bool:
        """Whether this Island can build any candidate on the given features.

        DragTAM leaves non-viable Islands out of the bandit: their empty candidates fail to fit,
        and a failed fit records no reward, so such an Island would stay "unqueried" and be
        selected first on every draw.
        """
        return True

    def mutate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any]) -> str:
        """
        DRAGAM/DRAGON-style structural graph edit of this Island's champion genome.

        Applies exactly ONE operator, then returns a covariate-lock-respecting RHS string:
          * Deletion (parsimony): strip one additive term.
          * Insertion (synergy): add a highly-weighted unused feature via the Island's topology.
          * Content modification (tuning): perturb one hyperparameter within the EffectSelector grid.
        Falls back to generate() when there is no champion yet.
        """
        if not self.champion_genome:
            return self.generate(kg, available_features, search_space)

        genome = [{'type': t['type'], 'feature': t['feature'], 'params': dict(t.get('params', {}))}
                  for t in self.champion_genome]
        present = {t['feature'] for t in genome}
        unused = [f for f in available_features if f not in present]

        ops = []
        if len(genome) > 1:
            ops.append('delete')
        if unused:
            ops.append('insert')
        if genome:
            ops.append('modify')
        if not ops:
            return self.generate(kg, available_features, search_space)

        op = random.choice(ops)
        if op == 'delete':
            genome.pop(random.randrange(len(genome)))

        elif op == 'insert':
            feat = random.choice(unused)
            allowed = search_space.get(feat, {}).get("eligible_effects", [])
            choices = [e for e in self.valid_effects if e in allowed and e != 'te'] or ['l']
            eff = kg.suggest_effect_for_feature(feat, choices)
            params = _get_safe_params(kg, search_space, feat, eff, complexity_cap=False)
            genome.append({'type': eff, 'feature': feat, 'params': params})

        elif op == 'modify':
            term = random.choice(genome)
            grid = search_space.get(term['feature'], {}).get("grids", {}).get(term['type'], {})
            numeric_keys = [k for k, v in grid.items() if isinstance(v, list) and v and isinstance(v[0], (int, float))]
            if numeric_keys:
                k = random.choice(numeric_keys)
                term['params'][k] = random.choice(grid[k])
                if k in ("max_depth", "max_leaves"):      # switching tree architecture drops the other
                    term['params'].pop("max_leaves" if k == "max_depth" else "max_depth", None)

        return _genome_to_rhs(genome)

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
    """Proposes linear ('l') and categorical ('c') terms."""
    def __init__(self, max_terms: int = 10):
        super().__init__("LinearIsland", ['l', 'c', 'pid', 'te'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class SplineIsland(BaseIsland):
    """Proposes smooth, localized non-linearities via penalized splines ('s')."""
    tensor_family = 's'

    def __init__(self, max_terms: int = 10):
        super().__init__("SplineIsland", ['s', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class FourierIsland(BaseIsland):
    """Proposes global periodic bases ('f') ideal for capturing strict seasonalities."""
    tensor_family = 'f'

    def __init__(self, max_terms: int = 10):
        super().__init__("FourierIsland", ['f', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class ChebyshevIsland(BaseIsland):
    """Proposes orthogonal polynomial expansions ('p') for continuous trends."""
    tensor_family = 'p'

    def __init__(self, max_terms: int = 10):
        super().__init__("ChebyshevIsland", ['p', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)


class WaveletIsland(BaseIsland):
    """Proposes highly localized wavelet bases ('w') for sharp structural breaks or spikes."""
    tensor_family = 'w'

    def __init__(self, max_terms: int = 10):
        super().__init__("WaveletIsland", ['w', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _standard_generate(self, kg, available_features, search_space, complexity_cap)
#: </standard_islands>

#: <interaction_islands>
class NeuralIsland(BaseIsland):
    """Proposes shallow neural network components ('n') capable of dense feature interactions."""
    def __init__(self, max_terms: int = 5):
        super().__init__("NeuralIsland", ['n', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect='n', complexity_cap=complexity_cap)


class RBFIsland(BaseIsland):
    """Proposes Radial Basis Functions ('rbf') for distance-based spatial/temporal modeling."""
    def __init__(self, max_terms: int = 5):
        super().__init__("RBFIsland", ['rbf', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect='rbf', complexity_cap=complexity_cap)


class TreeIsland(BaseIsland):
    """Proposes random-tree ('t') and linear-tree ('lt') components for jagged, high-frequency signals."""
    def __init__(self, max_terms: int = 5):
        super().__init__("TreeIsland", ['t', 'lt', 'l', 'c', 'te', 'pid'], max_terms)

    def generate(self, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
        return _interaction_generate(self, kg, available_features, search_space, interaction_effect=('t', 'lt'), complexity_cap=complexity_cap)
#: </interaction_islands>


#: <smallcontinent>
class SmallContinent(BaseIsland):
    """
    The 'Simple Meta' Island. 
    Allows all mathematical topologies but strictly enforces low-capacity 
    hyperparameters to prevent high-variance overfitting while maintaining diversity.
    """
    def __init__(self, max_terms: int = 4):
        all_effects = ['l', 'c', 's', 'f', 'p', 'w', 'n', 'rbf', 't', 'lt', 'pid', 'te']
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
        all_effects = ['l', 'c', 's', 'f', 'p', 'w', 'n', 'rbf', 't', 'lt', 'pid', 'te']
        super().__init__("Continent", all_effects, max_terms)
        
        self.specialized_islands = [
            LinearIsland(max_terms=2), SplineIsland(max_terms=2),
            FourierIsland(max_terms=2), ChebyshevIsland(max_terms=2),
            WaveletIsland(max_terms=2), NeuralIsland(max_terms=2),
            RBFIsland(max_terms=2), TreeIsland(max_terms=2)
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
# Smooth bases a tensor product may use on a continuous feature crossed with a categorical, at the
# feature's profiled capacity (never the linear 'l').
SMOOTH_INTERACTION_BASES = ('f', 's', 'w', 'p')
# Bound continuous tensor products to small fixed bases (max ~25 columns) to prevent VRAM explosions.
CONTINUOUS_TENSOR_BASES = {
    's': {'k': 3, 'deg': 2, 'p': 2},
    'f': {'m': 2, 's': 2},
    'p': {'deg': 4, 's': 2},
}
# Tensor products allowed in one formula.
MAX_TENSOR_TERMS = 2


def _smooth_interaction_bases(features: List[str], search_space: Dict[str, Any]) -> List[str]:
    """Continuous features diagnosed non-linear by the FeatureProfiler that admit a smooth basis.

    A feature without a profile (discrete topology, or not diagnosed) is excluded: the profile is
    the only evidence that a curve, rather than a line, is needed on this axis.
    """
    bases = []
    for feat in features:
        space = search_space.get(feat, {})
        profile = space.get("profile") or {}
        eligible = space.get("eligible_effects", [])
        if (space.get("topology") == "continuous" and profile.get("linear") is False
                and any(eff in eligible for eff in SMOOTH_INTERACTION_BASES)):
            bases.append(feat)
    return bases


def _categorical_interaction_partners(features: List[str], search_space: Dict[str, Any]) -> List[str]:
    """Discrete features with at least two levels in the fitting window.

    A categorical that is constant in the window (e.g. a crisis flag before the crisis) would make
    the tensor product collapse onto its continuous marginal.
    """
    partners = []
    for feat in features:
        space = search_space.get(feat, {})
        levels = space.get("grids", {}).get("c", {}).get("n_cat") or [0]
        if (space.get("topology") == "discrete" and 'c' in space.get("eligible_effects", [])
                and int(levels[0]) >= 2):
            partners.append(feat)
    return partners


def _sample_partners(kg: KnowledgeGraph, feat: str, partners: List[str]) -> List[str]:
    """Draws between one and MAX_INTERACTION_PARTNERS distinct partners for a deep-basis term.

    The count is uniform. Each partner is chosen by the Knowledge Graph's interaction score among
    the partners not yet chosen, so learnt synergies steer a subset as they steer a single partner.
    """
    if not partners:
        return []
    remaining = list(partners)
    chosen: List[str] = []
    for _ in range(random.randint(1, min(MAX_INTERACTION_PARTNERS, len(remaining)))):
        pick = kg.suggest_interaction(feat, remaining)
        if pick is None:
            break
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _render_term(eff: str, feat: str, params: Dict[str, Any]) -> str:
    """Renders one additive term, quoting string parameters."""
    param_str = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items())
    return f"{eff}({feat}, {param_str})" if param_str else f"{eff}({feat})"


#: <tensor_term>
def _tensor_partners(feat: str, features: List[str], search_space: Dict[str, Any]) -> List[str]:
    """Features a tensor product on ``feat`` may cross: categoricals with at least two levels and
    other continuous features diagnosed non-linear."""
    others = [f for f in features if f != feat]
    return (_categorical_interaction_partners(others, search_space)
            + _smooth_interaction_bases(others, search_space))


def _tensor_term(kg: KnowledgeGraph, search_space: Dict[str, Any], base: str, partner: str,
                 family: Optional[str] = None) -> Optional[str]:
    """Builds a tensor product of ``base`` with ``partner``; None when no allowed basis exists.

    With a categorical partner the product is te(<smooth basis>(base), c(partner)) at the base's
    profiled capacity: a conditional curve per level. With a continuous partner both sides take a
    small fixed basis from CONTINUOUS_TENSOR_BASES, so the surface stays within 25 columns. The
    basis of ``base`` is the Island's own family when it is allowed; otherwise, and for a continuous
    partner, the Knowledge Graph chooses.
    """
    def basis(feat: str, allowed) -> Optional[str]:
        eligible = search_space.get(feat, {}).get("eligible_effects", [])
        choices = [eff for eff in allowed if eff in eligible]
        if not choices:
            return None
        if feat == base and family in choices:
            return family
        return kg.suggest_effect_for_feature(feat, choices)

    if search_space.get(partner, {}).get("topology") == "discrete":
        eff = basis(base, SMOOTH_INTERACTION_BASES)
        if eff is None:
            return None
        base_params = _get_safe_params(kg, search_space, base, eff, complexity_cap=True)
        partner_params = _get_safe_params(kg, search_space, partner, 'c', complexity_cap=True)
        return f"te({_render_term(eff, base, base_params)}, {_render_term('c', partner, partner_params)})"

    sides = []
    for feat in (base, partner):
        eff = basis(feat, tuple(CONTINUOUS_TENSOR_BASES))
        if eff is None:
            return None
        params = dict(CONTINUOUS_TENSOR_BASES[eff])
        grid_ap = _get_safe_params(kg, search_space, feat, eff, complexity_cap=True).get('ap')
        if grid_ap is not None:
            params['ap'] = grid_ap
        sides.append(_render_term(eff, feat, params))
    return f"te({sides[0]}, {sides[1]})"


def _choose_effect(island: BaseIsland, kg: KnowledgeGraph, feat: str, choices: List[str],
                   available_features: List[str], search_space: Dict[str, Any],
                   n_tensors: int) -> Tuple[str, Optional[str]]:
    """Draws the effect for ``feat``; returns (effect, tensor-product term or None).

    'te' joins the Knowledge Graph's choices when the Island allows it, the formula holds fewer than
    MAX_TENSOR_TERMS products, ``feat`` is a continuous feature diagnosed non-linear and a partner
    exists. A product that cannot be built falls back to a draw among the other choices.
    """
    partners: List[str] = []
    if ('te' in island.valid_effects and n_tensors < MAX_TENSOR_TERMS
            and _smooth_interaction_bases([feat], search_space)):
        partners = _tensor_partners(feat, available_features, search_space)
    plain = [eff for eff in choices if eff != 'te'] or ['l']
    eff = kg.suggest_effect_for_feature(feat, (plain + ['te']) if partners else plain)
    if eff == 'te':
        partner = kg.suggest_interaction(feat, partners)
        term = _tensor_term(kg, search_space, feat, partner, island.tensor_family) if partner else None
        if term:
            return 'te', term
        eff = kg.suggest_effect_for_feature(feat, plain)
    return eff, None
#: </tensor_term>


def _clean_and_join_terms(term_list: List[str]) -> str:
    """
    Safely flattens, deduplicates, and joins additive formula terms.
    Prevents identical terms from compounding and causing design matrix singularities.
    Keeps at most MAX_TENSOR_TERMS tensor products, including across composed Island formulas.
    """
    flat_terms = []
    feature_base_counts: Dict[str, int] = {}
    n_tensors = 0
    for item in term_list:
        if not item or item == "1":
            continue
        for sub_term in item.split(" + "):
            cleaned = sub_term.strip()
            if not cleaned or cleaned in flat_terms:
                continue
            # Strict Covariate Lock: cap a feature at MAX_ACTIVE_EFFECTS_PER_FEATURE
            # bases (interactions 'te' are exempt, they are not single-feature bases).
            match = re.match(r'\s*([a-z]{1,3})\s*\(\s*([A-Za-z0-9_\.]+)', cleaned)
            if match and match.group(1) == 'te':
                if n_tensors >= MAX_TENSOR_TERMS:
                    continue
                n_tensors += 1
            elif match:
                feat = match.group(2)
                if feature_base_counts.get(feat, 0) >= MAX_ACTIVE_EFFECTS_PER_FEATURE:
                    continue
                feature_base_counts[feat] = feature_base_counts.get(feat, 0) + 1
            flat_terms.append(cleaned)

    return " + ".join(flat_terms) if flat_terms else "1"

def _genome_to_rhs(genome: List[Dict[str, Any]]) -> str:
    """Renders a genome (list of parsed-term dicts) back into a covariate-lock-respecting RHS."""
    terms = []
    for t in genome:
        eff, feat, params = t['type'], t['feature'], t.get('params', {})
        if eff == 'te':
            subs = [k for k in params.keys() if re.match(r'^\s*[a-zA-Z]{1,4}\s*\(', str(k))]
            terms.append(f"te({', '.join(subs)})")
        else:
            param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
            terms.append(f"{eff}({feat}, {param_str})" if param_str else f"{eff}({feat})")
    return _clean_and_join_terms(terms)

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
        return _one_tree_architecture(params, keep_depth=True)
        
    params = kg.suggest_parameters(feat, eff)
    if not params:
        params = {k: random.choice(v) for k, v in grid.items() if isinstance(v, list)}
    return _one_tree_architecture(params, keep_depth=False)


def _one_tree_architecture(params: Dict[str, Any], keep_depth: bool) -> Dict[str, Any]:
    """Keeps either max_depth or max_leaves: a tree is an oblivious binary tree or a flat histogram.

    TreeEffect ignores max_depth whenever max_leaves is set, so a term carrying both would silently
    become a histogram. The capped (low-capacity) draw keeps the depth; otherwise one is drawn.
    """
    if "max_depth" in params and "max_leaves" in params:
        params = dict(params)
        params.pop("max_leaves" if keep_depth or random.random() < 0.5 else "max_depth")
    return params

def _standard_generate(island: BaseIsland, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], complexity_cap: bool = False) -> str:
    """
    Standard generation loop for univariate and simple additive islands.
    Verifies choices against the strict covariate locks in the search space; a feature may also
    enter a tensor product (_choose_effect).
    """
    if not available_features:
        return "1"

    selected_features = random.sample(
        available_features,
        min(len(available_features), random.randint(1, island.max_terms))
    )
    terms = []
    n_tensors = 0

    for feat in selected_features:
        allowed_by_selector = search_space.get(feat, {}).get("eligible_effects", [])
        valid_choices = [e for e in island.valid_effects if e in allowed_by_selector] or ['l']

        eff, tensor = _choose_effect(island, kg, feat, valid_choices, available_features, search_space, n_tensors)
        if tensor:
            terms.append(tensor)
            n_tensors += 1
            continue
        params = _get_safe_params(kg, search_space, feat, eff, complexity_cap)
        terms.append(_render_term(eff, feat, params))

    return _clean_and_join_terms(terms)

def _interaction_generate(island: BaseIsland, kg: KnowledgeGraph, available_features: List[str], search_space: Dict[str, Any], interaction_effect: Union[str, Tuple[str, ...]], complexity_cap: bool = False) -> str:
    """
    Specialized generation loop for Deep Islands (Neural, RBF, Tree) that support passing
    interacting covariates via the 'others' parameter; a feature may also enter a tensor product
    (_choose_effect).
    """
    if not available_features:
        return "1"

    selected_features = random.sample(
        available_features,
        min(len(available_features), random.randint(1, island.max_terms))
    )
    terms = []
    n_tensors = 0

    for feat in selected_features:
        valid_effects = search_space.get(feat, {}).get("eligible_effects", island.valid_effects)
        valid_choices = [e for e in island.valid_effects if e in valid_effects] or ['l']

        topology = search_space.get(feat, {}).get("topology", "continuous")
        if topology == "continuous" and 'c' in valid_choices:
            valid_choices.remove('c')

        if not valid_choices:
            valid_choices = ['l']

        eff, tensor = _choose_effect(island, kg, feat, valid_choices, available_features, search_space, n_tensors)
        if tensor:
            terms.append(tensor)
            n_tensors += 1
            continue
        params = _get_safe_params(kg, search_space, feat, eff, complexity_cap)

        is_deep = eff == interaction_effect if isinstance(interaction_effect, str) else eff in interaction_effect
        if is_deep and random.random() >= PLAIN_DEEP_TERM_PROBABILITY:
            # Ensure categorical partners have variance in the fitting window to prevent flat marginals.
            partners = _categorical_interaction_partners(
                [f for f in available_features if f != feat], search_space)
            chosen = _sample_partners(kg, feat, partners)
            if chosen:
                params['others'] = "|".join(sorted(chosen))

        terms.append(_render_term(eff, feat, params))

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
    return [island.generate for island in get_island_objects()]


def get_island_objects() -> List[BaseIsland]:
    """
    Returns instantiated Island objects (the Mutant-UCB 'arms').

    Unlike get_island_generators (bare callables), these carry per-island state
    (name, champion_genome, mutate) required by the Mutant-UCB engine in DragTAM.
    """
    return [
        LinearIsland(),
        SplineIsland(),
        FourierIsland(),
        ChebyshevIsland(),
        WaveletIsland(),
        NeuralIsland(),
        RBFIsland(),
        TreeIsland(),
        SmallContinent(),
        Continent()
    ]
#: </population_nodes_registry>