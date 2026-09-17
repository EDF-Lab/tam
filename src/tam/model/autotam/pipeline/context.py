# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Pipeline Context for Automated TAM (AutoTAM).

This module defines the central state holder for the AutoTAM pipeline. 
It carries data splits, tracking metadata, and inferred physical parameters 
across all mathematical stages (Parsing, Profiling, Engineering, and Expansion) 
to ensure a unified state without passing excessive arguments between classes.
"""

#: <context_imports>
import math
import pandas as pd
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Union

from tam.common.utils import parse_formula_to_terms
#: </context_imports>

#: <context_complexity_helpers>
# Nested tensor products deeper than this are costed as a single column instead of recursing.
_MAX_TENSOR_DEPTH = 8


def _int_param(params: Dict[str, Any], name: str, default: int) -> int:
    """Integer value of a term parameter; the default when absent or not yet a number.

    Grid-search templates carry string tokens (e.g. k='grid_k_x_s') until they are resolved.
    """
    try:
        return int(params.get(name, default))
    except (TypeError, ValueError):
        return default


def _split_top_level(text: str, separator: str) -> List[str]:
    """Splits on `separator` outside parentheses, so nested arguments stay intact."""
    parts, current, depth = [], [], 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


@lru_cache(maxsize=4096)
def _parse_terms(rhs: str) -> Tuple[Dict[str, Any], ...]:
    """Parser term dictionaries of a right-hand side, cached. Callers must not mutate them."""
    _, terms = parse_formula_to_terms(f"DUMMY ~ {rhs}")
    return tuple(terms)


def _tensor_sub_terms(term: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Sub-terms of a parsed te(): the parser keeps them as parameter keys such as 's(x, k=10)'."""
    calls = []
    for key in (term.get("params") or {}):
        key = str(key).strip()
        head, bracket, _ = key.partition("(")
        if bracket and head.strip().isalpha():
            calls.append(key)
    if not calls:
        return ()
    try:
        return _parse_terms(" + ".join(calls))
    except ValueError:
        return ()


def term_complexity(term: Dict[str, Any], _depth: int = 0) -> int:
    """Structural degrees of freedom (Primal columns) of one parsed term.

    Read from the parser's parameter dictionary; defaults match an unparameterised term.
    """
    effect = term.get("type")
    params = term.get("params") or {}

    if effect == "te":
        if _depth >= _MAX_TENSOR_DEPTH:
            return 1
        dims = [max(1, term_complexity(sub, _depth + 1)) for sub in _tensor_sub_terms(term)] or [1]
        product = math.prod(dims)
        # Use discounted additive surrogate to prevent penalizing interactions out of the search early.
        return max(1, int(min(product, sum(dims) + product ** 0.5)))
    if effect == "c":
        return max(1, _int_param(params, "n_cat", 2) - 1)
    if effect == "s":
        return max(1, _int_param(params, "k", 10) + _int_param(params, "deg", 3))
    if effect == "f":
        # The Fourier parameter is 'm'; 'k' is accepted as a legacy spelling.
        return max(1, 2 * _int_param(params, "m", _int_param(params, "k", 10)))
    if effect == "p":
        return max(1, _int_param(params, "deg", 3))
    if effect == "rbf":
        return max(1, _int_param(params, "n_centers", 10))
    if effect == "t":
        # TreeEffect: max_leaves (a flat histogram) overrides max_depth (2**depth leaves per tree).
        leaves = _int_param(params, "max_leaves", 2 ** _int_param(params, "max_depth", 2))
        return max(1, _int_param(params, "n_trees", 10) * leaves)
    if effect == "lt":
        # One tree for the level and one for the slope; the factory forces n_trees=1.
        return max(1, 2 * _int_param(params, "max_leaves", 2 ** _int_param(params, "max_depth", 5)))
    if effect == "pid":
        return 3
    if effect == "w":
        return max(1, _int_param(params, "n_scales", 3) * _int_param(params, "n_locations", 10))
    if effect == "n":
        # Heavy structural penalty for deep randomized architectures.
        return max(1, _int_param(params, "n_neurons", 10) * _int_param(params, "n_hidden_layers", 1) * 5)
    return 1


def _token_complexity(token: str) -> int:
    """Cost of one top-level right-hand-side token: a function term, a bare column, or '1'."""
    token = _split_top_level(token, "-")[0].strip()   # drops an intercept removal such as '- 1'
    if not token or token in ("0", "1"):
        return 0
    if "(" not in token:
        return 1                                        # bare column entering linearly
    try:
        return sum(term_complexity(term) for term in _parse_terms(token))
    except ValueError:
        return 1


@lru_cache(maxsize=4096)
def _formula_complexity(formula: str) -> int:
    """Degrees of freedom of a formula string, summed over every model part it contains.

    A dynamic expert is logged as '<dynamic formula> + <static formula>', i.e. two '~'. Each
    part's right-hand side is costed, dropping the next part's left-hand side name, so the static
    base is included rather than read as a single stray token.
    """
    pieces = formula.split("~")
    total = 0
    for position in range(1, len(pieces)):
        rhs = pieces[position]
        if position < len(pieces) - 1:
            rhs = rhs.rsplit("+", 1)[0] if "+" in rhs else ""
        total += sum(_token_complexity(token) for token in _split_top_level(rhs, "+"))
    return max(1, total)
#: </context_complexity_helpers>

#: <context_class>
@dataclass
class PipelineContext:
    """
    Central state holder for the AutoTAM pipeline.
    Carries data splits, metadata, and physical tracking parameters across all stages.
    """
    # 1. Macro-Splits (Chronological continuity for plotting and final evaluation)
    df_fit: Optional[pd.DataFrame] = None
    df_dev: Optional[pd.DataFrame] = None
    df_val: Optional[pd.DataFrame] = None
    historical_tail: Optional[pd.DataFrame] = None
    df_all_aug: Optional[pd.DataFrame] = None

    # 2. Evaluation & Scoring Configuration
    validation_strategy: str = 'auto'
    optimization_metric: str = 'rmse'
    complexity_penalty: float = 1.0  
    cv_folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=list)

    # 3. Metadata & Configuration
    formula_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    target: str = ""
    date_col: Optional[str] = None
    group_col: Optional[str] = None
    lags: List[int] = field(default_factory=list)

    # 4. Search Space
    search_space: Dict[str, Any] = field(default_factory=dict)

    # 5. Physics & Horizon Inference (Populated dynamically during Expansion)
    steps_per_period: int = 1
    adaptive_lags: List[int] = field(default_factory=list)
    inferred_horizon: int = 1
    candidate_windows: List[int] = field(default_factory=list)

#: <context_complexity_methods>
    def estimate_complexity(self, formula: Union[str, List[Dict[str, Any]], None]) -> int:
        """
        Estimates the exact structural degrees of freedom (columns in the Primal matrix)
        generated by a formula to accurately penalize complexity.

        Accepts a formula string or an already parsed term list. A string is split into its
        top-level terms, each function term is parsed once into the parser's term dictionary and
        its parameters are read from that dictionary; results are cached per string, since the
        pipeline scores the same formula for several experts. A combined dynamic formula
        (several '~') is costed as the sum of all its model parts.
        """
        if isinstance(formula, list):
            return self.estimate_complexity_from_terms(formula)
        if not isinstance(formula, str) or "~" not in formula:
            return 1
        return _formula_complexity(formula)

    @staticmethod
    def estimate_complexity_from_terms(parsed_terms: List[Dict[str, Any]]) -> int:
        """Structural degrees of freedom of a parsed genome (parse_formula_to_terms output)."""
        return max(1, sum(term_complexity(term) for term in parsed_terms))

    def penalize_score(self, score: float, formula: str, n_samples: int) -> float:
        """Applies an Information Criterion penalty to mitigate overfitting."""
        if score == float('inf') or n_samples <= 0: return float('inf')
        k = self.estimate_complexity(formula)
        return score * (1.0 + self.complexity_penalty * (k / max(1, n_samples)))
#: </context_complexity_methods>
#: </context_class>