# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Syntax Decoder for Automated TAM (AutoTAM).

This module serves as the Semantic Interpreter for the AutoML pipeline. It translates 
user-defined, high-level string formulas into structured, machine-readable configurations. 

By strictly parsing the Left-Hand Side (targets) and Right-Hand Side (features, lags, 
and pipeline macros), this module establishes the foundational boundaries of the 
Evolutionary Search Space and provides the first layer of defense against Target Leakage.
"""

#: <parser_imports>
import re
import ast
import itertools
from typing import Dict, Any, List, Tuple, Optional, Union
from tam.common.utils import split_args_respecting_parentheses
#: </parser_imports>

__all__ = [
    "canonicalize_term",
    "terms_are_equivalent",
    "term_subsumes",
    "canonicalize_formula",
    "parse_formula_to_terms",
    "FormulaParser",
    "RECOGNIZED_PIPELINE_MACROS",
    "SUPPORTED_EFFECTS",
]

RECOGNIZED_PIPELINE_MACROS = {"AutoPipe", "AdaptTAM", "KalmanTAM", "StaticTAM", "AdTAM"}
SUPPORTED_EFFECTS = {"s", "c", "l", "te", "f", "p", "rbf", "w", "n", "phys", "pid", "t", "lt"}


#: <parser_canonicalization>
def _normalize_param_value(val: Any) -> Tuple[Any, str]:
    """
    Normalizes a parameter value into its typed representation and canonical string format.
    Strings are formatted with single quotes ('...').
    """
    if isinstance(val, str):
        val_str = val.strip()
        if (val_str.startswith("'") and val_str.endswith("'")) or (val_str.startswith('"') and val_str.endswith('"')):
            val_str = val_str[1:-1]
        try:
            parsed_val = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            parsed_val = val_str
    else:
        parsed_val = val

    if isinstance(parsed_val, str):
        formatted = f"'{parsed_val}'"
    elif isinstance(parsed_val, bool):
        formatted = str(parsed_val)
    elif isinstance(parsed_val, (int, float)):
        formatted = str(parsed_val)
    elif parsed_val is None:
        formatted = "None"
    else:
        formatted = repr(parsed_val)

    return parsed_val, formatted


def canonicalize_term(term: Union[str, Dict[str, Any]]) -> str:
    """
    Canonicalizes a mathematical term representation into a normalized, order-invariant string.

    Supports both string representations and AST dictionaries. Handles marginal effects
    (sorting parameters alphabetically, standardizing quotes and literals) and tensor product
    interactions (recursively canonicalizing and sorting sub-terms).

    Args:
        term: A string term (e.g. "c(WeekDays, topo='nominal', n_cat=7)") or an AST dictionary.

    Returns:
        The canonical string representation of the term.
    """
    if not term:
        return ""

    if isinstance(term, str):
        term_str = term.strip()
        if not term_str:
            return ""
        if term_str == "1":
            return "1"

        try:
            parts = split_args_respecting_parentheses(term_str, delimiter="+")
        except ValueError:
            parts = [term_str]

        if len(parts) > 1:
            canonical_parts = [canonicalize_term(p) for p in parts if p.strip() and p.strip() != "1"]
            return " + ".join(sorted(p for p in canonical_parts if p))

        func_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$", term_str)
        if not func_match:
            return term_str

        eff_type = func_match.group(1).strip()
        inner_content = func_match.group(2).strip()

        if eff_type == "te":
            parts = split_args_respecting_parentheses(inner_content)
            sub_terms = []
            kwargs = []
            kwarg_pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=\s*(.*)$")

            for part in parts:
                kw_match = kwarg_pattern.match(part)
                if kw_match:
                    k = kw_match.group(1).strip()
                    v = kw_match.group(2).strip()
                    _, formatted_v = _normalize_param_value(v)
                    kwargs.append((k, formatted_v))
                else:
                    sub_terms.append(canonicalize_term(part))

            sorted_subs = sorted(sub_terms)
            sorted_kwargs = [f"{k}={v}" for k, v in sorted(kwargs, key=lambda x: x[0])]
            all_elements = sorted_subs + sorted_kwargs
            return f"te({', '.join(all_elements)})"
        else:
            parts = split_args_respecting_parentheses(inner_content)
            if not parts:
                return f"{eff_type}()"

            feat = parts[0].strip()
            params = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    _, formatted_v = _normalize_param_value(v)
                    params[k] = formatted_v

            if params:
                param_strs = [f"{k}={v}" for k, v in sorted(params.items())]
                return f"{eff_type}({feat}, {', '.join(param_strs)})"
            else:
                return f"{eff_type}({feat})"

    elif isinstance(term, dict):
        eff_type = term.get("type", "")
        feat = term.get("feature", "")
        params = term.get("params", {})

        if eff_type == "te":
            sub_terms = []
            kwargs = []
            if "sub_terms" in term:
                for st in term["sub_terms"]:
                    sub_terms.append(canonicalize_term(st))
            else:
                for k, v in params.items():
                    if re.match(r"^\s*[a-zA-Z0-9_]+\s*\(", str(k)):
                        sub_terms.append(canonicalize_term(str(k)))
                    elif not str(k).startswith("__"):
                        _, formatted_v = _normalize_param_value(v)
                        kwargs.append((str(k), formatted_v))

            sorted_subs = sorted(sub_terms)
            sorted_kwargs = [f"{k}={v}" for k, v in sorted(kwargs, key=lambda x: x[0])]
            all_elements = sorted_subs + sorted_kwargs
            return f"te({', '.join(all_elements)})"
        else:
            formatted_params = {}
            for k, v in params.items():
                if not str(k).startswith("__"):
                    _, formatted_v = _normalize_param_value(v)
                    formatted_params[str(k)] = formatted_v

            if formatted_params:
                param_strs = [f"{k}={v}" for k, v in sorted(formatted_params.items())]
                return f"{eff_type}({feat}, {', '.join(param_strs)})"
            else:
                return f"{eff_type}({feat})"

    return str(term)


def terms_are_equivalent(term1: Union[str, Dict[str, Any]], term2: Union[str, Dict[str, Any]]) -> bool:
    """
    Checks if two terms are semantically equivalent under parameter permutations,
    quote differences, and sub-term ordering within tensor products.

    Args:
        term1: First term (string or AST dictionary).
        term2: Second term (string or AST dictionary).

    Returns:
        True if the canonicalized representations of both terms are identical.
    """
    return canonicalize_term(term1) == canonicalize_term(term2)


def _parse_term_structure(term: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parses a term (string or AST dictionary) into a normalized hierarchical structure:
    {
        'type': str,
        'feature': str,
        'params': Dict[str, Any],
        'sub_terms': List[Dict[str, Any]],
        'kwargs': Dict[str, Any]
    }

    Args:
        term: A string term representation or an AST dictionary.

    Returns:
        A dictionary describing the parsed term structure, or None if invalid.
    """
    if not term:
        return None

    if isinstance(term, str):
        term_str = term.strip()
        if not term_str or term_str == "1":
            return None

        func_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$", term_str)
        if not func_match:
            return {
                "type": "bare",
                "feature": term_str,
                "params": {},
                "sub_terms": [],
                "kwargs": {},
            }

        eff_type = func_match.group(1).strip()
        inner_content = func_match.group(2).strip()

        if eff_type == "te":
            parts = split_args_respecting_parentheses(inner_content)
            sub_terms = []
            kwargs = {}
            kwarg_pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=\s*(.*)$")

            for part in parts:
                kw_match = kwarg_pattern.match(part)
                if kw_match:
                    k = kw_match.group(1).strip()
                    v = kw_match.group(2).strip()
                    parsed_v, _ = _normalize_param_value(v)
                    kwargs[k] = parsed_v
                else:
                    sub_struct = _parse_term_structure(part)
                    if sub_struct:
                        sub_terms.append(sub_struct)

            return {
                "type": "te",
                "feature": "interaction",
                "params": {},
                "sub_terms": sub_terms,
                "kwargs": kwargs,
            }
        else:
            parts = split_args_respecting_parentheses(inner_content)
            if not parts:
                return {
                    "type": eff_type,
                    "feature": "",
                    "params": {},
                    "sub_terms": [],
                    "kwargs": {},
                }

            feat = parts[0].strip()
            params = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    parsed_v, _ = _normalize_param_value(v.strip())
                    params[k.strip()] = parsed_v

            return {
                "type": eff_type,
                "feature": feat,
                "params": params,
                "sub_terms": [],
                "kwargs": {},
            }

    elif isinstance(term, dict):
        eff_type = term.get("type", "")
        feat = term.get("feature", "")
        params_raw = term.get("params", {})

        if eff_type == "te":
            sub_terms = []
            kwargs = {}
            if "sub_terms" in term:
                for st in term["sub_terms"]:
                    sub_struct = _parse_term_structure(st)
                    if sub_struct:
                        sub_terms.append(sub_struct)
            else:
                for k, v in params_raw.items():
                    k_str = str(k).strip()
                    if re.match(r"^\s*[a-zA-Z0-9_]+\s*\(", k_str):
                        sub_struct = _parse_term_structure(k_str)
                        if sub_struct:
                            sub_terms.append(sub_struct)
                    elif not k_str.startswith("__"):
                        parsed_v, _ = _normalize_param_value(v)
                        kwargs[k_str] = parsed_v

            return {
                "type": "te",
                "feature": "interaction",
                "params": {},
                "sub_terms": sub_terms,
                "kwargs": kwargs,
            }
        else:
            params = {}
            for k, v in params_raw.items():
                k_str = str(k).strip()
                if not k_str.startswith("__"):
                    parsed_v, _ = _normalize_param_value(v)
                    params[k_str] = parsed_v

            return {
                "type": eff_type,
                "feature": feat,
                "params": params,
                "sub_terms": [],
                "kwargs": {},
            }

    return None


def _struct_subsumes(candidate_struct: Dict[str, Any], mandatory_struct: Dict[str, Any]) -> bool:
    """
    Internal recursive helper checking whether candidate structure subsumes mandatory structure.

    Args:
        candidate_struct: Normalized structure of candidate term.
        mandatory_struct: Normalized structure of mandatory term specification.

    Returns:
        True if candidate structure subsumes mandatory structure, False otherwise.
    """
    if candidate_struct["type"] != mandatory_struct["type"]:
        return False

    if mandatory_struct["type"] != "te":
        if candidate_struct["feature"] != mandatory_struct["feature"]:
            return False
        for k, v in mandatory_struct["params"].items():
            if k not in candidate_struct["params"]:
                return False
            if candidate_struct["params"][k] != v:
                return False
        return True
    else:
        for k, v in mandatory_struct["kwargs"].items():
            if k not in candidate_struct["kwargs"]:
                return False
            if candidate_struct["kwargs"][k] != v:
                return False

        candidate_subs = candidate_struct["sub_terms"]
        mandatory_subs = mandatory_struct["sub_terms"]
        if len(candidate_subs) != len(mandatory_subs):
            return False

        for perm in itertools.permutations(candidate_subs):
            if all(_struct_subsumes(c_sub, m_sub) for c_sub, m_sub in zip(perm, mandatory_subs)):
                return True
        return False


def term_subsumes(candidate: Union[str, Dict[str, Any]], mandatory: Union[str, Dict[str, Any]]) -> bool:
    """
    Checks if a candidate term subsumes a mandatory term specification.

    A candidate term subsumes a mandatory term if:
    1. Both have the same basis effect type (e.g., both are 's').
    2. Both target the same feature (or equivalent sub-features for tensor products).
    3. Every parameter explicitly defined in the mandatory specification is present
       with an identical value in the candidate term (free parameters may be tuned).
    4. For tensor products ('te(...)'), sub-terms are matched invariantly to ordering,
       each participating sub-term is subsumed, and top-level kwargs match.

    Args:
        candidate: Candidate term (string or AST dictionary) potentially with tuned free parameters.
        mandatory: Mandatory term specification (string or AST dictionary).

    Returns:
        True if candidate term subsumes the mandatory specification, False otherwise.
    """
    c_struct = _parse_term_structure(candidate)
    m_struct = _parse_term_structure(mandatory)

    if c_struct is None or m_struct is None:
        return False

    return _struct_subsumes(c_struct, m_struct)


def canonicalize_formula(formula: str) -> str:
    """
    Canonicalizes an AutoTAM formula by splitting into LHS and RHS, canonicalizing each
    additive term, removing duplicate terms, and sorting the unique terms alphabetically.

    Args:
        formula: Formula string (e.g. "Load ~ s(Temp, k=10) + c(Day, topo='nominal', n_cat=7)")

    Returns:
        A canonicalized formula string formatted as "{lhs} ~ {term1} + {term2}".

    Raises:
        ValueError: If formula does not contain '~'.
    """
    if "~" not in formula:
        raise ValueError(f"Invalid formula syntax: '{formula}'. Must contain '~'.")

    lhs_str, rhs_str = formula.split("~", 1)
    lhs = lhs_str.strip()

    parts = split_args_respecting_parentheses(rhs_str, delimiter="+")

    canonical_terms_set = set()
    for part in parts:
        c_term = canonicalize_term(part)
        if c_term:
            canonical_terms_set.add(c_term)

    if len(canonical_terms_set) > 1 and "1" in canonical_terms_set:
        canonical_terms_set.remove("1")

    if not canonical_terms_set:
        return f"{lhs} ~ 1"

    sorted_terms = sorted(canonical_terms_set)
    return f"{lhs} ~ {' + '.join(sorted_terms)}"
#: </parser_canonicalization>


#: <parser_utils>
def parse_formula_to_terms(formula: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts the target and parses mathematical terms from a standard GAM formula string.
    
    This utility breaks down complex Right-Hand Side (RHS) definitions into 
    individual mathematical components, isolating the effect type, the target feature, 
    and any specified hyperparameters.

    Args:
        formula (str): The string representation of the model (e.g., 'Y ~ s(X, k=10) + l(Z)')
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: 
            - The target variable string.
            - A list of parsed term dictionaries (e.g., [{'type': 's', 'feature': 'X', 'params': {'k': 10}}, ...]).
    """
    if "~" not in formula:
        raise ValueError(f"Invalid formula syntax: '{formula}'. Must contain '~'.")
        
    target_str, rhs_str = formula.split("~", 1)
    target = target_str.strip()
    
    terms = []
    
    term_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(\s*([^,)]+)(.*?)\)")
    
    for part in rhs_str.split("+"):
        part = part.strip()
        if not part or part == "1":
            continue
            
        match = term_pattern.match(part)
        if match:
            eff = match.group(1).strip()
            feat = match.group(2).strip()
            params_str = match.group(3).strip()
            
            params = {}
            if params_str:
                if params_str.startswith(","):
                    params_str = params_str[1:]
                    
                for p in params_str.split(","):
                    if "=" in p:
                        k, v = p.split("=", 1)
                        k = k.strip()
                        v = v.strip().replace("'", "").replace('"', '')
                        
                        try:
                            v = ast.literal_eval(v)
                        except (ValueError, SyntaxError):
                            pass 
                        params[k] = v
            
            terms.append({"type": eff, "feature": feat, "params": params})
            
    return target, terms
#: </parser_utils>


#: <parser_class>
class FormulaParser:
    """
    Translates the user's high-level AutoTAM formula into a structured AutoML configuration.
    
    Handles the interpretation of the 'AutoPipe' macro and specialized syntax like 
    lag injections (e.g., 'Feature@7' to inject a 7-step autoregressive lag).
    """
    
#: <parser_init>
    def __init__(self):
        """
        Initializes the FormulaParser and pre-compiles the necessary regular expressions
        for efficient, repeated structural extraction.
        """
        self.equation_regex = re.compile(r"^(.*?)\s*~\s*(.*)$")
        self.pipeline_regex = re.compile(r"^([a-zA-Z0-9_]+)\s*\((.*)\)$")
#: </parser_init>

#: <parser_parse_method>
    def parse(self, formula: str, date_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses a full string formula into a structured AutoML configuration.

        Args:
            formula (str): The raw user input (e.g., 'Load ~ AutoPipe(Temp, Humidity, Load@24)')
            date_col (str, optional): The time column to exclude from the mathematical search space.

        Returns:
            Dict[str, Any]: Configuration dictionary containing 'targets', 'features', 
                            'pipeline_type', 'lags', and 'mandatory_terms'.
        """
        match = self.equation_regex.match(formula.strip())
        if not match:
            raise ValueError(f"Invalid formula syntax: '{formula}'. Must contain '~'.")
            
        lhs, rhs = match.groups()
        targets = self._parse_targets(lhs)

        rhs_tokens = split_args_respecting_parentheses(rhs.strip(), delimiter="+")

        macro_tokens = []
        non_macro_tokens = []

        for token in rhs_tokens:
            token_str = token.strip()
            pipe_match = self.pipeline_regex.match(token_str)
            if pipe_match and pipe_match.group(1).strip() in RECOGNIZED_PIPELINE_MACROS:
                macro_tokens.append((token_str, pipe_match.group(1).strip(), pipe_match.group(2)))
            else:
                non_macro_tokens.append(token_str)

        if len(macro_tokens) == 0:
            raise ValueError("AutoTAM formula RHS must contain exactly one pipeline macro (e.g., 'AutoPipe(...)'). None found.")
        elif len(macro_tokens) > 1:
            macro_names = [m[1] for m in macro_tokens]
            raise ValueError(f"AutoTAM formula RHS must contain exactly one pipeline macro, but found {len(macro_tokens)}: {macro_names}.")

        macro_token, pipeline_type, args_str = macro_tokens[0]
        args_str_stripped = args_str.strip()
        if not args_str_stripped:
            raise ValueError(f"Pipeline macro '{pipeline_type}' cannot be empty; specify at least one feature or lag (e.g., 'AutoPipe(x1, x2)').")

        features = []
        lags = {}
        args = split_args_respecting_parentheses(args_str_stripped, delimiter=",")
        for arg in args:
            arg_str = arg.strip()
            if not arg_str:
                continue
            if '@' in arg_str:
                parts = arg_str.split('@')
                feat_name = parts[0].strip()
                try:
                    lag_val = int(parts[1].strip())
                    lags[f"{feat_name}_lag_{lag_val}"] = lag_val
                except ValueError:
                    pass
            else:
                features.append(arg_str)

        mandatory_terms = self._validate_mandatory_terms(non_macro_tokens, pipeline_type)

        if lags and pipeline_type in ["AdTAM", "StaticTAM"]:
            print(f"AutoTAM Parser Warning: Lags detected ({lags}), but pipeline '{pipeline_type}' "
                  f"is static. Consider using 'AdaptTAM', 'KalmanTAM', or 'AutoPipe' for native state-space tracking.")
        
        if date_col and date_col in features:
            features.remove(date_col)
        features = [f for f in features if f not in targets]

        return {
            "targets": targets,
            "features": features,
            "pipeline_type": pipeline_type,
            "lags": lags,
            "mandatory_terms": mandatory_terms,
        }
#: </parser_parse_method>

#: <parser_mandatory_terms_helper>
    def _validate_mandatory_terms(self, non_macro_tokens: List[str], pipeline_type: str) -> List[str]:
        """
        Validates the syntax and effect functions of non-pipeline terms, ensuring they are
        wrapped in supported basis functions, rejecting intercepts/bare identifiers,
        and preventing canonical duplicate terms.

        Args:
            non_macro_tokens: List of stripped non-macro term strings from RHS.
            pipeline_type: The name of the isolated pipeline macro (used in error messages).

        Returns:
            List[str]: Verbatim stripped mandatory terms preserving user declaration order.
        """
        mandatory_terms = []
        seen_canonical = set()

        for token in non_macro_tokens:
            if token == "1":
                raise ValueError(
                    "Explicit intercept '1' is not permitted as a mandatory term. "
                    "TAM automatically prepends a global intercept."
                )

            func_match = self.pipeline_regex.match(token)
            if not func_match:
                raise ValueError(
                    f"Invalid mandatory term '{token}': Bare identifiers without an effect wrapper "
                    f"are not permitted. Wrap the variable in an effect function (e.g., 'l({token})' "
                    f"for linear or 's({token})' for spline), or move it inside '{pipeline_type}(...)'."
                )

            eff_name = func_match.group(1).strip()
            if eff_name not in SUPPORTED_EFFECTS:
                raise ValueError(f"Unknown effect basis function '{eff_name}' in term '{token}'.")

            canon = canonicalize_term(token)
            if canon in seen_canonical:
                raise ValueError(f"Duplicate mandatory term detected in formula: '{token}'.")
            seen_canonical.add(canon)

            mandatory_terms.append(token)

        return mandatory_terms
#: </parser_mandatory_terms_helper>

#: <parser_targets_helper>
    def _parse_targets(self, lhs: str) -> List[str]:
        """
        Extracts and deduplicates target variables from the left-hand side of the formula.
        Handles multi-target syntax if specified (e.g., 'Y1 + Y2 ~ ...' or 'Y1 = Y2 ~ ...').
        """
        targets = []
        normalized_lhs = lhs.replace("=", "+")
        
        parts = [t.strip() for t in normalized_lhs.split("+")]
        for part in parts:
            if part and part not in targets:
                targets.append(part)
                
        return targets
#: </parser_targets_helper>
#: </parser_class>