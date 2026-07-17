# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
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
from typing import Dict, Any, List, Tuple, Optional
#: </parser_imports>


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
                            'pipeline_type', and dynamically extracted 'lags'.
        """
        match = self.equation_regex.match(formula.strip())
        if not match:
            raise ValueError(f"Invalid formula syntax: '{formula}'. Must contain '~'.")
            
        lhs, rhs = match.groups()
        targets = self._parse_targets(lhs)
        
        pipeline_type = "AutoPipe"
        features = []
        lags = {}
        
        pipe_match = self.pipeline_regex.match(rhs.strip())
        if pipe_match:
            pipeline_type = pipe_match.group(1).strip()
            args_str = pipe_match.group(2)
            
            args = [arg.strip() for arg in args_str.split(",")]
            for arg in args:
                if not arg:
                    continue
                if '@' in arg:
                    parts = arg.split('@')
                    feat_name = parts[0].strip()
                    try:
                        lag_val = int(parts[1].strip())
                        lags[f"{feat_name}_lag_{lag_val}"] = lag_val
                    except ValueError:
                        pass
                else:
                    features.append(arg)
        else:
            features = [f.strip() for f in rhs.split("+") if f.strip()]

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
            "lags": lags
        }
#: </parser_parse_method>

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