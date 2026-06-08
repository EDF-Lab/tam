r"""
Utility functions for data loading, preprocessing, and formula parsing.

This module contains shared constants, helper functions for formula interpretation,
and data manipulation routines required by the TAM models.
"""

import importlib.resources as resources
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch

#: <config>
from .hardware import hw
TORCH_DEVICE = hw.device

if hw.supports_float64:
    torch.set_default_dtype(torch.float64)
    NUMPY_DTYPE = np.float64
else:
    torch.set_default_dtype(torch.float32)
    NUMPY_DTYPE = np.float32
#: </config>


def split_args_respecting_parentheses(args_str: str) -> List[str]:
    r"""
    Splits a comma-separated string while respecting nested parentheses.

    This ensures that commas inside function calls (e.g., inside a nested term)
    do not cause incorrect splitting.

    Args:
        args_str: The raw arguments string.

    Returns:
        A list of separated argument strings.
    """
    parts = []
    current_part = []
    paren_count = 0
    
    for char in args_str:
        if char == '(':
            paren_count += 1
            current_part.append(char)
        elif char == ')':
            paren_count -= 1
            current_part.append(char)
        elif char == ',' and paren_count == 0:
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
            
    if current_part:
        part_str = "".join(current_part).strip()
        if part_str:
            parts.append(part_str)
        
    return parts

#: <parsing>
def parse_formula_to_terms(formula_str: str) -> Tuple[str, List[Dict]]:
    r"""
    Parses an R-style formula string into a target variable and term definitions.

    Supports standard terms (splines, linear, fourier) and nested interactions
    such as tensor products.

    Syntax Example:
        'Y ~ s(X1, k=10) + l(X2) + te(s(X1), s(X2))'

    Args:
        formula_str: The formula string to parse.

    Returns:
        A tuple containing:
        - target_col (str): The name of the target variable.
        - parsed_terms (List[Dict]): A list of dictionaries defining each term:
          {'feature': str,  # Feature name or 'interaction'
          'type': str,     # Term type (e.g., 's', 'l', 'te')
          'params': dict   # Parameters extracted from arguments}

    Raises:
        ValueError: If the formula format is invalid or terms are malformed.
    """
    try:
        target_col, terms_part = [s.strip() for s in formula_str.split('~')]
    except ValueError:
        raise ValueError(f"Invalid formula. Must contain exactly one '~'. Received: {formula_str}")
        
    if not terms_part:
         raise ValueError(f"Invalid formula. No terms found after '~'. Received: {formula_str}")
         
    terms_list_str = [s.strip() for s in terms_part.split('+')]
    parsed_terms = []
    
    func_regex = re.compile(r'^\s*(\w+)\s*\((.*?)\)\s*$')
    param_regex = re.compile(r'^\s*(\w+)\s*=\s*(.+)\s*$')

    for term_str in terms_list_str:
        if term_str == '1':
            continue
        
        func_match = func_regex.match(term_str)
        
        if func_match:
            term_type_key = func_match.group(1).lower()
            args_str = func_match.group(2)
            
            args_parts = split_args_respecting_parentheses(args_str)
            
            params = {}
            feature_name = "unknown"

            if term_type_key == 'te':
                feature_name = "interaction"
                for i, arg in enumerate(args_parts):
                    param_match = param_regex.match(arg)
                    if param_match:
                        key = param_match.group(1)
                        value_str = param_match.group(2).strip()
                        
                        if (value_str.startswith("'") and value_str.endswith("'")) or \
                           (value_str.startswith('"') and value_str.endswith('"')):
                            value = value_str[1:-1]
                        elif value_str == 'True':
                            value = True
                        elif value_str == 'False':
                            value = False
                        elif value_str == 'None':
                            value = None
                        else:
                            try:
                                value = int(value_str)
                            except ValueError:
                                try:
                                    value = float(value_str)
                                except ValueError:
                                    value = value_str
                        params[key] = value
                    else:
                        unique_arg = arg + " " * i
                        params[unique_arg] = None 
                        try:
                            _, sub_parsed = parse_formula_to_terms(f"DUMMY ~ {arg}")
                            for j, st in enumerate(sub_parsed):
                                for sub_k, sub_v in st['params'].items():
                                    params[f"__sub_{st['feature']}_{sub_k}_{i}_{j}"] = sub_v
                        except ValueError:
                            pass
            
            else:
                if not args_parts:
                    raise ValueError(f"Term '{term_str}' has no arguments.")
                
                feature_name = args_parts[0].strip()
                
                if len(args_parts) > 1:
                    for arg in args_parts[1:]:
                        if not arg:
                            continue
                        
                        param_match = param_regex.match(arg)
                        if param_match:
                            key = param_match.group(1)
                            value_str = param_match.group(2).strip()
                            
                            # Handle quoted strings, booleans, ints, and floats natively
                            if (value_str.startswith("'") and value_str.endswith("'")) or \
                               (value_str.startswith('"') and value_str.endswith('"')):
                                value = value_str[1:-1]
                            elif value_str == 'True':
                                value = True
                            elif value_str == 'False':
                                value = False
                            elif value_str == 'None':
                                value = None
                            else:
                                try:
                                    value = int(value_str)
                                except ValueError:
                                    try:
                                        value = float(value_str)
                                    except ValueError:
                                        value = value_str
                                
                            params[key] = value
                        else:
                            raise ValueError(f"Malformed argument '{arg}' in term '{term_str}'")
                        
            parsed_terms.append({
                'feature': feature_name,
                'type': term_type_key,
                'params': params
            })
        
        else:
            raise ValueError(
                f"Term '{term_str}' is malformed. "
                f"All terms must be function calls (e.g., 'l({term_str})')."
            )
            
    return target_col, parsed_terms
#: </parsing>

#: <dataset>
def load_national_dataset() -> pd.DataFrame:
    r"""
    Loads and preprocesses the national French electricity consumption dataset.

    Performs feature engineering to encode historical events (COVID-19 lockdowns,
    energy crisis) and creates a normalized time feature.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    path = resources.files('tam.data').joinpath('dataset_national.csv')
    with path.open('r') as f:
        data = pd.read_csv(f)

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data['day_type_week'] = data['day_type_week'].astype(np.float64)

    # --- Feature Engineering: Events ---
    data["confinement_covid"] = 0
    data["crise_covid"] = 0
    data["crise_energie"] = 0

    # COVID-19 Lockdowns
    lockdowns = [
        (20200317, 20200511),
        (20201030, 20201215),
        (20210403, 20210503)
    ]
    for start, end in lockdowns:
        mask = ((data["Date"] >= start) & (data["Date"] < end))
        data.loc[mask, "confinement_covid"] = 1

    # Post-lockdown Crisis
    mask_covid = ((data["Date"] >= 20200511) & (data["Date"] < 20221006))
    data.loc[mask_covid, "crise_covid"] = data.loc[mask_covid, "day_type_week"] + 1
    
    jf_mask = (mask_covid) & (data["day_type_jf"] == 1)
    data.loc[jf_mask, "crise_covid"] = 10 * data.loc[jf_mask, "day_type_jf"] + 1

    # Energy Crisis
    mask_energy = data["Date"] >= 20221006
    data.loc[mask_energy, "crise_energie"] = data.loc[mask_energy, "day_type_week"] + 1
    
    jf_energy_mask = (mask_energy) & (data["day_type_jf"] == 1)
    data.loc[jf_energy_mask, "crise_energie"] = 10 * data.loc[jf_energy_mask, "day_type_jf"] + 1

    # Normalized Time (0 to pi)
    n = len(data)
    data['time'] = np.array([i / n * np.pi for i in range(n)], dtype=NUMPY_DTYPE)

    return data
#: </dataset>


def _check_features(dataset: pd.DataFrame, required_features: List[str]) -> None:
    r"""
    Ensures all required feature columns exist in the dataset.

    Args:
        dataset: The DataFrame to validate.
        required_features: List of mandatory column names.

    Raises:
        KeyError: If any features are missing.
    """
    missing_features = set(required_features) - set(dataset.columns)
    if missing_features:
        raise KeyError(f"Missing required features: {list(missing_features)}")

#: <balance>
def _balance_groups(
    dataset: pd.DataFrame,
    group_col: str,
    date_col: str,
    method: str = 'drop'
) -> Tuple[pd.Series, pd.DataFrame]:
    r"""
    Balances groups in a DataFrame to ensure consistent sizes.

    Args:
        dataset: The source DataFrame.
        group_col: Column defining the groups.
        date_col: Date column used for padding logic.
        method: Strategy to use ('drop' to truncate, 'fill' to pad).

    Returns:
        Tuple[pd.Series, pd.DataFrame]: A boolean mask indicating original rows,
        and the balanced DataFrame.
    """    
    if method not in ['drop', 'fill']:
        raise ValueError("Method must be 'drop' or 'fill'.")

    if dataset.empty:
        return pd.Series(dtype=bool), dataset.copy()

    group_counts = dataset[group_col].value_counts()
    
    if group_counts.empty:
        return pd.Series(dtype=bool), dataset.copy()

    min_count, max_count = group_counts.min(), group_counts.max()

    if min_count == max_count:
        return pd.Series(True, index=dataset.index), dataset.copy()

    if method == 'drop':
        balanced_df = dataset.groupby(group_col).head(min_count)
        mask = dataset.index.isin(balanced_df.index)
        return mask, balanced_df.copy()

    if method == 'fill':
        rows_to_add = []
        groups_to_fill = group_counts[group_counts < max_count].index

        if len(dataset) > 1 and dataset[date_col].max() != dataset[date_col].min():
            delta = (dataset[date_col].max() - dataset[date_col].min()) / len(dataset)
        else:
            delta = pd.Timedelta(days=1)
            
        fake_date = dataset[date_col].max() + delta

        for value in groups_to_fill:
            num_missing = max_count - group_counts[value]
            
            group_df = dataset[dataset[group_col] == value]
            if group_df.empty: 
                continue
            
            last_row = group_df.iloc[-1:].copy()
            
            for _ in range(num_missing):
                last_row[date_col] = fake_date
                rows_to_add.append(last_row.copy())
                fake_date += delta
        
        if not rows_to_add:
            return pd.Series(True, index=dataset.index), dataset.copy()
            
        new_rows_df = pd.concat(rows_to_add, ignore_index=True)
        balanced_df = pd.concat([dataset, new_rows_df], ignore_index=True)
        
        original_mask = pd.Series(True, index=dataset.index)
        new_mask = pd.Series(False, index=new_rows_df.index)
        
        mask = pd.concat([original_mask, new_mask])
        
        return mask.reset_index(drop=True), balanced_df.reset_index(drop=True)
#: </balance>

def _ensure_dummies(df: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    r"""Injects dummy grouping and temporal columns if they are absent."""
    df_out = df.copy()
    if group_col == "__dummy_group__" and "__dummy_group__" not in df_out.columns:
        df_out["__dummy_group__"] = "global_group"
    if date_col == "__dummy_date__" and "__dummy_date__" not in df_out.columns:
        df_out["__dummy_date__"] = pd.date_range(start="2000-01-01", periods=len(df_out), freq="D")
    return df_out

def _cleanup_dummies(df: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    r"""Removes the dummy columns before returning the DataFrame to the user."""
    cols_to_drop = [
        c for c in [group_col, date_col] 
        if c in ["__dummy_group__", "__dummy_date__"] and c in df.columns
    ]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df