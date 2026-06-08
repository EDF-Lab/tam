"""
Factory functions for creating and assembling effects.

This module acts as the "Builder" pattern implementation for the model. 
It orchestrates:
1.  Instantiation of specific `BaseEffect` subclasses from parsed formula terms.
2.  Aggregation of individual feature maps into the global Design Matrix.
3.  Construction of the global block-diagonal Penalty Matrix.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
import re 
from ._base_effects import BaseEffect
from ._linear import OffsetEffect, LinearEffect
from ._fourier import FourierEffect
from ._spline import SplineEffect
from ._categorical import CategoricalEffect
from ._chebyshev import ChebyshevEffect
from ._wavelet import WaveletEffect
from ._neural import NeuralEffect
from ._rbf import RBFEffect
from ._tensor import TensorProductEffect
from ._physics import UniversalPhysicsEffect
from ._tree import TreeEffect
from ._linear_tree import LinearTreeEffect
from._pid import PIDEffect
from tam.common.utils import parse_formula_to_terms

#: <create_effects>
def create_effects_from_parsed_terms(
    parsed_terms: List[Dict],
    token_values: Dict[str, Any],
    default_alpha_p: float,
    include_offset: bool = True,
    data_info: Optional[Dict[str, Any]] = None 
) -> List[BaseEffect]:
    """
    Instantiates a list of Effect objects based on parsed formula terms.

    This function handles:
    - Token substitution for hyperparameters (Dependency Injection from Grid Search).
    - Parsing of specific arguments for each effect type (Splines, Fourier, etc.).
    - Recursive creation of sub-effects for Tensor Products.

    Args:
        parsed_terms: List of term dictionaries returned by the formula parser.
        token_values: Dictionary of concrete values for hyperparameter tokens
                      (e.g., {'gk_la': 10}).
        default_alpha_p: Default log10(lambda_p) if not specified.
        include_offset: Whether to prepend an OffsetEffect (Intercept). 
                        False for recursive calls (e.g., inside 'te()').

    Returns:
        List of instantiated BaseEffect objects.
    """
    effects_list = []
    
    token_name_regex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)')

    if include_offset:
        offset_ap = token_values.get('ap_offset', default_alpha_p)
        lambda_p=10**float(offset_ap)
        effects_list.append(OffsetEffect(lambda_p, 'continue'))
    
    for term in parsed_terms:
        feature_name = term['feature']
        ttype = term['type']
        params = term['params'].copy()

        params_resolved = {}
        for key, val in params.items():
            resolved_val = val
            
            if isinstance(val, str):
                if val in token_values:
                    resolved_val = token_values[val]
            
            params_resolved[key] = resolved_val
        
        ap_val = params_resolved.get('ap', default_alpha_p)
        try:
            lambda_p = 10**float(ap_val)
        except (ValueError, TypeError):
             raise ValueError(f"Invalid value for 'ap' in term '{feature_name}': {ap_val}")

        #: <parse_linear>
        if ttype == 'l':
            scaled = float(params_resolved.get('scaled', np.pi))
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(LinearEffect(feature_name, scaled, lambda_p, extrap_val))
        #: </parse_linear>
            
        #: <parse_fourier>
        elif ttype == 'f':
            m = int(params_resolved.get('m', 10))
            s = int(params_resolved.get('s', 1))
            cyclic_raw = params_resolved.get('cyclic', 'False')
            is_cyclic = str(cyclic_raw).strip().lower() in ['true', '1', 't', 'y', 'yes']
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(FourierEffect(feature_name, m, s, lambda_p, is_cyclic, extrap_val))
        #: </parse_fourier>
            
        #: <parse_spline>
        elif ttype == 's':
            k = int(params_resolved.get('k', 10))
            deg = int(params_resolved.get('deg', 3))
            p = int(params_resolved.get('p', 2))
            extrap_val = params_resolved.get('extrapolate', 'linear')
            effects_list.append(SplineEffect(feature_name, k, deg, p, lambda_p, extrap_val))
        #: </parse_spline>
            
        #: <parse_categorical>
        elif ttype == 'c':
            n_cat = params_resolved.get('n_cat')
            if n_cat is None:
                if data_info is not None and feature_name in data_info:
                    n_cat = data_info[feature_name]
                else:
                    raise ValueError(f"Categorical term 'c({feature_name})' requires 'n_cat' or data context to infer it.")
            else:
                n_cat = int(n_cat)
            topo = params_resolved.get('topo', 'nominal')
            p_order = int(params_resolved.get('p_order', 1))
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(CategoricalEffect(feature_name, n_cat, topo, lambda_p, p_order, extrap_val))
        #: </parse_categorical>
            
        #: <parse_chebyshev>
        elif ttype == 'p':
            deg = int(params_resolved.get('deg', 5))
            s = int(params_resolved.get('s', 0))
            extrap_val = params_resolved.get('extrapolate', 'saturation')
            effects_list.append(ChebyshevEffect(feature_name, deg, s, lambda_p, extrap_val))
        #: </parse_chebyshev>
            
        #: <parse_wavelet>
        elif ttype == 'w':
            n_scales = int(params_resolved.get('n_scales', 5))
            n_locs = int(params_resolved.get('n_locations', 20))
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(WaveletEffect(feature_name, n_scales, n_locs, lambda_p, extrap_val))
        #: </parse_wavelet>
            
        #: <parse_neural>
        elif ttype == 'n':
            n_neurons = int(params_resolved.get('n_neurons', 500))
            act = params_resolved.get('act', 'relu')
            seed = int(params_resolved.get('seed', 42))
            n_hidden_layers = int(params_resolved.get('n_hidden_layers', 1))
            others_str = params_resolved.get('others', None)
            additional_features = None
            if others_str:
                additional_features = [s.strip() for s in others_str.split('|') if s.strip()]
            extrap_val = params_resolved.get('extrapolate', 'linear')
            effects_list.append(NeuralEffect(feature_name, n_neurons, act, lambda_p, additional_features, seed, n_hidden_layers, extrap_val))
        #: </parse_neural>
            
        #: <parse_rbf>
        elif ttype == 'rbf':
            n_centers = int(params_resolved.get('n_centers', 50))
            gamma = params_resolved.get('gamma', None)
            if gamma is not None: gamma = float(gamma)
            nu = params_resolved.get('nu', None)
            others_str = params_resolved.get('others', None)
            additional_features = None
            if others_str:
                additional_features = [s.strip() for s in others_str.split('|') if s.strip()]
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(RBFEffect(feature_name, n_centers, gamma, nu, lambda_p, additional_features, extrap_val))
        #: </parse_rbf>

        #: <parse_tree>
        elif ttype == 't':
            n_trees = int(params_resolved.get('n_trees', 50))
            max_depth = int(params_resolved.get('max_depth', 5))
            max_leaves_raw = params_resolved.get('max_leaves', None)
            max_leaves = int(max_leaves_raw) if max_leaves_raw is not None else None
            seed = int(params_resolved.get('seed', 42))
            others_str = params_resolved.get('others', None)
            additional_features = None
            if others_str:
                additional_features = [s.strip() for s in others_str.split('|') if s.strip()]
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(TreeEffect(feature_name, n_trees, max_depth, max_leaves, lambda_p, additional_features, seed, extrap_val))
        #: </parse_tree>

        #: <parse_tensor>
        elif ttype == 'te':
            raw_arguments = list(params.keys())
            
            functional_sub_term_strings = [
                s for s in raw_arguments 
                if re.match(r'^\s*(\w+)\s*\(', s)
            ]

            dummy_formula = "DUMMY ~ " + " + ".join(functional_sub_term_strings)
            
            try:
                _, sub_parsed_terms = parse_formula_to_terms(dummy_formula)
                
                sub_effects = create_effects_from_parsed_terms(
                    sub_parsed_terms, 
                    token_values,
                    default_alpha_p,
                    include_offset=False,
                    data_info=data_info
                )
                
                if len(sub_effects) < 2:
                     raise ValueError(f"Tensor Product requires at least two functional sub-terms. Found {len(sub_effects)}.")
                extrap_val = params_resolved.get('extrapolate', 'continue')
                effects_list.append(TensorProductEffect(sub_effects, lambda_p, extrap_val))

            except ValueError as e:
                raise ValueError(f"Tensor Product failed to parse functional sub-terms {functional_sub_term_strings}: {e}")
        #: </parse_tensor>
            
        #: <parse_physics>
        elif ttype == 'phys':
            basis = params_resolved.get('basis', 'spline')
            n_coeffs = int(params_resolved.get('k', 20) if basis != 'fourier' else params_resolved.get('n_coeffs', 20))
            
            diff_weights = {}
            for pk, pv in params_resolved.items():
                if pk.startswith('D') and pk[1:].isdigit():
                    diff_weights[pk] = float(pv)
            if not diff_weights: diff_weights = {'D2': 1.0}
                
            reserved_keys = ['k', 'n_coeffs', 'ap', 'basis', 'extrapolate']
            basis_kwargs = {k: v for k, v in params_resolved.items() if k not in reserved_keys and not k.startswith('D')}
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(UniversalPhysicsEffect(
                feature_name, basis, n_coeffs, diff_weights, lambda_p, extrap_val, **basis_kwargs
            ))
        #: </parse_physics>

        #: <parse_pid>
        elif ttype == 'pid':
            w = int(params_resolved.get('w', 7))
            d_pen = float(params_resolved.get('d_pen', 10.0))
            extrap_val = params_resolved.get('extrapolate', 'continue')
            effects_list.append(PIDEffect(feature_name, w, lambda_p, d_pen, extrap_val))
        #: </parse_pid>

        #: <parse_linear_tree>
        elif ttype == 'lt':
            slope_feat = params_resolved.pop('slope', feature_name)
            extrap_val = params_resolved.pop('extrapolate', 'linear')
            
            # Guardrail: Force n_trees=1 to prevent overlapping collinearity inside a single feature
            params_resolved['n_trees'] = 1 
            n_trees = 1
            
            max_depth = int(params_resolved.get('max_depth', 5))
            max_leaves_raw = params_resolved.get('max_leaves', None)
            max_leaves = int(max_leaves_raw) if max_leaves_raw is not None else None
            seed = int(params_resolved.get('seed', 42))
            
            others_str = params_resolved.get('others', None)
            additional_features = [s.strip() for s in others_str.split('|') if s.strip()] if others_str else None
            
            # Directly append the composite effect! No sub-effect macro hacks needed.
            effects_list.append(LinearTreeEffect(
                feature_name=feature_name,
                slope_feature=slope_feat,
                n_trees=n_trees,
                max_depth=max_depth,
                max_leaves=max_leaves,
                lambda_p=lambda_p,
                additional_features=additional_features,
                seed=seed,
                extrapolate=extrap_val
            ))
        #: </parse_linear_tree>

        else:
            raise ValueError(f"Unknown effect type identifier: '{ttype}'")
            
    return effects_list
#: </create_effects>

#: <infer_columns>
def _infer_feature_columns(effects_list: List[BaseEffect]) -> List[str]:
    """
    Reconstructs the ordered list of unique feature columns directly from the 
    instantiated effects list. Prevent column exhaustion when multiple effects target the same feature.
    """
    feature_columns = []
    seen = set()
    
    def _add_from_effect(eff):
        if eff.__class__.__name__ == 'OffsetEffect':
            return
        if hasattr(eff, 'effects'):
            for sub in eff.effects:
                _add_from_effect(sub)
        else:
            feats = getattr(eff, 'input_features', [getattr(eff, 'feature_name', None)])
            for f in feats:
                if f and f not in seen:
                    seen.add(f)
                    feature_columns.append(f)
                    
    for effect in effects_list:
        _add_from_effect(effect)
        
    return feature_columns
#: </infer_columns>

#: <build_phi>
def build_phi_from_effects(
    x_data: torch.Tensor, 
    effects_list: List[BaseEffect],
    feature_columns: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Constructs the global Design Matrix by concatenating effect feature maps.
    """
    if feature_columns is None:
        feature_columns = _infer_feature_columns(effects_list)
        
    phi_components = []
    col_idx = 0
    name_to_idx = {name: i for i, name in enumerate(feature_columns)} if feature_columns else None
    
    for effect in effects_list:
        if isinstance(effect, OffsetEffect):
            phi_components.append(effect.transform(x_data))
            
        elif isinstance(effect, (TensorProductEffect, NeuralEffect, RBFEffect, TreeEffect, LinearTreeEffect)):
            
            if isinstance(effect, TensorProductEffect):
                req_features = []
                for e in effect.effects:
                    req_features.extend(getattr(e, 'input_features', [e.feature_name]))
            else:
                req_features = getattr(effect, 'input_features', [effect.feature_name])

            if name_to_idx:
                try:
                    indices = [name_to_idx[name] for name in req_features]
                except KeyError as e:
                     raise ValueError(f"Multivariate sub-feature not found in data: {e}.")
                x_cols = x_data[..., indices]
            else:
                n_inputs = len(req_features)
                if col_idx + n_inputs > x_data.shape[-1]:
                     raise ValueError(f"Not enough columns for multivariate effect at index {col_idx}.")
                x_cols = x_data[..., col_idx : col_idx + n_inputs]
                col_idx += n_inputs
            
            phi_components.append(effect.transform(x_cols))
            
        else:
            if name_to_idx:
                idx = name_to_idx.get(effect.feature_name)
                if idx is None:
                     raise ValueError(f"Feature '{effect.feature_name}' not found in data.")
                x_col = x_data[..., idx]
            else:
                if col_idx >= x_data.shape[-1]:
                     raise ValueError(f"Not enough columns for effect '{effect.feature_name}'.")
                x_col = x_data[..., col_idx]
                col_idx += 1
            
            phi_components.append(effect.transform(x_col))
            
    return torch.cat(phi_components, dim=-1).to(x_data.device)
#: </build_phi>

#: <build_penalty>
def build_penalty_from_effects(effects_list: List[BaseEffect]) -> torch.Tensor:
    """
    Constructs the global block-diagonal Penalty Matrix.

    Aggregates individual penalty matrices into a large sparse block matrix.
    Safely handles both dense and pre-sparsified matrices (like Trees) to prevent OOM.
    """
    
    matrices = [e.build_penalty_matrix() for e in effects_list]
    if not matrices:
        return torch.zeros((0, 0), dtype=torch.get_default_dtype())
        
    total_size = sum(m.shape[0] for m in matrices)
    run_device = matrices[0].device
    
    if total_size > 5000:
        indices_list = []
        values_list = []
        current_idx = 0
        
        for m in matrices:
            k = m.shape[0]
            
            if m.is_sparse:
                m = m.coalesce()
                row_idx = m.indices()[0] + current_idx
                col_idx = m.indices()[1] + current_idx
                indices_list.append(torch.stack([row_idx, col_idx], dim=0))
                values_list.append(m.values())
            else:
                nz = m.nonzero(as_tuple=True)
                if nz[0].numel() > 0:
                    row_idx = nz[0] + current_idx
                    col_idx = nz[1] + current_idx
                    indices_list.append(torch.stack([row_idx, col_idx], dim=0))
                    values_list.append(m[nz])
                
            current_idx += k
            
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list, dim=0)
            P = torch.sparse_coo_tensor(
                indices, values, size=(total_size, total_size), device=run_device
            )
        else:
            P = torch.sparse_coo_tensor(
                size=(total_size, total_size), dtype=torch.get_default_dtype(), device=run_device
            )
        return P
        
    else:
        P = torch.zeros((total_size, total_size), dtype=torch.get_default_dtype(), device=run_device)
        current_idx = 0
        for m in matrices:
            k = m.shape[0]
            if m.is_sparse:
                P[current_idx:current_idx+k, current_idx:current_idx+k] = m.to_dense()
            else:
                P[current_idx:current_idx+k, current_idx:current_idx+k] = m
            current_idx += k
        return P
#: </build_penalty>