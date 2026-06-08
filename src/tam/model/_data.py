"""
Data Processing and Tensorization Module.

This module handles the core logic for transforming Pandas DataFrames into
PyTorch tensors required for model training and prediction. It manages:
1.  Per-group data normalization.
2.  Stacking data into batches (tensors) for standard training.
3.  Creating sliding-window tensors for adaptive (online) learning.
4.  Reassembling tensor predictions back into Pandas DataFrames.
"""

from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE

def _fit_normalization_params(
    data: pd.DataFrame, 
    features: List[str], 
    group_col: str
) -> Tuple[Dict, List]:
    r"""
    Calculates the min/max normalization parameters for features, computed per group.

    Args:
        data: The training DataFrame.
        features: A list of feature column names to normalize.
        group_col: The column name used to group the data.

    Returns:
        A tuple (norm_params, unique_groups):
        - norm_params (Dict): A nested dictionary structured as:
          {group_name: {'min': pd.Series, 'max': pd.Series}}
        - unique_groups (List): A sorted list of all unique group names found.
    """

    grouped = data.groupby(group_col)
    unique_groups = sorted(grouped.groups.keys())
    
    norm_params = {
        group_name: {
            'min': grouped.get_group(group_name)[features].min(axis=0),
            'max': grouped.get_group(group_name)[features].max(axis=0)
        }
        for group_name in unique_groups
    }
        
    return norm_params, unique_groups

#: <normalize>
def normalize(df_to_normalize: pd.DataFrame, params: dict) -> pd.DataFrame:
   
    r"""
    Normalizes a DataFrame to the range [-1, 1] using min/max parameters.
    
    This range is standard for Splines, Chebyshev polynomials, and Neural Networks.
    Effects requiring specific domains (e.g., Fourier on [-pi, pi]) perform 
    their own internal rescaling.

    Args:
        df_to_normalize: The DataFrame (or slice) containing features to normalize.
        params: A dictionary {'min': pd.Series, 'max': pd.Series}.

    Returns:
        The normalized DataFrame."""
   
    amplitude = params['max'] - params['min']
    center = (params['max'] + params['min']) / 2
    
    # Handle constant features (amplitude is 0) to avoid division by zero
    amplitude[amplitude == 0] = 1.0
    
    return (df_to_normalize - center) / (amplitude / 2.0)
#: </normalize>

#: <transform_stacked>
def _transform_data_stacked(
    data: pd.DataFrame,
    features: List[str],
    group_col: str,
    norm_params: Dict,
    unique_groups: List,
    target_col: Optional[str] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Normalizes data per group and stacks it into 3D tensors.

    Used for standard (non-adaptive) batch training.

    Args:
        data: The DataFrame to transform.
        features: List of feature columns.
        group_col: The grouping column name.
        norm_params: Fitted normalization parameters.
        unique_groups: Fitted list of unique group names.
        target_col: Target variable name (optional).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
        - x_stacked: (n_groups, n_samples_per_group, n_features).
        - y_stacked: (n_groups, n_samples_per_group, 1) or None.
    
    Raises:
        ValueError: If `unique_groups` is None.
    """
    formatted_data = []
    
    if unique_groups is None:
        raise ValueError("`unique_groups` cannot be None.")

    for group_name in unique_groups:
        if group_name not in norm_params:
            continue
            
        group_data = data[data[group_col] == group_name].reset_index(drop=True)
        if group_data.empty:
            continue

        params = norm_params[group_name]
        normalized_features = normalize(df_to_normalize=group_data[features], params=params)
        
        x_tensor = torch.tensor(normalized_features.values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
        
        y_tensor = None
        if target_col in data.columns:
            y_vals = group_data[target_col].values
            y_tensor = torch.tensor(y_vals, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
            y_tensor = y_tensor.view(-1, 1)

        formatted_data.append([x_tensor, y_tensor])

    if not formatted_data:
        x_empty = torch.empty(0, 0, len(features), device=TORCH_DEVICE)
        y_empty = torch.empty(0, 0, 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) if target_col else None
        return x_empty, y_empty

    x_stacked = torch.stack([entry[0] for entry in formatted_data])
    y_stacked = torch.stack([entry[1] for entry in formatted_data]) if target_col in data.columns else None

    return x_stacked, y_stacked
#: </transform_stacked>

#: <reassemble>
def _reassemble_predictions(
    original_data: pd.DataFrame,
    predictions_stacked: torch.Tensor,
    group_col: str,
    unique_groups: List,
    target_col: str
) -> pd.DataFrame:
    r"""
    Reassembles stacked tensor predictions back into the original DataFrame structure.

    Args:
        original_data: The source DataFrame.
        predictions_stacked: Tensor of predictions (n_groups, n_samples).
        group_col: Grouping column.
        unique_groups: List of group names corresponding to tensor dimensions.
        target_col: Original target name (used to name the prediction column).

    Returns:
        pd.DataFrame: Original data with a new `Estimated{target_col}` column.
    """
    is_3d_input = predictions_stacked.dim() == 3
    all_predictions_series = []
    
    for i, group_name in enumerate(unique_groups):
        group_indices_full = original_data.index[original_data[group_col] == group_name]
        
        if i >= predictions_stacked.shape[0]:
            continue

        if is_3d_input:
            preds_group = predictions_stacked[i].cpu().numpy().flatten()
        else:
            preds_group = predictions_stacked[i].cpu().numpy()
        
        if len(preds_group) == 0:
            continue
            
        # Align to the end of the group's indices (handling potential truncation)
        group_indices_aligned = group_indices_full[-len(preds_group):]
        
        preds_series = pd.Series(preds_group, index=group_indices_aligned)
        all_predictions_series.append(preds_series)
    
    result_df = original_data.copy()
    
    if not all_predictions_series:
         result_df[f"Estimated{target_col}"] = np.nan
         return result_df

    final_predictions = pd.concat(all_predictions_series)
    result_df[f"Estimated{target_col}"] = final_predictions
    
    return result_df
#: </reassemble>


def _reassemble_decomposed_predictions(
    original_data: pd.DataFrame,
    decomposed_effects: Dict[str, torch.Tensor],
    group_col: str,
    unique_groups: List
) -> pd.DataFrame:
    r"""
    Reassembles decomposed feature effects into the DataFrame.

    Adds `effect_{feature_name}` columns to the data.

    Args:
        original_data: Source DataFrame.
        decomposed_effects: Dictionary {feature_name: effect_tensor}.
        group_col: Grouping column.
        unique_groups: List of group names.

    Returns:
        pd.DataFrame: Data with effect columns added.
    """
    result_df = original_data.copy()

    for feature_name, effect_tensor in decomposed_effects.items():
        col_name = f"effect_{feature_name}"
        result_df[col_name] = np.nan
        
        for i, group_name in enumerate(unique_groups):
            if i >= effect_tensor.shape[0]:
                continue
            
            group_mask = (result_df[group_col] == group_name)
            group_len = group_mask.sum()
            
            if effect_tensor.dim() == 2:
                effect_data = effect_tensor[i,:].cpu().numpy()
            elif effect_tensor.dim() == 3:
                effect_data = effect_tensor[i,:,:].cpu().numpy().flatten()
            else:
                raise ValueError(f"Unrecognized tensor shape: {effect_tensor.shape}")
            
            if group_len == 0 or len(effect_data) == 0:
                continue

            if group_len < len(effect_data):
                effect_data = effect_data[-group_len:]
            
            # Pad with NaNs at the beginning if necessary
            padded_effect_data = np.concatenate([
                np.full(group_len - len(effect_data), np.nan), 
                effect_data
            ])
            
            result_df.loc[group_mask, col_name] = padded_effect_data

    return result_df
    
#: <transform_adaptive>
def _transform_data_adaptive(
    data: pd.DataFrame,
    features: List[str],
    group_col: str,
    norm_params: Dict,
    unique_groups: List,
    target_col: str,
    update_interval_periods: int,
    training_window_periods: int,
    steps_per_period: int,
    horizon_steps: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Prepares data for adaptive learning using vectorized sliding window indexing.

    Returns tensors for (X_train, Y_train, X_predict) for each simulation step.

    Args:
        data: Validation/Test DataFrame.
        features: Feature list.
        group_col: Grouping column.
        norm_params: Normalization parameters.
        unique_groups: Group names.
        target_col: Target column.
        update_interval_periods: Prediction window size.
        training_window_periods: Training history size.
        steps_per_period: Samples per period.
        horizon_steps: Horizon of forecasting

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x_stacked, y_stacked, x_to_predict)
    """
    learning_size_steps = training_window_periods * steps_per_period
    window_size_steps = update_interval_periods * steps_per_period

    all_groups_x_train = []
    all_groups_y_train = []
    all_groups_x_predict = []
    
    if unique_groups is None:
        raise ValueError("`unique_groups` cannot be None.")
    
    prep_device = 'cpu'

    for group_name in unique_groups:
        if group_name not in norm_params:
            continue
            
        data_group = data[data[group_col] == group_name].reset_index(drop=True)
        
        params = norm_params.get(group_name)
        if params is None: continue
        
        total_available_steps = len(data_group)
        required_history = learning_size_steps + (horizon_steps - 1)
        if total_available_steps <= required_history:
            continue

        # Normalize group data
        data_group[features] = normalize(df_to_normalize=data_group[features], params=params)
        
        x_group = torch.tensor(data_group[features].values, dtype=torch.float32, device=prep_device)
        y_group = torch.tensor(data_group[target_col].values, dtype=torch.float32, device=prep_device).view(-1, 1)

        #  Calculate valid start indices (reverse chronological)
        start_indices_list = []
        first_predict_start = total_available_steps - (total_available_steps - learning_size_steps) % window_size_steps
        if first_predict_start == total_available_steps and total_available_steps > learning_size_steps:
             first_predict_start -= window_size_steps
        
        current_predict_start = first_predict_start
        while current_predict_start >= learning_size_steps:
            if current_predict_start + window_size_steps <= total_available_steps:
                start_indices_list.append(current_predict_start)
            current_predict_start -= window_size_steps
        
        if not start_indices_list:
            continue
            
        start_indices_list.reverse()
        start_indices = torch.tensor(start_indices_list, device=prep_device, dtype=torch.long)

        #  Vectorized Window Indexing
        train_end_offset = -(horizon_steps - 1) if horizon_steps > 1 else 0
        train_start_offset = train_end_offset - learning_size_steps       
        train_offsets = torch.arange(train_start_offset, train_end_offset, device=prep_device)
        predict_offsets = torch.arange(0, window_size_steps, device=prep_device)

        train_indices = start_indices.view(-1, 1) + train_offsets
        predict_indices = start_indices.view(-1, 1) + predict_offsets

        #  Gather
        group_x_train = x_group[train_indices]
        group_y_train = y_group[train_indices]
        group_x_predict = x_group[predict_indices]
        
        all_groups_x_train.append(group_x_train)
        all_groups_y_train.append(group_y_train)
        all_groups_x_predict.append(group_x_predict)

    if not all_groups_x_train:
        raise ValueError("No simulation data could be generated. Check dataset length/window sizes.")

    x_stacked = torch.stack(all_groups_x_train).to(TORCH_DEVICE)
    y_stacked = torch.stack(all_groups_y_train).to(TORCH_DEVICE)
    x_to_predict = torch.stack(all_groups_x_predict).to(TORCH_DEVICE)

    return x_stacked, y_stacked, x_to_predict
#: </transform_adaptive>