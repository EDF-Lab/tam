# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-FileContributor: Éloi Bedek
# SPDX-License-Identifier: LGPL-3.0-or-later

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
from .spectrum._factory import categorical_range

def _fit_normalization_params(
    data: pd.DataFrame, 
    features: List[str], 
    group_col: str,
    categorical_levels: Optional[Dict[str, int]] = None,
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[Dict, List]:
    r"""
    Calculates the min/max normalization parameters for features, computed per group.

    Args:
        data: The training DataFrame.
        features: A list of feature column names to normalize.
        group_col: The column name used to group the data.
        categorical_levels: ``{feature: n_cat}`` of the categorical features: their range is the full level range
            (``categorical_range``), whatever levels the training rows hold.
        fixed_ranges: ``{feature: (low, high)}`` of the features read on a fixed domain (a cyclic Fourier term reads [-1, 1]):
            that range is used whatever values the training rows hold.

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
    for group_name, params in norm_params.items():
        for feature, n_cat in (categorical_levels or {}).items():
            if feature in params['min'].index:
                column = grouped.get_group(group_name)[feature]
                params['min'][feature], params['max'][feature] = categorical_range(
                    n_cat, params['min'][feature], params['max'][feature], bool((column.dropna() == np.round(column.dropna())).all()))
        for feature, (low, high) in (fixed_ranges or {}).items():
            if feature in params['min'].index:
                params['min'][feature], params['max'][feature] = float(low), float(high)

    return norm_params, unique_groups

def _resolve_fixed_ranges(data: pd.DataFrame, ranges: Optional[Dict]) -> Dict[str, Tuple[float, float]]:
    r"""
    Turns the ``"auto"`` periods of ``fixed_ranges`` into ``(low, high)`` read from the training rows; given ranges pass through.

    ``"auto"`` is (first value, last value + one step), the step being the smallest gap between the distinct values (rounded
    to 10 decimals): hours 0..23 give (0, 24), months 1..12 give (1, 13), a time of year sampled once a day gives (0, 1).
    It is computed once on all the rows given (all groups, the whole series), so every group and every window shares it.
    A feature with a single distinct value keeps the min/max rule.

    Args:
        data: The training rows (or the whole series of an adaptive simulation).
        ranges: ``{feature: (low, high) or "auto"}``.

    Returns:
        ``{feature: (low, high)}``.
    """
    resolved: Dict[str, Tuple[float, float]] = {}
    for feature, value in (ranges or {}).items():
        if value != "auto":
            resolved[feature] = (float(value[0]), float(value[1]))
            continue
        if feature not in data.columns:
            continue
        distinct = np.unique(np.round(data[feature].dropna().to_numpy(dtype=float), 10))
        if distinct.size < 2:
            continue
        resolved[feature] = (float(distinct[0]), float(distinct[-1] + np.min(np.diff(distinct))))
    return resolved

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
    date_col: Optional[str] = None,
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
            
        group_data = data[data[group_col] == group_name]
        if date_col is not None:
            group_data = group_data.sort_values(date_col)
        group_data = group_data.reset_index(drop=True)
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

def _groups_in_data(data: pd.DataFrame, group_col: str, unique_groups: List, norm_params: dict) -> List:
    r"""The fitted groups that `data` holds rows for, in fitted order: the groups `_transform_data_stacked` stacks."""
    present = set(data[group_col].dropna().unique())
    return [g for g in unique_groups if g in norm_params and g in present]


def _coefficients_of_groups(coefficients: torch.Tensor, groups_stacked: List, unique_groups: List) -> torch.Tensor:
    r"""Per-group coefficients of the groups actually stacked (all of them when the frame holds every fitted group)."""
    if list(groups_stacked) == list(unique_groups):
        return coefficients
    positions = [list(unique_groups).index(g) for g in groups_stacked]
    return coefficients[torch.tensor(positions, device=coefficients.device)]


def _check_known_groups(data: pd.DataFrame, group_col: str, known_groups: List) -> None:
    r"""Raises a ValueError naming the groups of `data` that the model never saw in training."""
    if known_groups is None:
        return
    known = set(known_groups)
    unseen = [g for g in data[group_col].dropna().unique() if g not in known]
    if unseen:
        raise ValueError(
            f"TAM [Data Error]: group(s) {unseen[:10]} of column '{group_col}' were not seen in training; "
            f"the model knows {list(known_groups)[:20]}."
        )


def _date_keys(data: pd.DataFrame, date_col: Optional[str]) -> Optional[np.ndarray]:
    r"""
    Sortable keys of the date column, read once per placement of the predictions (not once per group).

    ``Series.to_numpy()`` on a timezone-aware date column builds a Python ``Timestamp`` per row, every call: with 48 groups that
    made ``predict`` several times slower. The integer view of the datetimes sorts the same way (a missing date goes last, as
    for ``datetime64``) and costs nothing.
    """
    if date_col is None or date_col not in data.columns:
        return None
    column = data[date_col]
    as_integers = getattr(column.array, "asi8", None)
    if as_integers is None:
        return column.to_numpy()
    return np.where(as_integers == np.iinfo(np.int64).min, np.iinfo(np.int64).max, as_integers)


def _group_positions(data: pd.DataFrame, group_col: str, group_name, date_col: Optional[str] = None, date_keys: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    Row positions (0 .. len(data) - 1) of one group, in date order (frame order when there is no date, ties keep frame order).

    Predictions are placed by position, never by index label: a frame built by ``pd.concat`` without ``ignore_index`` repeats
    labels, and the rows of the groups can be interleaved or in any order. ``date_keys`` is ``_date_keys(data, date_col)`` when
    the caller places several groups.
    """
    positions = np.flatnonzero((data[group_col] == group_name).to_numpy())
    if date_keys is None:
        date_keys = _date_keys(data, date_col)
    if date_keys is not None and len(positions) > 1:
        positions = positions[np.argsort(date_keys[positions], kind="stable")]
    return positions


#: <reassemble>
def _reassemble_predictions(
    original_data: pd.DataFrame,
    predictions_stacked: torch.Tensor,
    group_col: str,
    unique_groups: List,
    target_col: str,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    r"""
    Reassembles stacked tensor predictions back into the original DataFrame structure.

    Every prediction goes on the row it was computed for, whatever the index of `original_data` (duplicate labels, interleaved
    groups, any row order): rows are located by position, in date order inside each group, and the index is returned unchanged.
    A group with fewer predictions than rows (a warm-up) is aligned to the end of the group; its first rows stay NaN.

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
    estimate = np.full(len(original_data), np.nan)
    date_keys = _date_keys(original_data, date_col)

    for i, group_name in enumerate(unique_groups):
        if i >= predictions_stacked.shape[0]:
            continue

        if is_3d_input:
            preds_group = predictions_stacked[i].cpu().numpy().flatten()
        else:
            preds_group = predictions_stacked[i].cpu().numpy()

        if len(preds_group) == 0:
            continue

        positions = _group_positions(original_data, group_col, group_name, date_col, date_keys)
        # Align to the end of the group's rows (handling potential truncation)
        estimate[positions[-len(preds_group):]] = preds_group

    result_df = original_data.copy()
    result_df[f"Estimated{target_col}"] = estimate
    return result_df
#: </reassemble>


def _reassemble_decomposed_predictions(
    original_data: pd.DataFrame,
    decomposed_effects: Dict[str, torch.Tensor],
    group_col: str,
    unique_groups: List,
    date_col: Optional[str] = None
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
    date_keys = _date_keys(original_data, date_col)

    for feature_name, effect_tensor in decomposed_effects.items():
        col_name = f"effect_{feature_name}"
        column = np.full(len(original_data), np.nan)

        for i, group_name in enumerate(unique_groups):
            if i >= effect_tensor.shape[0]:
                continue

            positions = _group_positions(original_data, group_col, group_name, date_col, date_keys)
            group_len = len(positions)

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
            column[positions] = np.concatenate([
                np.full(group_len - len(effect_data), np.nan),
                effect_data
            ])

        result_df[col_name] = column

    return result_df
    
def _window_starts(n_rows: int, training_steps: int, window_steps: int, horizon_steps: int = 1) -> List[int]:
    r"""
    Positions of the first forecast row of every window, anchored on the start of the data.

    The first window trains on the first ``training_steps`` rows and forecasts right after them, ``horizon_steps - 1`` rows later
    (the target of a row is only known ``horizon_steps`` rows after it); the next windows follow every ``window_steps`` rows, and
    the last one is cut at the end of the data.
    """
    return list(range(training_steps + horizon_steps - 1, n_rows, window_steps))


#: <transform_adaptive>
class _AdaptiveWindows:
    r"""Where the windows of an adaptive simulation sit in the data, and the normalisation each one trained with."""

    def __init__(self, positions, starts, n_valid, norm_min, norm_max, y_min, y_max):
        self.positions = positions      # per group: row positions of the group in `data`, in date order
        self.starts = starts            # per group: first forecast position of each window (the last one may be the data end)
        self.n_valid = n_valid          # per group: forecast rows of each window that exist in the data
        self.norm_min = norm_min        # (n_groups, n_windows, n_features): minimum of the training rows of the window
        self.norm_max = norm_max        # (n_groups, n_windows, n_features)
        self.y_min = y_min              # (n_groups, n_windows): target range of the training rows of the window
        self.y_max = y_max

    @property
    def final_index(self) -> List[int]:
        r"""Per group, the window an operational refit would hold after the last row (-1 if the group is too short)."""
        return [len(st) - 1 for st in self.starts]


def _transform_data_adaptive(
    data: pd.DataFrame,
    features: List[str],
    group_col: str,
    unique_groups: List,
    target_col: str,
    update_interval_periods: int,
    training_window_periods: int,
    steps_per_period: int,
    horizon_steps: int = 1,
    date_col: Optional[str] = None,
    categorical_levels: Optional[Dict[str, int]] = None,
    fixed_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, _AdaptiveWindows]:
    r"""
    Cuts the data into the windows of a rolling refit, and normalises each window on its own training rows.

    A window trains on ``training_window_periods`` periods and forecasts the ``update_interval_periods`` that follow; windows are
    anchored on the start of the data (see ``_window_starts``). Every (group, window) is normalised to [-1, 1] with the minimum and
    maximum of its own training rows, applied to its training and its forecast rows: this is what ``StaticTAM.fit`` then
    ``predict`` does on that window. A group that does not hold a full window gets an all-zero placeholder and no forecast.

    Args:
        data: DataFrame holding only real rows (no balancing fill).
        features: Feature list.
        group_col: Grouping column.
        unique_groups: Group names, in the order of the output tensors.
        target_col: Target column.
        update_interval_periods: Forecast length, in periods.
        training_window_periods: Training length, in periods.
        steps_per_period: Rows of one group in a period.
        horizon_steps: Horizon of forecasting.
        categorical_levels: ``{feature: n_cat}`` of the categorical features: normalised on their full level range.
        fixed_ranges: ``{feature: (low, high)}`` of the features read on a fixed domain: every window uses that range.

    Returns:
        (x_stacked, y_stacked, x_to_predict, windows): (n_groups, n_windows, L, F), (n_groups, n_windows, L, 1),
        (n_groups, n_windows, W, F) and the layout of the windows.
    """
    if unique_groups is None:
        raise ValueError("`unique_groups` cannot be None.")
    learning_size_steps = training_window_periods * steps_per_period
    window_size_steps = update_interval_periods * steps_per_period
    dtype = torch.get_default_dtype()
    n_features = len(features)
    categorical = categorical_levels or {}
    fixed = fixed_ranges or {}

    x_all = data[features].to_numpy(dtype=np.float64)
    y_all = data[target_col].to_numpy(dtype=np.float64)
    group_values = data[group_col].to_numpy()
    date_values = data[date_col].to_numpy() if date_col is not None else None

    built = []
    positions, starts_all, n_valid_all = [], [], []
    for group_name in unique_groups:
        pos = np.flatnonzero(group_values == group_name)
        if date_values is not None:
            pos = pos[np.argsort(date_values[pos], kind="stable")]
        n = len(pos)
        starts = _window_starts(n, learning_size_steps, window_size_steps, horizon_steps)
        if n >= learning_size_steps + horizon_steps - 1 and (n - (learning_size_steps + horizon_steps - 1)) % window_size_steps == 0:
            starts.append(n)        # the window an operational refit holds after the last row; it has no forecast row yet
        positions.append(pos)
        starts_all.append(starts)
        n_valid_all.append([int(min(window_size_steps, max(n - s, 0))) for s in starts])
        if not starts:
            built.append(None)
            continue
        s = torch.tensor(starts, dtype=torch.long)
        x_g = torch.tensor(x_all[pos], dtype=torch.float64).view(n, n_features)
        y_g = torch.tensor(y_all[pos], dtype=torch.float64).view(n, 1)
        train_idx = s.view(-1, 1) + torch.arange(-(horizon_steps - 1) - learning_size_steps, -(horizon_steps - 1) if horizon_steps > 1 else 0)
        predict_idx = (s.view(-1, 1) + torch.arange(window_size_steps)).clamp(max=n - 1)
        x_train = x_g[train_idx]                                                  # (n_windows, L, F)
        low = x_train.amin(dim=1, keepdim=True)
        high = x_train.amax(dim=1, keepdim=True)
        for j, name in enumerate(features):
            if name in categorical:
                n_cat = categorical[name]
                integer = (x_train[..., j] == x_train[..., j].round()).all(dim=1, keepdim=True)
                start = torch.where(high[..., j] <= n_cat - 1, torch.zeros_like(low[..., j]), low[..., j])
                low[..., j] = torch.where(integer, start, low[..., j])
                high[..., j] = torch.where(integer, start + (n_cat - 1), high[..., j])
            elif name in fixed:
                low[..., j], high[..., j] = fixed[name]
        amplitude = high - low
        amplitude = torch.where(amplitude == 0, torch.ones_like(amplitude), amplitude)
        center = (high + low) / 2
        y_train = y_g[train_idx]
        built.append((
            ((x_train - center) / (amplitude / 2.0)).to(dtype),
            y_train.to(dtype),
            ((x_g[predict_idx] - center) / (amplitude / 2.0)).to(dtype),
            low.squeeze(1), high.squeeze(1), y_train.amin(dim=(1, 2)), y_train.amax(dim=(1, 2)),
        ))

    if all(b is None for b in built):
        raise ValueError("No simulation data could be generated. Check dataset length/window sizes.")

    n_windows = max(len(st) for st in starts_all)
    placeholders = (
        torch.zeros(n_windows, learning_size_steps, n_features, dtype=dtype), torch.zeros(n_windows, learning_size_steps, 1, dtype=dtype),
        torch.zeros(n_windows, window_size_steps, n_features, dtype=dtype), torch.zeros(n_windows, n_features, dtype=torch.float64),
        torch.ones(n_windows, n_features, dtype=torch.float64), torch.zeros(n_windows, dtype=torch.float64), torch.ones(n_windows, dtype=torch.float64),
    )
    columns = [[] for _ in range(7)]
    for entry in built:
        for k in range(7):
            if entry is None:
                columns[k].append(placeholders[k])
                continue
            tensor = entry[k]
            if tensor.shape[0] < n_windows:         # a shorter group repeats its last window; the repeats are never read
                tensor = torch.cat([tensor, tensor[-1:].expand(n_windows - tensor.shape[0], *tensor.shape[1:])], dim=0)
            columns[k].append(tensor)
    stacked = [torch.stack(c) for c in columns]
    windows = _AdaptiveWindows(positions, starts_all, n_valid_all, stacked[3], stacked[4], stacked[5], stacked[6])
    return stacked[0].to(TORCH_DEVICE), stacked[1].to(TORCH_DEVICE), stacked[2].to(TORCH_DEVICE), windows
#: </transform_adaptive>