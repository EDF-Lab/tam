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
import warnings
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from tam.common.utils import TORCH_DEVICE
from .spectrum._factory import categorical_range


def _as_pandas_if_polars(data):
    r"""Converts duck-typed columnar frames (e.g., Polars DataFrames) to Pandas zero-copy where possible."""
    if not isinstance(data, pd.DataFrame) and hasattr(data, "to_pandas"):
        return data.to_pandas()
    return data


def _extract_feature_matrix(data: pd.DataFrame, features: List[str]) -> np.ndarray:
    r"""Extracts feature columns as a 2D float64 NumPy matrix without constructing intermediate DataFrames."""
    n_rows = len(data)
    if not features:
        return np.empty((n_rows, 0), dtype=np.float64)
    cols = []
    for f in features:
        col = data[f]
        if isinstance(col.dtype, pd.api.extensions.ExtensionDtype):
            cols.append(col.to_numpy(dtype=np.float64, na_value=np.nan))
        else:
            try:
                cols.append(col.to_numpy(dtype=np.float64, copy=False))
            except (TypeError, ValueError):
                cols.append(col.to_numpy(dtype=np.float64, na_value=np.nan))
    if len(cols) == 1:
        return cols[0].reshape(n_rows, 1)
    return np.column_stack(cols)


def _is_sorted_nondecreasing(arr: np.ndarray) -> bool:
    r"""Returns True if a 1D array is already non-decreasing, avoiding O(N log N) sorts on ordered time series."""
    if len(arr) <= 1:
        return True
    try:
        return bool(np.all(arr[:-1] <= arr[1:]))
    except TypeError:
        return False


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
    data = _as_pandas_if_polars(data)
    group_arr = data[group_col].to_numpy()
    feat_mat = _extract_feature_matrix(data, features)
    feature_idx = {f: idx for idx, f in enumerate(features)}

    if len(group_arr) > 0 and not pd.isna(group_arr[0]) and bool((group_arr == group_arr[0]).all()):
        unique_groups = [group_arr[0]]
        if features:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mins = np.nanmin(feat_mat, axis=0)
                maxs = np.nanmax(feat_mat, axis=0)
            norm_params = {
                group_arr[0]: {
                    'min': pd.Series(mins, index=features, dtype=np.float64),
                    'max': pd.Series(maxs, index=features, dtype=np.float64),
                }
            }
        else:
            empty_s = pd.Series(dtype=np.float64)
            norm_params = {group_arr[0]: {'min': empty_s, 'max': empty_s.copy()}}
        group_rows = {group_arr[0]: None}
    else:
        grouped_indices = data.groupby(group_col, sort=True).indices
        unique_groups = sorted(grouped_indices.keys())
        norm_params = {}
        group_rows = grouped_indices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for group_name in unique_groups:
                if not features:
                    empty_s = pd.Series(dtype=np.float64)
                    norm_params[group_name] = {'min': empty_s, 'max': empty_s.copy()}
                    continue
                sub_mat = feat_mat[grouped_indices[group_name]]
                norm_params[group_name] = {
                    'min': pd.Series(np.nanmin(sub_mat, axis=0), index=features, dtype=np.float64),
                    'max': pd.Series(np.nanmax(sub_mat, axis=0), index=features, dtype=np.float64),
                }

    for group_name, params in norm_params.items():
        pos = group_rows[group_name]
        for feature, n_cat in (categorical_levels or {}).items():
            if feature in feature_idx:
                col_vals = feat_mat[:, feature_idx[feature]] if pos is None else feat_mat[pos, feature_idx[feature]]
                valid = col_vals[~np.isnan(col_vals)]
                is_int = bool(np.all(valid == np.round(valid))) if len(valid) > 0 else True
                params['min'][feature], params['max'][feature] = categorical_range(
                    n_cat, params['min'][feature], params['max'][feature], is_int
                )
        for feature, (low, high) in (fixed_ranges or {}).items():
            if feature in feature_idx:
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
    data = _as_pandas_if_polars(data)
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
    if unique_groups is None:
        raise ValueError("`unique_groups` cannot be None.")

    data = _as_pandas_if_polars(data)
    has_target = target_col is not None and target_col in data.columns
    feat_mat = _extract_feature_matrix(data, features)
    if has_target:
        y_series = data[target_col]
        if isinstance(y_series.dtype, pd.api.extensions.ExtensionDtype):
            y_arr = y_series.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            try:
                y_arr = y_series.to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError):
                y_arr = y_series.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        y_arr = None

    date_keys = _date_keys(data, date_col)
    group_arr = data[group_col].to_numpy()
    dtype = torch.get_default_dtype()

    if len(unique_groups) == 1:
        group_name = unique_groups[0]
        if group_name in norm_params and len(data) > 0:
            if not pd.isna(group_arr[0]) and bool((group_arr == group_name).all()):
                pos = None
                if date_keys is not None and len(date_keys) > 1 and not _is_sorted_nondecreasing(date_keys):
                    pos = np.argsort(date_keys, kind="stable")
            else:
                pos = np.flatnonzero(group_arr == group_name)
                if len(pos) > 1 and date_keys is not None:
                    dk_g = date_keys[pos]
                    if not _is_sorted_nondecreasing(dk_g):
                        pos = pos[np.argsort(dk_g, kind="stable")]
                if len(pos) == 0:
                    pos = np.empty(0, dtype=np.intp)
            if pos is None or len(pos) > 0:
                params = norm_params[group_name]
                min_vals = np.asarray([params['min'][f] for f in features], dtype=np.float64) if features else np.empty(0, dtype=np.float64)
                max_vals = np.asarray([params['max'][f] for f in features], dtype=np.float64) if features else np.empty(0, dtype=np.float64)
                amplitude = max_vals - min_vals
                amplitude[amplitude == 0] = 1.0
                center = (max_vals + min_vals) / 2.0
                x_slice = feat_mat if pos is None else feat_mat[pos]
                norm_x = (x_slice - center) / (amplitude / 2.0)
                x_stacked = torch.tensor(np.ascontiguousarray(norm_x), dtype=dtype, device=TORCH_DEVICE).unsqueeze(0)
                y_stacked = None
                if has_target:
                    y_slice = y_arr if pos is None else y_arr[pos]
                    y_stacked = torch.tensor(np.ascontiguousarray(y_slice), dtype=dtype, device=TORCH_DEVICE).view(1, -1, 1)
                return x_stacked, y_stacked

        x_empty = torch.empty(0, 0, len(features), device=TORCH_DEVICE)
        y_empty = torch.empty(0, 0, 1, device=TORCH_DEVICE, dtype=dtype) if target_col else None
        return x_empty, y_empty

    grouped_indices = data.groupby(group_col, sort=False).indices
    x_list = []
    y_list = []
    for group_name in unique_groups:
        if group_name not in norm_params:
            continue
        pos = grouped_indices.get(group_name)
        if pos is None or len(pos) == 0:
            continue
        if date_keys is not None and len(pos) > 1:
            dk_g = date_keys[pos]
            if not _is_sorted_nondecreasing(dk_g):
                pos = pos[np.argsort(dk_g, kind="stable")]
        params = norm_params[group_name]
        min_vals = np.asarray([params['min'][f] for f in features], dtype=np.float64) if features else np.empty(0, dtype=np.float64)
        max_vals = np.asarray([params['max'][f] for f in features], dtype=np.float64) if features else np.empty(0, dtype=np.float64)
        amplitude = max_vals - min_vals
        amplitude[amplitude == 0] = 1.0
        center = (max_vals + min_vals) / 2.0
        norm_x = (feat_mat[pos] - center) / (amplitude / 2.0)
        x_list.append(torch.tensor(np.ascontiguousarray(norm_x), dtype=dtype, device=TORCH_DEVICE))
        if has_target:
            y_list.append(torch.tensor(np.ascontiguousarray(y_arr[pos]), dtype=dtype, device=TORCH_DEVICE).view(-1, 1))

    if not x_list:
        x_empty = torch.empty(0, 0, len(features), device=TORCH_DEVICE)
        y_empty = torch.empty(0, 0, 1, device=TORCH_DEVICE, dtype=dtype) if target_col else None
        return x_empty, y_empty

    x_stacked = torch.stack(x_list)
    y_stacked = torch.stack(y_list) if has_target else None

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
        dk_pos = date_keys[positions]
        if not _is_sorted_nondecreasing(dk_pos):
            positions = positions[np.argsort(dk_pos, kind="stable")]
    return positions


def _all_group_positions(
    data: pd.DataFrame,
    group_col: str,
    unique_groups: List,
    date_col: Optional[str] = None,
    date_keys: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    r"""Computes row position arrays for all groups in a single O(N) pass."""
    if not unique_groups:
        return []
    if date_keys is None:
        date_keys = _date_keys(data, date_col)
    if len(unique_groups) == 1:
        g0 = unique_groups[0]
        group_arr = data[group_col].to_numpy()
        if len(group_arr) > 0 and not pd.isna(group_arr[0]) and bool((group_arr == g0).all()):
            pos = np.arange(len(data), dtype=np.intp)
            if date_keys is not None and len(pos) > 1 and not _is_sorted_nondecreasing(date_keys):
                pos = np.argsort(date_keys, kind="stable")
            return [pos]
        return [_group_positions(data, group_col, g0, date_col, date_keys)]

    grouped_indices = data.groupby(group_col, sort=False).indices
    positions_list = []
    for g in unique_groups:
        pos = grouped_indices.get(g)
        if pos is None:
            positions_list.append(np.empty(0, dtype=np.intp))
            continue
        if date_keys is not None and len(pos) > 1:
            dk_pos = date_keys[pos]
            if not _is_sorted_nondecreasing(dk_pos):
                pos = pos[np.argsort(dk_pos, kind="stable")]
        positions_list.append(pos)
    return positions_list


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
    original_data = _as_pandas_if_polars(original_data)
    is_3d_input = predictions_stacked.dim() == 3
    estimate = np.full(len(original_data), np.nan)
    date_keys = _date_keys(original_data, date_col)
    active_groups = unique_groups[:predictions_stacked.shape[0]]
    positions_list = _all_group_positions(original_data, group_col, active_groups, date_col, date_keys)

    for i, positions in enumerate(positions_list):
        if is_3d_input:
            preds_group = predictions_stacked[i].cpu().numpy().flatten()
        else:
            preds_group = predictions_stacked[i].cpu().numpy()

        if len(preds_group) == 0 or len(positions) == 0:
            continue

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
    original_data = _as_pandas_if_polars(original_data)
    result_df = original_data.copy()
    date_keys = _date_keys(original_data, date_col)
    positions_list = _all_group_positions(original_data, group_col, unique_groups, date_col, date_keys)

    for feature_name, effect_tensor in decomposed_effects.items():
        col_name = f"effect_{feature_name}"
        column = np.full(len(original_data), np.nan)

        for i, positions in enumerate(positions_list):
            if i >= effect_tensor.shape[0]:
                continue

            group_len = len(positions)

            if effect_tensor.dim() == 2:
                effect_data = effect_tensor[i, :].cpu().numpy()
            elif effect_tensor.dim() == 3:
                effect_data = effect_tensor[i, :, :].cpu().numpy().flatten()
            else:
                raise ValueError(f"Unrecognized tensor shape: {effect_tensor.shape}")

            if group_len == 0 or len(effect_data) == 0:
                continue

            if group_len < len(effect_data):
                effect_data = effect_data[-group_len:]

            # Pad with NaNs at the beginning if necessary
            if group_len > len(effect_data):
                column[positions] = np.concatenate([
                    np.full(group_len - len(effect_data), np.nan),
                    effect_data
                ])
            else:
                column[positions] = effect_data

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

    def __init__(self, positions, starts, n_valid, norm_min, norm_max, y_min, y_max, raw_x_groups=None, raw_y_groups=None):
        self.positions = positions      # per group: row positions of the group in `data`, in date order
        self.starts = starts            # per group: first forecast position of each window (the last one may be the data end)
        self.n_valid = n_valid          # per group: forecast rows of each window that exist in the data
        self.norm_min = norm_min        # (n_groups, n_windows, n_features): minimum of the training rows of the window
        self.norm_max = norm_max        # (n_groups, n_windows, n_features)
        self.y_min = y_min              # (n_groups, n_windows): target range of the training rows of the window
        self.y_max = y_max
        self.raw_x_groups = raw_x_groups  # per group (n_rows_g, n_features) float64 unnormalized features in date order
        self.raw_y_groups = raw_y_groups  # per group (n_rows_g, 1) float64 target in date order

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
    data = _as_pandas_if_polars(data)
    learning_size_steps = training_window_periods * steps_per_period
    window_size_steps = update_interval_periods * steps_per_period
    dtype = torch.get_default_dtype()
    n_features = len(features)
    categorical = categorical_levels or {}
    fixed = fixed_ranges or {}

    x_all = _extract_feature_matrix(data, features)
    y_all = data[target_col].to_numpy(dtype=np.float64)
    all_pos = _all_group_positions(data, group_col, unique_groups, date_col=date_col)
    n_groups = len(all_pos)

    positions, starts_all, n_valid_all = [], [], []
    group_lengths = []
    for pos in all_pos:
        n = len(pos)
        starts = _window_starts(n, learning_size_steps, window_size_steps, horizon_steps)
        if n >= learning_size_steps + horizon_steps - 1 and (n - (learning_size_steps + horizon_steps - 1)) % window_size_steps == 0:
            starts.append(n)        # the window an operational refit holds after the last row; it has no forecast row yet
        positions.append(pos)
        starts_all.append(starts)
        n_valid_all.append([int(min(window_size_steps, max(n - s, 0))) for s in starts])
        group_lengths.append(n)

    if all(len(st) == 0 for st in starts_all):
        raise ValueError("No simulation data could be generated. Check dataset length/window sizes.")

    n_windows = max(len(st) for st in starts_all)
    out_x_train = torch.zeros(n_groups, n_windows, learning_size_steps, n_features, dtype=dtype)
    out_y_train = torch.zeros(n_groups, n_windows, learning_size_steps, 1, dtype=dtype)
    out_x_pred = torch.zeros(n_groups, n_windows, window_size_steps, n_features, dtype=dtype)
    out_low = torch.zeros(n_groups, n_windows, n_features, dtype=torch.float64)
    out_high = torch.ones(n_groups, n_windows, n_features, dtype=torch.float64)
    out_ymin = torch.zeros(n_groups, n_windows, dtype=torch.float64)
    out_ymax = torch.ones(n_groups, n_windows, dtype=torch.float64)

    raw_x_groups, raw_y_groups = [], []
    base_train_off = -(horizon_steps - 1) - learning_size_steps
    train_offsets = torch.arange(base_train_off, -(horizon_steps - 1) if horizon_steps > 1 else 0)
    pred_offsets = torch.arange(window_size_steps)
    cat_cols = [(j, name, categorical[name]) for j, name in enumerate(features) if name in categorical]
    fixed_cols = [(j, fixed[name][0], fixed[name][1]) for j, name in enumerate(features) if name in fixed and name not in categorical]
    free_mask = [i for i, name in enumerate(features) if name not in categorical and name not in fixed]
    origin = -(learning_size_steps // 2)

    for g, (pos, starts, n) in enumerate(zip(all_pos, starts_all, group_lengths)):
        if not starts:
            raw_x_groups.append(None)
            raw_y_groups.append(None)
            continue

        w_g = len(starts)
        x_np = np.ascontiguousarray(x_all[pos])
        y_np = np.ascontiguousarray(y_all[pos]).reshape(n, 1)
        x_g = torch.from_numpy(x_np).view(n, n_features)
        y_g = torch.from_numpy(y_np).view(n, 1)
        raw_x_groups.append(x_g)
        raw_y_groups.append(y_g)

        t0_first = starts[0] + base_train_off
        regular_starts = (
            w_g == 1 or (starts[-1] - starts[0] == (w_g - 1) * window_size_steps)
        )
        if regular_starts and t0_first >= 0 and t0_first + (w_g - 1) * window_size_steps + learning_size_steps <= n:
            end_u = t0_first + (w_g - 1) * window_size_steps + learning_size_steps
            x_train = x_g[t0_first:end_u].unfold(0, learning_size_steps, window_size_steps).transpose(1, 2)
            y_train = y_g[t0_first:end_u].unfold(0, learning_size_steps, window_size_steps).transpose(1, 2)
        else:
            s_tensor = torch.tensor(starts, dtype=torch.long)
            train_idx = s_tensor.view(-1, 1) + train_offsets
            x_train = x_g[train_idx]
            y_train = y_g[train_idx]

        s_tensor = torch.tensor(starts, dtype=torch.long)
        predict_idx = (s_tensor.view(-1, 1) + pred_offsets).clamp(max=n - 1)

        t0_np = np.asarray(starts, dtype=np.intp) + base_train_off
        use_fast_filter = (
            w_g >= 16
            and learning_size_steps >= 16
            and int(t0_np.min()) >= 0
            and int(t0_np.max()) + learning_size_steps <= n
            and not bool(np.isnan(x_np).any())
            and not bool(np.isnan(y_np).any())
        )
        if use_fast_filter:
            # Sliding min/max via van Herk / SciPy filter1d in O(n) independent of L.
            low_mat = torch.from_numpy(minimum_filter1d(x_np, size=learning_size_steps, axis=0, origin=origin, mode='nearest')[t0_np])
            high_mat = torch.from_numpy(maximum_filter1d(x_np, size=learning_size_steps, axis=0, origin=origin, mode='nearest')[t0_np])
            if cat_cols:
                low_mat = low_mat.clone()
                high_mat = high_mat.clone()
            ymin_vec = torch.from_numpy(minimum_filter1d(y_np[:, 0], size=learning_size_steps, axis=0, origin=origin, mode='nearest')[t0_np])
            ymax_vec = torch.from_numpy(maximum_filter1d(y_np[:, 0], size=learning_size_steps, axis=0, origin=origin, mode='nearest')[t0_np])
            low = low_mat.unsqueeze(1)
            high = high_mat.unsqueeze(1)
        else:
            low = x_train.amin(dim=1, keepdim=True)
            high = x_train.amax(dim=1, keepdim=True)
            ymin_vec = y_train.amin(dim=(1, 2))
            ymax_vec = y_train.amax(dim=(1, 2))

        for j, _, n_cat in cat_cols:
            all_int = bool((x_g[:, j] == x_g[:, j].round()).all())
            if all_int:
                start = torch.where(high[..., j] <= n_cat - 1, torch.zeros_like(low[..., j]), low[..., j])
                low[..., j] = start
                high[..., j] = start + (n_cat - 1)
            else:
                integer = (x_train[..., j] == x_train[..., j].round()).all(dim=1, keepdim=True)
                start = torch.where(high[..., j] <= n_cat - 1, torch.zeros_like(low[..., j]), low[..., j])
                low[..., j] = torch.where(integer, start, low[..., j])
                high[..., j] = torch.where(integer, start + (n_cat - 1), high[..., j])
        for j, lo_f, hi_f in fixed_cols:
            low[..., j] = lo_f
            high[..., j] = hi_f

        amplitude = high - low
        amplitude = torch.where(amplitude == 0, torch.ones_like(amplitude), amplitude)
        half_amp = amplitude / 2.0
        center = (high + low) / 2

        out_x_train[g, :w_g] = ((x_train - center) / half_amp).to(dtype)
        out_y_train[g, :w_g] = y_train.to(dtype)
        out_x_pred[g, :w_g] = ((x_g[predict_idx] - center) / half_amp).to(dtype)
        out_low[g, :w_g] = low.squeeze(1)
        out_high[g, :w_g] = high.squeeze(1)
        out_ymin[g, :w_g] = ymin_vec
        out_ymax[g, :w_g] = ymax_vec
        if w_g < n_windows:         # a shorter group repeats its last window; the repeats are never read
            out_x_train[g, w_g:] = out_x_train[g, w_g - 1 : w_g].expand(n_windows - w_g, -1, -1)
            out_y_train[g, w_g:] = out_y_train[g, w_g - 1 : w_g].expand(n_windows - w_g, -1, -1)
            out_x_pred[g, w_g:] = out_x_pred[g, w_g - 1 : w_g].expand(n_windows - w_g, -1, -1)
            out_low[g, w_g:] = out_low[g, w_g - 1 : w_g].expand(n_windows - w_g, -1)
            out_high[g, w_g:] = out_high[g, w_g - 1 : w_g].expand(n_windows - w_g, -1)
            out_ymin[g, w_g:] = out_ymin[g, w_g - 1 : w_g].expand(n_windows - w_g)
            out_ymax[g, w_g:] = out_ymax[g, w_g - 1 : w_g].expand(n_windows - w_g)

    windows = _AdaptiveWindows(positions, starts_all, n_valid_all, out_low, out_high, out_ymin, out_ymax, raw_x_groups=raw_x_groups, raw_y_groups=raw_y_groups)
    return out_x_train.to(TORCH_DEVICE), out_y_train.to(TORCH_DEVICE), out_x_pred.to(TORCH_DEVICE), windows
#: </transform_adaptive>