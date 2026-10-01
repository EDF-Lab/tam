# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Plotting utilities for visualizing the components of an StaticTAM model.

This module provides functions to create scatter plots of individual
feature effects to facilitate model interpretation.

``decompose_prediction`` names one column ``effect_<component>`` per effect
(``decomposition_names``): a feature used by a single effect keeps its own name,
while a tensor product is named ``te_<a>_x_<b>`` and effects sharing a feature
are prefixed by their basis (``s_x``, ``l_x``). The helpers below resolve a
feature name to its component before plotting.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from tam.model.additive import StaticTAM

# Prefixes decomposition_names() puts in front of a feature shared by several effects.
_BASIS_PREFIXES = ("l", "f", "s", "w", "p", "c", "n", "rbf", "phys")


def _effect_features(effect) -> List[str]:
    r"""Raw feature names an effect reads (both margins for a tensor product)."""
    margins = getattr(effect, "effects", None)
    if margins:
        return [f for m in margins for f in _effect_features(m)]
    return list(getattr(effect, "input_features", None) or [effect.feature_name])


def _name_features(name: str, columns) -> List[str]:
    r"""Features named by a component name, read from a decomposed frame (no model at hand)."""
    if name.startswith("te_"):
        return name[3:].split("_x_")
    prefix, _, rest = name.partition("_")
    if prefix in _BASIS_PREFIXES and rest in columns:
        return [rest]
    return [name]


def _pick(feature: str, candidates: dict, component: Optional[str], color_by: Optional[str]) -> str:
    r"""Choose one component among ``{name: features}`` for ``feature``."""
    if component is not None:
        if component not in candidates:
            raise ValueError(f"Unknown component '{component}'. Components: {sorted(candidates)}.")
        return component
    matching = [name for name, feats in candidates.items() if feature in feats]
    if len(matching) > 1 and color_by is not None:
        narrowed = [name for name in matching if color_by in candidates[name]]
        if len(narrowed) == 1:
            return narrowed[0]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ValueError(f"No component uses the feature '{feature}'. Components: {sorted(candidates)}.")
    raise ValueError(
        f"The feature '{feature}' is used by several components: {matching}. "
        "Pass component=<one of them> (or color_by=<the other margin> for a tensor product)."
    )


def resolve_component(
    model: 'StaticTAM',
    feature: str,
    component: Optional[str] = None,
    color_by: Optional[str] = None,
) -> str:
    r"""
    Returns the decomposition component to plot for ``feature``.

    Exactly one component uses the feature: that one. Several: ``component`` picks one,
    otherwise ``color_by`` picks the tensor product whose other margin it names; if that
    still leaves several, a ``ValueError`` lists them.

    Args:
        model: A fitted model exposing ``effects_list_``.
        feature: The raw feature name (a column of the data).
        component: An explicit component name (as in ``effect_<component>``).
        color_by: The column used for coloring; disambiguates tensor products.

    Returns:
        str: The component name, without the ``effect_`` prefix.
    """
    from tam.model._math import decomposition_names
    from tam.model.spectrum import OffsetEffect

    effects = model.effects_list_
    candidates = {name: _effect_features(e)
                  for e, name in zip(effects, decomposition_names(effects)) if not isinstance(e, OffsetEffect)}
    return _pick(feature, candidates, component, color_by)


#: <plot_logic>
def plot_effect_with_data_decomposed(
    data: pd.DataFrame,
    effect: str,
    color_by: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    r"""
    Generates a scatter plot visualizing the contribution of a specific feature.

    This function expects a DataFrame containing both the feature values and
    their pre-calculated effects, as returned by ``decompose_prediction``. For a
    feature 'X', the contribution column is 'effect_X' when a single effect uses X;
    otherwise the component is resolved from the column names (see ``resolve_component``).
    If `color_by` is categorical, a legend is generated instead of a colorbar.

    Args:
        data: A DataFrame containing the feature's original values and
            its pre-calculated effect contribution.
        effect: The name of the feature to analyze.
        color_by: The name of another column in `data` to use for
            coloring the scatter points. Defaults to None.
        component: The component to plot when several use the feature
            (e.g. 's_x5' or 'te_x1_x_x8'). Defaults to None.

    Returns:
        None. Displays a matplotlib plot.
    """
    import matplotlib.pyplot as plt

    if component is None and f"effect_{effect}" in data.columns:
        component = effect
    elif component is None or f"effect_{component}" not in data.columns:
        candidates = {c[len("effect_"):]: _name_features(c[len("effect_"):], data.columns)
                      for c in data.columns if c.startswith("effect_") and c != "effect_offset"}
        component = _pick(effect, candidates, component, color_by)
    column = f"effect_{component}"

    fig, ax = plt.subplots(figsize=(12, 7))

    if color_by:
        if is_numeric_dtype(data[color_by]):
            # Numeric coloring with colorbar
            scatter = ax.scatter(
                x=data[effect],
                y=data[column],
                c=data[color_by],
                cmap='viridis',
                s=15,
                alpha=0.7,
                edgecolors='k',
                linewidth=0.5
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_by, fontsize=12)
        else:
            # Categorical coloring with legend
            unique_categories = data[color_by].unique()
            cmap = plt.get_cmap('Set2')

            for i, category in enumerate(unique_categories):
                subset = data[data[color_by] == category]
                ax.scatter(
                    x=subset[effect],
                    y=subset[column],
                    label=category,
                    color=cmap(i % cmap.N),
                    s=15,
                    alpha=0.7,
                    edgecolors='k',
                    linewidth=0.5
                )

            ax.legend(title=color_by, fontsize=10, title_fontsize=12,
                      bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    else:
        # Default plot if no color_by is provided
        ax.scatter(
            x=data[effect],
            y=data[column],
            color='steelblue',
            s=15,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )

    title = f"Effect of '{effect}' on Prediction"
    if component != effect:
        title += f" (component '{component}')"
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(f"Value of '{effect}'", fontsize=12)
    ax.set_ylabel("Contribution to Prediction", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
#: </plot_logic>

#: <plot_wrapper>
def plot_effect_with_model_and_data(
    model: 'StaticTAM',
    data: pd.DataFrame,
    effect: str,
    color_by: Optional[str] = None,
    component: Optional[str] = None,
) -> None:
    r"""
    Computes feature effects using the model and visualizes the result.

    This wrapper decomposes the prediction using the provided model instance,
    resolves ``effect`` to its component (``resolve_component``) and delegates
    plotting to `plot_effect_with_data_decomposed`.

    Args:
        model: The trained `StaticTAM` model instance.
        data: The dataset (e.g., training or validation) on which to
            compute and visualize the effects.
        effect: The name of the feature to analyze.
        color_by: The name of a column in `data` to use for coloring.
            Defaults to None.
        component: The component to plot when several use the feature.
            Defaults to None.

    Returns:
        None. Displays a matplotlib plot.
    """
    component = resolve_component(model, effect, component, color_by)
    decomposed_df = model.decompose_prediction(data)
    plot_effect_with_data_decomposed(decomposed_df, effect, color_by, component)
#: </plot_wrapper>


def _component_on_grid(model, data: pd.DataFrame, component: str, a: str, b: str, resolution: int, group):
    r"""
    Evaluates a two-margin component on a regular (a, b) grid for one group.

    The other columns are held at a representative value (median, or first value if not numeric); they do
    not enter the component. The grid spans the 1st-99th percentiles of each margin in ``data``.

    Returns:
        Tuple of the grid arrays (A, B, Z) and the group label used (None without groups).
    """
    group_col = getattr(model, "group_col_", None)
    grouped = group_col is not None and group_col in data.columns
    if grouped:
        group = data[group_col].mode().iloc[0] if group is None else group
        in_group = data[data[group_col] == group]
        if in_group.empty:
            raise ValueError(f"No row of group {group!r} in data; groups present: {sorted(data[group_col].unique())[:20]}.")
    else:
        in_group = data
    base = {c: (data[c].median() if is_numeric_dtype(data[c]) and not pd.api.types.is_datetime64_any_dtype(data[c])
                else data[c].iloc[0]) for c in data.columns}
    ga = np.linspace(*np.nanquantile(in_group[a].to_numpy(dtype=float), [0.01, 0.99]), resolution)
    gb = np.linspace(*np.nanquantile(in_group[b].to_numpy(dtype=float), [0.01, 0.99]), resolution)
    A, B = np.meshgrid(ga, gb)
    grid = pd.DataFrame({c: [v] * A.size for c, v in base.items()})
    grid[a], grid[b] = A.ravel(), B.ravel()
    date_col = getattr(model, "date_col_", None)
    if date_col is not None and date_col in data.columns:
        # Distinct, increasing timestamps keep the grid order through the pipeline's date sort.
        grid[date_col] = pd.Timestamp(data[date_col].min()) + pd.to_timedelta(np.arange(A.size), unit="s")
    if grouped:
        grid[group_col] = group
    out = model.decompose_prediction(grid)
    Z = out[f"effect_{component}"].to_numpy(dtype=float).reshape(A.shape)
    return A, B, Z, (group if grouped else None)


#: <plot_component>
def plot_component(
    model: 'StaticTAM',
    data: pd.DataFrame,
    component: str,
    kind: str = "auto",
    group=None,
    resolution: int = 60,
):
    r"""
    Plots one additive component, with a view chosen from its dimension.

    - one feature: contribution against the feature;
    - ``te(x, c)`` with a categorical or low-cardinality margin: one curve per level of ``c``;
    - ``te(x1, x2)``: a 3D surface over a regular (x1, x2) grid, or ``kind="heatmap"`` for a filled contour
      with the data points overlaid; with ``group_col``, for one group (``group``, default the most frequent);
    - ``te(x1, x2, x3)``: a 3D scatter of (x1, x2, x3) coloured by the contribution.

    Args:
        model: A fitted model exposing ``effects_list_`` and ``decompose_prediction``.
        data: The data on which to evaluate the component (and whose range sets the grid).
        component: The component name, as in ``effect_<component>`` (see ``decomposition_names``).
        kind: "auto" (default), "surface" or "heatmap" (two continuous margins only).
        group: The group to draw a two-margin surface for; default the most frequent group in ``data``.
        resolution: Grid points per axis for a two-margin surface or heatmap.

    Returns:
        matplotlib.figure.Figure: The figure (also shown).
    """
    import matplotlib.pyplot as plt
    from tam.model._math import decomposition_names

    names = decomposition_names(model.effects_list_)
    if component not in names:
        raise ValueError(f"Unknown component '{component}'. Components: {names}.")
    features = _effect_features(model.effects_list_[names.index(component)])
    frame = model.decompose_prediction(data)
    # Rows whose contribution or coordinates are not finite (e.g. a group missing at prediction time) are left out.
    finite = np.isfinite(frame[f"effect_{component}"].to_numpy(dtype=float))
    for col in features:
        if is_numeric_dtype(frame[col]):
            finite &= np.isfinite(frame[col].to_numpy(dtype=float))
    frame = frame[finite]
    z = frame[f"effect_{component}"]

    if len(features) == 1:
        (x,) = features
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(frame[x], z, color='steelblue', s=15, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(f"Value of '{x}'", fontsize=12)
        ax.set_ylabel("Contribution to Prediction", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    elif len(features) == 2:
        a, b = features
        low_card = lambda col: (not is_numeric_dtype(frame[col])) or frame[col].nunique() <= 12
        if low_card(b) or low_card(a):
            x, level = (a, b) if low_card(b) else (b, a)
            fig, ax = plt.subplots(figsize=(12, 7))
            cmap = plt.get_cmap('tab10')
            for i, value in enumerate(sorted(frame[level].unique())):
                part = frame[frame[level] == value].sort_values(x)
                ax.plot(part[x], part[f"effect_{component}"], marker='o', markersize=3, linewidth=1.2,
                        color=cmap(i % cmap.N), label=str(value))
            ax.legend(title=level, fontsize=10, title_fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_xlabel(f"Value of '{x}'", fontsize=12)
            ax.set_ylabel("Contribution to Prediction", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            A, B, Z, used_group = _component_on_grid(model, data, component, a, b, resolution, group)
            group_col = getattr(model, "group_col_", None)
            suffix = f" ({group_col} = {used_group})" if used_group is not None else ""
            if kind == "heatmap":
                fig, ax = plt.subplots(figsize=(10, 8))
                filled = ax.contourf(A, B, Z, levels=20, cmap='viridis')
                fig.colorbar(filled, ax=ax, label="Contribution to Prediction")
                support = frame if used_group is None else frame[frame[group_col] == used_group]
                # Observed points: the surface outside their cloud is extrapolated.
                ax.scatter(support[a], support[b], s=10, color='white', edgecolors='k', linewidth=0.4, alpha=0.8, label="data")
                ax.legend(loc='upper right', fontsize=9)
                ax.set_xlim(A.min(), A.max())
                ax.set_ylim(B.min(), B.max())
                ax.set_xlabel(a, fontsize=12)
                ax.set_ylabel(b, fontsize=12)
            else:
                fig = plt.figure(figsize=(11, 8))
                ax = fig.add_subplot(projection='3d')
                surf = ax.plot_surface(A, B, Z, cmap='viridis', linewidth=0, antialiased=True)
                fig.colorbar(surf, ax=ax, shrink=0.6, label="Contribution to Prediction")
                ax.set_xlabel(a)
                ax.set_ylabel(b)
                ax.set_zlabel("Contribution")
            ax.set_title(f"Component '{component}'{suffix}", fontsize=15)
            plt.tight_layout()
            plt.show()
            return fig
    else:
        a, b, c = features[:3]
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(projection='3d')
        points = ax.scatter(frame[a], frame[b], frame[c], c=z, cmap='viridis', s=12)
        fig.colorbar(points, ax=ax, shrink=0.6, label="Contribution to Prediction")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_zlabel(c)

    ax.set_title(f"Component '{component}'", fontsize=15)
    plt.tight_layout()
    plt.show()
    return fig
#: </plot_component>
