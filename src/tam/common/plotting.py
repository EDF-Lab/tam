"""
Plotting utilities for visualizing the components of an StaticTAM model.

This module provides functions to create scatter plots of individual
feature effects to facilitate model interpretation.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tam.model.additive import StaticTAM

#: <plot_logic>
def plot_effect_with_data_decomposed(
    data: pd.DataFrame, 
    effect: str, 
    color_by: Optional[str] = None
) -> None:
    r"""
    Generates a scatter plot visualizing the contribution of a specific feature.

    This function expects a DataFrame containing both the feature values and 
    their pre-calculated effects. For a feature named 'X', the DataFrame must 
    contain columns 'X' and 'effect_X'. If `color_by` is categorical, a legend 
    is generated instead of a colorbar.

    Args:
        data: A DataFrame containing the feature's original values and
            its pre-calculated effect contribution.
        effect: The name of the feature to analyze.
        color_by: The name of another column in `data` to use for
            coloring the scatter points. Defaults to None.

    Returns:
        None. Displays a matplotlib plot.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7))

    if color_by:
        if is_numeric_dtype(data[color_by]):
            # Numeric coloring with colorbar
            scatter = ax.scatter(
                x=data[effect],
                y=data[f"effect_{effect}"],
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
                    y=subset[f"effect_{effect}"],
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
            y=data[f"effect_{effect}"],
            color='steelblue',
            s=15, 
            alpha=0.7, 
            edgecolors='k', 
            linewidth=0.5
        )

    ax.set_title(f"Effect of '{effect}' on Prediction", fontsize=15)
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
    color_by: Optional[str] = None
) -> None:
    r"""
    Computes feature effects using the model and visualizes the result.

    This wrapper decomposes the prediction using the provided model instance
    and delegates plotting to `plot_effect_with_data_decomposed`.

    Args:
        model: The trained `StaticTAM` model instance.
        data: The dataset (e.g., training or validation) on which to
            compute and visualize the effects.
        effect: The name of the feature to analyze.
        color_by: The name of a column in `data` to use for coloring.
            Defaults to None.

    Returns:
        None. Displays a matplotlib plot.
    """
    decomposed_df = model.decompose_prediction(data)
    plot_effect_with_data_decomposed(decomposed_df, effect, color_by)
#: </plot_wrapper>