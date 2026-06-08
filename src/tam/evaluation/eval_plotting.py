#: <plotting_module_doc>
"""
Visualization and Benchmarking Module for AutoTAM.

Provides universal plotting capabilities using pure Matplotlib to generate 
comprehensive dashboards. Supports both time-series tracking and standard 
cross-sectional regression diagnostics, including residual distributions 
and temporal error heatmaps.
"""
#: </plotting_module_doc>

#: <plotting_imports>
import numpy as np
import pandas as pd
import textwrap
#: </plotting_imports>

#: <plotting_dashboard>
def plot_benchmark_dashboard(data_dict, trackers_dict, target_col='value', date_col='date', 
                             is_timeseries=True, primary_metric='MAPE', title="Benchmark Comparison",
                             heatmap_col='month', summary_text=None, forecast_smoothing=None):
    """
    Generates a universal benchmarking dashboard using Matplotlib.
    
    Dynamically adjusts layout based on whether the data is a time-series 
    or cross-sectional, ranking models by the specified primary metric.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.5, 'axes.facecolor': '#fcfcfc'})
    
    if not trackers_dict: 
        return
        
    trackers_list = list(trackers_dict.values())
    models_to_plot = [t.model_name for t in trackers_list]
    tab10_colors = plt.cm.tab10.colors
    color_map = {name: tab10_colors[i % len(tab10_colors)] for i, name in enumerate(models_to_plot)}
    
    ascending = any(m in primary_metric.upper() for m in ['RMSE', 'MAE', 'MAPE', 'SMAPE', 'NMAE', 'ERROR'])
    
    sorted_trackers = sorted(
        [t for t in trackers_list if not np.isnan(t.get_metric('test', primary_metric))],
        key=lambda x: x.get_metric('test', primary_metric),
        reverse=not ascending
    )
    if not sorted_trackers: 
        return

    fig = plt.figure(figsize=(18, 12))
    
    if is_timeseries:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[0, 1])
        ax_bottom = fig.add_subplot(gs[1, :])
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[0, 1])
        ax_bottom = fig.add_subplot(gs[1, 0]) 
        ax_text = fig.add_subplot(gs[1, 1])   
        ax_text.axis('off') 

    names = [t.model_name for t in sorted_trackers]
    scores = [t.get_metric('test', primary_metric) for t in sorted_trackers]
    
    y_pos = np.arange(len(names))
    bar_colors = [color_map[name] for name in names]
    
    ax_bar.barh(y_pos, scores, color=bar_colors, edgecolor='none')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f"Test {primary_metric}")
    ax_bar.set_title(f"Model Ranking ({'Lower' if ascending else 'Higher'} is Better)", fontweight='bold')
    ax_bar.grid(axis='y', visible=False)
    for i, v in enumerate(scores): 
        ax_bar.text(v + (max(scores)*0.02), i, f"{v:.3f}", va='center', fontweight='bold')
        
    top_n = min(4, len(sorted_trackers))
    
    if is_timeseries:
        df_full = pd.concat(data_dict.values())
        plot_df = df_full.copy()
        
        if forecast_smoothing:
            plot_df = plot_df.set_index(date_col)
            if isinstance(forecast_smoothing, int):
                plot_df = plot_df.rolling(window=forecast_smoothing, min_periods=1).mean(numeric_only=True)
            elif isinstance(forecast_smoothing, str):
                plot_df = plot_df.resample(forecast_smoothing).mean(numeric_only=True)
            plot_df = plot_df.reset_index()

        ax_main.plot(plot_df[date_col], plot_df[target_col], color='gray', alpha=0.5, label='Actuals', lw=1)
        
        for i in range(top_n):
            t = sorted_trackers[i]
            if t.y_pred_full is not None:
                temp_df = df_full.copy()
                temp_df['pred'] = t.y_pred_full
                
                if forecast_smoothing:
                    temp_df = temp_df.set_index(date_col)
                    if isinstance(forecast_smoothing, int):
                        temp_df = temp_df.rolling(window=forecast_smoothing, min_periods=1).mean(numeric_only=True)
                    elif isinstance(forecast_smoothing, str):
                        temp_df = temp_df.resample(forecast_smoothing).mean(numeric_only=True)
                    temp_df = temp_df.reset_index()
                
                ax_main.plot(temp_df[date_col], temp_df['pred'], color=color_map[t.model_name], linestyle='-', lw=1.5, label=t.model_name)
        
        current_len = 0
        splits_dates = {}
        splits_keys = list(data_dict.keys())
        y_bottom_lim, y_top_lim = ax_main.get_ylim()
        
        y_pos_main = y_bottom_lim + (y_top_lim - y_bottom_lim) * 0.95
        y_pos_sub = y_bottom_lim + (y_top_lim - y_bottom_lim) * 0.88
        
        for i, (split_name, df) in enumerate(data_dict.items()):
            split_len = len(df)
            
            if i < len(splits_keys) - 1: 
                mid_idx = min(current_len + (split_len // 2), len(df_full) - 1)
                mid_date = df_full.iloc[mid_idx][date_col]
                ax_main.text(mid_date, y_pos_sub, split_name.upper(), ha='center', va='center', 
                             fontweight='bold', fontsize=10, color='#495057')

            current_len += split_len
            if i < len(splits_keys) - 1:
                boundary_date = df_full.iloc[current_len][date_col]
                splits_dates[split_name] = boundary_date
                
                if split_name == splits_keys[-2]:
                    ax_main.axvline(x=boundary_date, color='black', linestyle='-', lw=3)
                    ax_main.text(boundary_date, y_pos_main, "← STAGE 1: SEARCH & REFIT  ", ha='right', va='center', fontweight='bold', fontsize=11)
                    ax_main.text(boundary_date, y_pos_main, "  STAGE 2: TEST →", ha='left', va='center', fontweight='bold', fontsize=11, color='#1d3557')
                else:
                    ax_main.axvline(x=boundary_date, color='black', linestyle=':', lw=2, alpha=0.7)

        if splits_dates:
            start_date = df_full[date_col].iloc[0]
            test_start_date = list(splits_dates.values())[-1]
            end_date = df_full[date_col].iloc[-1]
            ax_main.axvspan(start_date, test_start_date, color='gray', alpha=0.1)
            ax_main.axvspan(test_start_date, end_date, color='#457b9d', alpha=0.1)
            
        smoothing_label = f" [Smoothed: {forecast_smoothing}]" if forecast_smoothing else ""
        ax_main.set_title(f"Chronological Forecast (Top {top_n}){smoothing_label}", fontweight='bold')
        ax_main.legend(loc='upper left')

        if summary_text:
            ax_bottom.clear()
            ax_bottom.axis('off')
            wrapped_text = textwrap.fill(summary_text, width=130) 
            ax_bottom.text(0.5, 0.5, wrapped_text, transform=ax_bottom.transAxes, 
                           fontsize=13, va='center', ha='center', family='monospace',
                           bbox=dict(boxstyle="round,pad=1.5", fc="#f8f9fa", ec="#dee2e6", lw=1.5))
        else:
            try:
                df_test = data_dict.get('test')
                if df_test is not None and heatmap_col in df_test.columns:
                    ape_records = []
                    for t in sorted_trackers:
                        if 'test' in t.predictions:
                            trues, preds = df_test[target_col].values, t.predictions['test']
                            
                            if 'RMSE' in primary_metric.upper() or 'MSE' in primary_metric.upper():
                                metric_vals = (trues - preds)**2
                            elif 'MAPE' in primary_metric.upper():
                                metric_vals = np.abs((trues - preds) / np.maximum(np.abs(trues), 1e-8)) * 100
                            else:
                                metric_vals = np.abs(trues - preds)
                                
                            groups = df_test[heatmap_col].values
                            for grp, val in zip(groups, metric_vals):
                                ape_records.append({'Model': t.model_name, 'Group': grp, 'Error': val})

                    df_err = pd.DataFrame(ape_records)
                    pivot_err = df_err.groupby(['Model', 'Group'])['Error'].mean().reset_index().pivot(index='Model', columns='Group', values='Error')
                    pivot_err = pivot_err.reindex(sorted(pivot_err.columns), axis=1)
                    pivot_err = pivot_err.reindex(names) 
                    
                    if 'RMSE' in primary_metric.upper():
                        pivot_err = np.sqrt(pivot_err)
                    
                    data = pivot_err.values
                    masked_data = np.ma.masked_invalid(data) 
                    im = ax_bottom.imshow(masked_data, cmap="Reds", aspect="auto")
                    cbar = fig.colorbar(im, ax=ax_bottom)
                    cbar.set_label(primary_metric)
                    
                    ax_bottom.set_xticks(np.arange(len(pivot_err.columns)))
                    ax_bottom.set_yticks(np.arange(len(pivot_err.index)))
                    ax_bottom.set_xticklabels(pivot_err.columns)
                    ax_bottom.set_yticklabels(pivot_err.index)
                    
                    for i in range(len(pivot_err.index)):
                        for j in range(len(pivot_err.columns)):
                            val = data[i, j]
                            if not np.isnan(val):
                                text_color = "white" if val > np.nanmax(data)*0.5 else "black"
                                ax_bottom.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
                                
                    ax_bottom.grid(False) 
                    ax_bottom.set_title(f"Test Set Vulnerability: {primary_metric} by '{heatmap_col}'", fontweight='bold', pad=15)
                else:
                    ax_bottom.set_title(f"Temporal Heatmap unavailable. Column '{heatmap_col}' not found in Test set.")
            except Exception as e:
                ax_bottom.set_title(f"Temporal Heatmap error: {e}")

    else:
        df_test = data_dict.get('test')
        if df_test is not None:
            ax_main.plot([df_test[target_col].min(), df_test[target_col].max()], 
                         [df_test[target_col].min(), df_test[target_col].max()], 
                         color='black', linestyle='--', lw=2, label='Perfect Fit')
            for i in range(top_n):
                t = sorted_trackers[i]
                if 'test' in t.predictions: 
                    ax_main.scatter(df_test[target_col], t.predictions['test'], color=color_map[t.model_name], alpha=0.6, label=t.model_name, s=15)
            ax_main.set_title(f"True vs Predicted (Top {top_n})", fontweight='bold')
            ax_main.set_xlabel("True Values"); ax_main.set_ylabel("Predicted Values"); ax_main.legend()

            for i in range(top_n):
                t = sorted_trackers[i]
                if 'test' in t.predictions:
                    residuals = df_test[target_col].values - t.predictions['test']
                    ax_bottom.hist(residuals, bins=30, density=True, color=color_map[t.model_name], alpha=0.3, label=t.model_name, histtype='stepfilled')
                    ax_bottom.hist(residuals, bins=30, density=True, color=color_map[t.model_name], lw=2, histtype='step')
                    
            ax_bottom.axvline(0, color='black', linestyle='--', lw=2)
            ax_bottom.set_title("Residual Density Distribution", fontweight='bold')
            ax_bottom.set_xlabel("Residual Error (True - Pred)"); ax_bottom.legend()

            if summary_text:
                wrapped_text = textwrap.fill(summary_text, width=70) 
                ax_text.text(0.05, 0.90, wrapped_text, transform=ax_text.transAxes, 
                             fontsize=12, va='top', ha='left', family='monospace',
                             bbox=dict(boxstyle="round,pad=1.5", fc="#f8f9fa", ec="#dee2e6", lw=1.5))
            else:
                ax_text.text(0.5, 0.5, "Pass 'summary_text' to render diagnostics here.", 
                             transform=ax_text.transAxes, fontsize=12, va='center', ha='center', color='gray')

    plt.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()
#: </plotting_dashboard>

#: <plotting_summary_table>
def generate_summary_table(trackers_dict, primary_metric='MAPE', splits_to_show=None):
    """
    Constructs a tabular summary of model performance across data splits.
    """
    if splits_to_show is None: 
        splits_to_show = ['fit', 'dev', 'val', 'train', 'test']
    records = []
    
    for t in trackers_dict.values():
        rec = {"Model": t.model_name}
        for split in splits_to_show:
            if split in t.metrics: 
                rec[f"{split.capitalize()} {primary_metric}"] = round(t.get_metric(split, primary_metric), 4)
        rec["Fit Time (s)"] = round(t.time_fit, 3)
        rec["Predict Time (s)"] = round(t.time_predict, 3)
        
        if hasattr(t, 'diagnostics') and 'test' in t.diagnostics and 'RMSE_Drift_H2_vs_H1 (%)' in t.diagnostics['test']:
            rec["Test Drift (%)"] = round(t.diagnostics['test']['RMSE_Drift_H2_vs_H1 (%)'], 2)
            
        records.append(rec)
        
    df_sum = pd.DataFrame(records)
    ascending = any(m in primary_metric.upper() for m in ['RMSE', 'MAE', 'MAPE', 'SMAPE', 'NMAE', 'ERROR'])
    sort_col = f"Test {primary_metric}"
    
    if sort_col in df_sum.columns: 
        return df_sum.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    return df_sum
#: </plotting_summary_table>