#: <autotam_report_generator_module_doc>
"""
Report Generator for Automated TAM (AutoTAM).

This module parses the diagnostic CSV artifacts exported by the pipeline
and compiles them into a comprehensive, multi-pane Matplotlib dashboard. 
It visualizes the evolutionary search trajectory, collinearity pruning, 
and the composition of the final Minimax ensembles.
"""
#: </autotam_report_generator_module_doc>

#: <autotam_report_generator_imports>
import os
import glob
import re
import pandas as pd
import numpy as np
#: </autotam_report_generator_imports>

#: <autotam_report_generator_helpers>
def get_latest_run_id(export_path: str) -> str:
    """
    Scans the export directory to find the most recent AutoTAM run based on timestamp.
    """
    files = glob.glob(os.path.join(export_path, "AutoTAM_final_architectures_*.csv"))
    if not files: 
        return None
    latest_file = max(files, key=os.path.getmtime)
    match = re.search(r'_(\d{8}_\d{6})\.csv', os.path.basename(latest_file))
    return match.group(1) if match else None

def extract_family(model_name: str) -> str:
    """
    Parses the model name to determine its originating mathematical family.
    """
    parts = str(model_name).split('_')
    for part in parts:
        if 'Island' in part or 'Continent' in part: 
            return part.replace('Top', '').replace('Island', '')
    return "Other"

def shorten_label(label: str, width: int = 20) -> str:
    """
    Truncates long feature names for cleaner visualization on plot axes.
    """
    if len(str(label)) > width:
        return str(label)[:width-3] + '...'
    return str(label)
#: </autotam_report_generator_helpers>

#: <autotam_report_generator_main>
def generate_autotam_report(export_path: str = "", run_id: str = None, metric: str = "RMSE") -> None:
    """
    Generates a 3x3 visual dashboard analyzing the AutoTAM selection process.
    
    Args:
        export_path (str): Directory containing the CSV artifacts.
        run_id (str): Specific timestamp ID to plot. If None, uses the latest run.
        metric (str): The name of the optimization metric to display on the axes.
    """
    import matplotlib.pyplot as plt
    if run_id is None:
        run_id = get_latest_run_id(export_path)
        if not run_id:
            print("No evaluation logs found in the specified directory.")
            return

    f_purge = os.path.join(export_path, f"AutoTAM_collinearity_purge_log_{run_id}.csv")
    f_tests = os.path.join(export_path, f"AutoTAM_chronological_tests_{run_id}.csv")
    f_arch = os.path.join(export_path, f"AutoTAM_final_architectures_{run_id}.csv")
    f_apex = os.path.join(export_path, f"AutoTAM_opera_weights_AutoTAM_Apex_Ensemble_{run_id}.csv")
    f_isl = os.path.join(export_path, f"AutoTAM_opera_weights_Ensemble_Island_Federation_{run_id}.csv")
    f_stat = os.path.join(export_path, f"AutoTAM_opera_weights_Ensemble_Static_{run_id}.csv")
    f_adapt = os.path.join(export_path, f"AutoTAM_opera_weights_Ensemble_Adaptive_{run_id}.csv")
    f_pdp = os.path.join(export_path, f"AutoTAM_pdp_variance_Global_Static_Champion_{run_id}.csv")

    fig, axes = plt.subplots(3, 3, figsize=(24, 16))
    fig.suptitle(f"Automated Forecasting AI: Selection & Performance Report (Run: {run_id})", fontsize=22, fontweight='bold')
    cmap = plt.get_cmap('tab10')
    metric_label = str(metric).upper()

    if os.path.exists(f_purge):
        df_purge = pd.read_csv(f_purge).head(10) 
        ax = axes[0, 0]
        y_labels = [shorten_label(x) for x in df_purge['Dropped_Feature']]
        ax.barh(y_labels, df_purge['Pearson_Correlation'], color='salmon', edgecolor='k')
        ax.set_xlim(0.8, 1.05)
        ax.set_title("1. Discarded Redundant Data (Collinearity)", fontweight='bold')
        ax.set_xlabel("Pearson Correlation")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    if os.path.exists(f_tests):
        df_tests = pd.read_csv(f_tests).dropna(subset=['Validation_RMSE'])
        cap_val = df_tests['Validation_RMSE'].quantile(0.90)
        df_tests = df_tests[df_tests['Validation_RMSE'] <= cap_val].reset_index(drop=True)
        ax = axes[0, 1]
        for i, phase in enumerate(df_tests['Phase'].unique()):
            subset = df_tests[df_tests['Phase'] == phase]
            ax.scatter(subset.index, subset['Validation_RMSE'], label=phase, alpha=0.7, edgecolors='k', color=cmap(i))
        ax.set_title("2. AI Learning Curve (Search Progress)", fontweight='bold')
        ax.set_ylabel(f"Cross-Validation Error ({metric_label} - Lower = Better)")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    if os.path.exists(f_arch):
        df_arch = pd.read_csv(f_arch)
        cap_val = df_arch['Validation_RMSE'].quantile(0.85)
        df_clean = df_arch[df_arch['Validation_RMSE'] <= cap_val].copy()
        ax = axes[0, 2]
        types = df_clean['Type'].unique()
        data = [df_clean[df_clean['Type'] == t]['Validation_RMSE'].values for t in types]
        ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.set_xticks(range(1, len(types) + 1))
        ax.set_xticklabels([str(t).capitalize() for t in types], fontsize=10)
        ax.set_title("3. Fixed Rules (Static) vs. Rolling Window (Adaptive)", fontweight='bold')
        ax.set_ylabel(f"Cross-Validation Error ({metric_label} - Lower = Better)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    if 'df_clean' in locals():
        df_clean['Family'] = df_clean['Model_Name'].apply(extract_family)
        ax = axes[1, 0]
        families = df_clean['Family'].unique()
        data = [df_clean[df_clean['Family'] == f]['Validation_RMSE'].values for f in families]
        ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        ax.set_yticks(range(1, len(families) + 1))
        ax.set_yticklabels(families, fontsize=9)
        ax.set_title("4. Best Performing Math Engines (By Base Type)", fontweight='bold')
        ax.set_xlabel(f"Cross-Validation Error ({metric_label} - Lower = Better)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    if os.path.exists(f_pdp):
        df_pdp = pd.read_csv(f_pdp).sort_values('Variance', ascending=True).tail(6)
        ax = axes[1, 1]
        y_labels = [shorten_label(x, 25) for x in df_pdp['Feature_Effect']]
        ax.barh(y_labels, df_pdp['Variance'], color='mediumpurple', edgecolor='k')
        ax.set_title("5. Top Drivers of the Winning Forecast", fontweight='bold')
        ax.set_xlabel("Importance Score")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    if os.path.exists(f_apex):
        df_apex = pd.read_csv(f_apex)
        df_apex['date'] = pd.to_datetime(df_apex['date'])
        cols = [c for c in df_apex.columns if c != 'date']
        ax = axes[1, 2]
        
        labels = [c.replace('weight_', '').replace('Adaptive_', 'A_').replace('Static_', 'S_')[:15] for c in cols]
        
        ax.stackplot(df_apex['date'], [df_apex[c] for c in cols], labels=labels, alpha=0.8)
        ax.set_title("6. Final Forecast Blend (Model Weights Over Time)", fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Models")
        ax.tick_params(axis='x', rotation=30, labelsize=9)

    if os.path.exists(f_isl):
        df_isl = pd.read_csv(f_isl)
        cols = [c for c in df_isl.columns if c != 'date']
        means = df_isl[cols].mean().sort_values(ascending=False).head(8)
        ax = axes[2, 0]
        x_labels = [c.replace('weight_Top', '').replace('Island', '') for c in means.index]
        ax.bar(x_labels, means.values, color='gold', edgecolor='k')
        ax.set_title("7. Most Trusted Components in the Blend", fontweight='bold')
        ax.set_ylabel("Average Blend Weight")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    if os.path.exists(f_stat):
        df_stat = pd.read_csv(f_stat)
        r_weights = {'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0}
        for col in df_stat.columns:
            if col == 'date': continue
            for r in r_weights.keys():
                if f'_{r}_' in col: r_weights[r] += df_stat[col].mean()
        ax = axes[2, 1]
        ax.bar(list(r_weights.keys()), list(r_weights.values()), color='coral', edgecolor='k')
        ax.set_title("8. Preferred Model Complexity (R1=Simple → R5=Deep)", fontweight='bold')
        ax.set_ylabel("Total Weight in Blend")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    if os.path.exists(f_adapt):
        df_adapt = pd.read_csv(f_adapt)
        strats = {'AR_Only': 0, 'Effects_Only': 0, 'Full_ECM': 0}
        for col in df_adapt.columns:
            if col == 'date': continue
            for s in strats.keys():
                if s in col: strats[s] += df_adapt[col].mean()
        ax = axes[2, 2]
        x_labels = [s.replace('_', ' ') for s in strats.keys()]
        ax.bar(x_labels, list(strats.values()), color='teal', edgecolor='k')
        ax.set_title("9. Best Adaptive Strategies", fontweight='bold')
        ax.set_ylabel("Total Weight in Blend")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
#: </autotam_report_generator_main>

#: <autotam_report_generator_recipe>
def _print_expert_details(model_instance, expert_name: str, indent: str = "") -> None:
    """
    Helper method to print the internal mechanics of a single expert.
    """
    expert = next((e for e in model_instance.trained_experts if e["name"] == expert_name), None)
    if not expert:
        print(f"{indent}- {expert_name}: Model details not found.")
        return

    m_type = expert.get("type", "unknown").capitalize()
    island = expert.get("island", "Unknown")
    print(f"{indent}- {expert_name} ({m_type} | {island})")
    
    if m_type in ["Kalman", "Adaptive"]:
        dyn_form = expert.get("dynamic_formula", "Unknown")
        params = expert.get("params", {})
        print(f"{indent}    * Dynamic Update Formula : {dyn_form}")
        print(f"{indent}    * State Hyperparameters  : {params}")
        
        base_m = expert.get("model")
        if base_m:
            base_obj = getattr(base_m, '_saved_base_model', getattr(base_m, 'base_model_', None))
            base_form = getattr(base_obj, 'formula_', 'Unknown')
            print(f"{indent}    * Base Static Formula    : {base_form}")
    else:
        base_m = expert.get("model")
        base_form = getattr(base_m, 'formula_', 'Unknown') if base_m else 'Unknown'
        print(f"{indent}    * Static Formula         : {base_form}")

def print_model_recipe(model_instance, internal_model_name: str) -> None:
    """
    Prints the complete mathematical recipe for a given model or ensemble.
    Includes aggregation weights for OPERA models and base formulas for dynamic models.
    """
    print(f"\n{'='*90}\nRECIPE FOR: {internal_model_name}\n{'='*90}")
    
    opera_weights = {}
    if internal_model_name == "AutoTAM_Apex_Ensemble":
        opera_weights = model_instance.weights_top10
    elif internal_model_name == "Ensemble_Island_Federation":
        opera_weights = model_instance.league_weights.get("Island_Federation", {})
    elif internal_model_name.startswith("Ensemble_"):
        league = internal_model_name.replace("Ensemble_", "")
        opera_weights = model_instance.league_weights.get(league, {})
        
    if opera_weights:
        print("[OPERA Minimax Ensemble]")
        print("Aggregation Formula:")
        terms = [f"({w:.3f} * {m})" for m, w in opera_weights.items()]
        target = getattr(model_instance.ctx, 'target', 'Target') if model_instance.ctx else 'Target'
        print(f"  {target} = " + " + \n      ".join(terms))
        print("\n[Component Base Models]:")
        for m in opera_weights.keys():
            real_m = model_instance.island_aliases.get(m, m)
            _print_expert_details(model_instance, real_m, indent="  ")
        print("="*90)
        return

    real_name = model_instance.island_aliases.get(internal_model_name, internal_model_name)
    _print_expert_details(model_instance, real_name, indent="")
    print("="*90)
#: </autotam_report_generator_recipe>