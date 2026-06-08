import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_folder = "slides/output"
os.makedirs(output_folder, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

c_dark = '#2c3e50'
c_kalman = '#e67e22'   # Orange
c_hier = '#16a085'     # Teal
c_auto = '#9b59b6'     # Purple
c_safety = '#27ae60'   # Green

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "Expanding the Frontier: Research Ecosystem", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "BETA & EXP Modules: State-Space, Hierarchies, AutoML, and Risk Bounds", fontsize=14, color='#7f8c8d', va='top')

# =======================================================
# 1. KalmanTAM (Top Left) | Box: Y=0.48 to 0.82 (H:0.34)
# =======================================================
draw_container(ax, 0.04, 0.48, 0.44, 0.34, edgecolor=c_kalman)
ax.text(0.06, 0.77, "KalmanTAM (BETA) - Continuous Parameter Drift", fontsize=14, fontweight='bold', color=c_kalman)
ax.text(0.06, 0.68, 
        "• Tracks evolving coefficients step-by-step using an Extended\n"
        "  Kalman Filter (EKF) in a state-space formulation.\n"
        "• Ideal for smooth, stochastic drift rather than sudden shocks.", 
        fontsize=10.5, color='#34495e', linespacing=1.4)
ax.text(0.06, 0.58, 
        "ta.KalmanTAM(base_model=m, kalman_formula='y ~ l(L_Res)',\n"
        "             date_col='date', horizon_steps=1)", 
        fontsize=10.5, color='#f39c12', family='monospace', fontweight='bold')

# =======================================================
# 2. HierarchicalTAM (Top Right) | Box: Y=0.48 to 0.82 (H:0.34)
# =======================================================
draw_container(ax, 0.52, 0.48, 0.44, 0.34, edgecolor=c_hier)
ax.text(0.54, 0.77, "HierarchicalTAM (BETA) - Hierarchical Coherence", fontsize=14, fontweight='bold', color=c_hier)
ax.text(0.54, 0.68, 
        "• Solves the aggregation problem by forcing joint optimization\n"
        "  directly in the Primal RKHS space.\n"
        "• Mathematically guarantees: National Forecast = Sum of Regions.", 
        fontsize=10.5, color='#34495e', linespacing=1.4)
ax.text(0.54, 0.58, 
        "ta.HierarchicalTAM(structure={'y': ['yA','yB']},\n"
        "                   formulas=node_formulas, node_col='Node')", 
        fontsize=10.5, color='#1abc9c', family='monospace', fontweight='bold')

# =======================================================
# 3. AutoTAM (Bottom Left) | Box: Y=0.08 to 0.44 (H:0.36)
# =======================================================
draw_container(ax, 0.04, 0.08, 0.44, 0.36, edgecolor=c_auto)
ax.text(0.06, 0.39, "AutoTAM (EXP) - Evolutionary AutoML", fontsize=14, fontweight='bold', color=c_auto)
ax.text(0.06, 0.30, 
        "• Automates the discovery of the optimal formula topology.\n"
        "• Uses evolutionary search and parsimony pruning to select\n"
        "  the best combinations of splines, fourier, and neural features.", 
        fontsize=10.5, color='#34495e', linespacing=1.4)
ax.text(0.06, 0.18, 
        "ta.AutoTAM(formula='y ~ AutoPipe(x1, x2, Lag_y)',\n"
        "           n_experts=5, pop_size=20, use_opera=True)", 
        fontsize=10.5, color='#9b59b6', family='monospace', fontweight='bold')

# =======================================================
# 4. SafetyTAM (Bottom Right) | Box: Y=0.08 to 0.44 (H:0.36)
# =======================================================
draw_container(ax, 0.52, 0.08, 0.44, 0.36, edgecolor=c_safety)
ax.text(0.54, 0.39, "SafetyTAM (EXP) - Uncertainty & Risk", fontsize=14, fontweight='bold', color=c_safety)
ax.text(0.54, 0.30, 
        "• Upgrades standard point forecasts into statistically\n"
        "  guaranteed confidence intervals.\n"
        "• Powered by Adaptive Conformal Inference (ACI).", 
        fontsize=10.5, color='#34495e', linespacing=1.4)
ax.text(0.54, 0.20, 
        "safety = ta.SafetyTAM(alpha=0.1)\n"
        "safety.calibrate(y_true=y_val, y_pred=preds_val)\n"
        "safety.predict_intervals(y_pred_test, y_true_test, method='aci')", 
        fontsize=10, color='#2ecc71', family='monospace', fontweight='bold', linespacing=1.6)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(output_folder, "frontier_modules_slide.png"), dpi=300, bbox_inches='tight')
plt.close()