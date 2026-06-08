import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_folder = "slides/output"
os.makedirs(output_folder, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

c_primary = '#0097e6'
c_dark = '#2c3e50'
c_accent = '#74b9ff'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "Meta-Learner: OperaTAM", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Online Prediction by Expert Aggregation & Polynomial Minimax Regret", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.44, 0.86, 0.52, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.46, 0.92, "Declarative Orchestrator API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.46, 0.88, "OperaTAM(formula='y ~ l(E1) + l(E2)', algorithm='MLpol', loss_type='square', horizon_steps=1)", fontsize=10, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.80, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.81, "1. Regret Theory & Hardware Operators", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.78, 
        r"• Statically selecting a single 'best' model fails during Concept Drift." + "\n" +
        r"• OperaTAM mathematically reallocates trust using exact regret bounds.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

draw_container(ax, x_left + 0.02, 0.44, 0.40, 0.28, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_left + 0.03, 0.70, "EWA (Exponentially Weighted Average)", fontsize=10.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.66, r"Absolute Loss: $L_{k,t} = \sum_{s=1}^t \ell(x_{k,s}, y_s)$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.62, r"Softmax Trick: $w_{k,t+1} \propto \exp\left(-\eta (L_{k, t} - \max_j L_{j, t})\right)$", fontsize=10.5, color=c_primary)

ax.text(x_left + 0.03, 0.56, "MLpol (Polynomial Minimax Strategy)", fontsize=10.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.52, r"Pseudo-Regret: $r_{k,t} = \nabla \ell(\hat{y}_t, y_t) \cdot (x_{k,t} - \hat{y}_t) \quad \rightarrow \quad R_{k,t} = \sum r_{k,s}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.49, r"Weight (ReLU): $w_{k,t+1} \propto \eta_{k,t} \max(0, R_{k,t})$", fontsize=10, color=c_primary)
ax.text(x_left + 0.03, 0.46, r"Adaptive Decay: $\eta_{k,t} = \left( 1/\eta_{k,t-1} + r_{k,t}^2 + \max_j ( w_{j,t} \max(0, r_{j,t})^2 ) \right)^{-1}$", fontsize=10, color=c_penalty)

ax_ins_drift = ax.inset_axes([x_left + 0.03, 0.06, 0.38, 0.35])
t = np.linspace(0, 100, 300)
e1 = np.exp(-0.05 * t)
e2 = np.exp(-0.05 * (t - 50)**2) * 1.5
e3 = 1 / (1 + np.exp(-0.1 * (t - 60)))
sum_e = e1 + e2 + e3
w1, w2, w3 = e1/sum_e, e2/sum_e, e3/sum_e

ax_ins_drift.stackplot(t, w1, w2, w3, labels=['Expert 1 (Physics)', 'Expert 2 (Spline)', 'Expert 3 (Neural)'], 
                       colors=['#bdc3c7', c_accent, c_primary], alpha=0.85)
ax_ins_drift.set_xlim(0, 100)
ax_ins_drift.set_ylim(0, 1)
ax_ins_drift.axvline(50, color=c_dark, linestyle=':', lw=2)
ax_ins_drift.text(52, 0.1, "Drift Detected", rotation=90, fontsize=8, fontweight='bold', color=c_dark)

ax_ins_drift.set_title("Dynamic Weight Reallocation", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_drift.legend(loc='lower left', fontsize=8, frameon=True)
ax_ins_drift.set_xticks([])
ax_ins_drift.set_yticks([])

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.80, edgecolor=c_primary)

ax.text(x_right + 0.02, 0.81, "2. Data Scientist API & Tensor Architecture", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_right + 0.02, 0.78, 
        r"• Sequential Python loops over millions of rows paralyze GPU kernel launches." + "\n" +
        r"• OperaTAM reshapes data into a contiguous 3D Tensor: (Batch, Time, Experts)." + "\n" +
        r"• The sequential loop is compiled to native C++ via @torch.jit.script.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

draw_container(ax, x_right + 0.02, 0.41, 0.40, 0.30, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_right + 0.03, 0.68, "Key Orchestrator Hyperparameters:", fontsize=11, fontweight='bold', color=c_dark)

ax.text(x_right + 0.03, 0.63, "algorithm:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.14, 0.63, "'EWA' (requires tuning) or 'MLPOL' (parameter-free).", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.58, "loss_type:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.14, 0.58, "Evaluates experts via 'square' ($L_2$) or 'absolute' ($L_1$).", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.53, "eta ($\eta$):", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.12, 0.53, "Static learning rate (exclusively utilized by EWA).", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.48, "horizon_steps:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.18, 0.48, "Blind delay ($H$) shifting weights to enforce causality.", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.43, "group/date_col:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.19, 0.43, "Triggers chronological padding for 3D GPU tensors.", fontsize=9.5, color=c_dark)

ax_ins_tensor = ax.inset_axes([x_right + 0.04, 0.07, 0.36, 0.34])
ax_ins_tensor.axis('off')
ax_ins_tensor.text(0.5, 0.90, "Vectorized GPU Tensor Batching", ha='center', fontsize=10, fontweight='bold', color=c_dark)

ox, oy = 0.25, 0.25
w, h, d = 0.4, 0.35, 0.25
dx, dy = d * 0.7, d * 0.7

front = patches.Polygon([[ox, oy], [ox+w, oy], [ox+w, oy+h], [ox, oy+h]], closed=True, fill=True, facecolor=c_accent, edgecolor=c_dark, lw=1.5, alpha=0.9)
top = patches.Polygon([[ox, oy+h], [ox+w, oy+h], [ox+w+dx, oy+h+dy], [ox+dx, oy+h+dy]], closed=True, fill=True, facecolor='#bdc3c7', edgecolor=c_dark, lw=1.5, alpha=0.9)
side = patches.Polygon([[ox+w, oy], [ox+w+dx, oy+dy], [ox+w+dx, oy+h+dy], [ox+w, oy+h]], closed=True, fill=True, facecolor=c_primary, edgecolor=c_dark, lw=1.5, alpha=0.9)

ax_ins_tensor.add_patch(front)
ax_ins_tensor.add_patch(top)
ax_ins_tensor.add_patch(side)

ax_ins_tensor.text(ox + w/2, oy - 0.05, "Time ($T$)", ha='center', va='top', fontsize=9, fontweight='bold', color=c_dark)
ax_ins_tensor.text(ox - 0.04, oy + h/2, "Batch ($G$)", ha='right', va='center', fontsize=9, fontweight='bold', color=c_dark, rotation=90)
ax_ins_tensor.text(ox + w + dx/2 + 0.04, oy + h/2 + dy/2, "Experts ($K$)", ha='left', va='center', fontsize=9, fontweight='bold', color=c_dark, rotation=45)

ax_ins_tensor.text(0.5, 0.00, "Single @torch.jit.script pass over all groups simultaneously", ha='center', fontsize=8.5, color='#7f8c8d')

msg = r"Theoretical Rule: By tracking sequential regret instead of absolute error, the ensemble mathematically guarantees it will perform at least as well as the best individual expert in hindsight."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "opera_tam_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()