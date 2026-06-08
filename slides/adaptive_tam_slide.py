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

c_primary = '#e74c3c'
c_dark = '#2c3e50'
c_accent = '#ff7675'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "Meta-Learner: AdaptiveTAM", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Parallel Sliding-Windows for Sudden Concept Drift Correction", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.44, 0.86, 0.52, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.46, 0.92, "Declarative Orchestrator API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.46, 0.88, "AdaptiveTAM(base=m, adaptive_formula=..., training_window_periods=360, ...)", fontsize=12, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.81, "1. The Two-Stage Architecture & Target Modes", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.77, 
        r"• Standard GAMs assume global stationarity, failing during sudden structural breaks." + "\n" +
        r"• AdaptiveTAM decouples the forecast into a slow-moving expert and a reactive corrector.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

draw_container(ax, x_left + 0.02, 0.47, 0.40, 0.25, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_left + 0.03, 0.69, "Option 1: Residual Tracking (Pure Correction)", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.65, r"`adaptive_formula='Residualy ~ l(Lag_Res)'`", fontsize=10, color=c_penalty, family='monospace')
ax.text(x_left + 0.03, 0.61, r"$\hat{y}_t - \hat{y}_{base, t} = \beta_0(t) + \beta_1(t) \cdot \mathrm{Lag\_Residual}_t$", fontsize=11, color=c_dark)

ax.text(x_left + 0.03, 0.55, "Option 2: Dynamic Ensemble Recalibration", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.51, r"`adaptive_formula='y ~ ...', add_base_effects=True`", fontsize=10, color=c_penalty, family='monospace')
ax.text(x_left + 0.03, 0.48, r"$\hat{y}_t = \beta_0(t) + \dots + \sum_k \beta_k(t) \cdot \mathrm{Base\_Effect}_{k, t}$", fontsize=11, color=c_dark)

ax.text(x_left + 0.22, 0.44, "Tracking Unobserved Localized Shocks", ha='center', fontsize=10, fontweight='bold', color=c_dark)

ax_ins_drift = ax.inset_axes([x_left + 0.03, 0.07, 0.38, 0.35])
t = np.linspace(0, 100, 400)
y_base = np.sin(t * 0.1) 
y_true = np.copy(y_base)
y_true[200:] += 2.5 
y_adapt = np.copy(y_base)
y_adapt[200:] += 2.5 * (1 - np.exp(-(t[200:] - t[200]) / 10))

ax_ins_drift.plot(t, y_true, color='#bdc3c7', lw=3, label='True Signal (Structural Break)')
ax_ins_drift.plot(t, y_base, color=c_dark, lw=1.5, linestyle='--', label='Long-Term Expert (Fails)')
ax_ins_drift.plot(t, y_adapt, color=c_primary, lw=2.5, label='AdaptiveTAM (Local Corrector)')
ax_ins_drift.axvline(t[200], color=c_penalty, linestyle=':', lw=2)
ax_ins_drift.text(t[200] - 2, 2.0, "Sudden Concept Drift", rotation=90, va='center', fontsize=9, fontweight='bold', color=c_penalty)

ax_ins_drift.legend(loc='lower right', fontsize=8, frameon=True)
ax_ins_drift.set_xticks([])
ax_ins_drift.set_yticks([])

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.81, "2. Parallel-in-Time Sliding Windows", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.795, 
        r"• Unlike sequential recursive filters (Kalman) which step iteratively via Python loops," + "\n" +
        r"  AdaptiveTAM vectorizes time, gathering history into a massive discrete batch dimension." + "\n" +
        r"• Computes the exact primal minimizer across all windows $W$ simultaneously.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

draw_container(ax, x_right + 0.02, 0.46, 0.40, 0.27, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_right + 0.03, 0.70, "Key Sliding-Window Hyperparameters:", fontsize=11, fontweight='bold', color=c_dark)

ax.text(x_right + 0.03, 0.67, "steps_per_period:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.17, 0.67, "Base temporal unit (e.g., for 30min data:", fontsize=9.5, color=c_dark)
ax.text(x_right + 0.17, 0.67-0.02, "if 48 then 1 group per time-of-day if 1 then all dataset = one group).", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.62, "training_window_periods:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.23, 0.62, r"History size (e.g., 360 for 30min data means :", fontsize=9.5, color=c_dark)
ax.text(x_right + 0.23, 0.62-0.02, r"360 days if steps = 48 or 360/48 days if steps=1).", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.57, "horizon_steps:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.14, 0.57, "Blind delay (e.g., 1 for 30min data means :", fontsize=9.5, color=c_dark)
ax.text(x_right + 0.14, 0.57-0.02, "1 day-ahead if steps=48 or 30min-ahead if steps=1)", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.52, "update_interval_periods:", fontsize=9.5, fontweight='bold', color=c_primary)
ax.text(x_right + 0.22, 0.52, "Slide frequency (e.g., 1 for 30min data means :", fontsize=9.5, color=c_dark)
ax.text(x_right + 0.22, 0.52-0.02, "slide 1 day if steps=48, or 30min if steps=1)", fontsize=9.5, color=c_dark)

ax.text(x_right + 0.03, 0.47, "Result:", fontsize=9.5, fontweight='bold', color=c_dark)
ax.text(x_right + 0.09, 0.47, r"Adds one aligned $W_{train}$ & $W_{pred}$ block per interval.", fontsize=9.5, color=c_dark, style='italic')

ax.text(x_right + 0.22, 0.435, "Vectorized GPU Tensor Batching by Group (Reverse Chronological)", ha='center', fontsize=10, fontweight='bold', color=c_dark)

ax_ins_tensor = ax.inset_axes([x_right + 0.04, 0.08, 0.36, 0.33])
ax_ins_tensor.axis('off')

train_w = 0.38
pred_w = 0.09
gap_w = 0.04
shift_w = 0.09

for step in range(4):
    y_pos = 0.85 - (step * 0.23)
    ax_ins_tensor.plot([0, 1], [y_pos, y_pos], color='#ecf0f1', lw=4, zorder=1)
    
    start_x = 0.10 + (step * shift_w)
    
    ax_ins_tensor.plot([start_x, start_x + train_w], [y_pos, y_pos], color=c_dark, lw=8, zorder=2)
    
    ax_ins_tensor.plot([start_x + train_w, start_x + train_w + gap_w], [y_pos, y_pos], color=c_penalty, lw=2, linestyle=':', zorder=2)
    
    pred_start = start_x + train_w + gap_w
    ax_ins_tensor.plot([pred_start, pred_start + pred_w], [y_pos, y_pos], color=c_primary, lw=8, zorder=2)
    
    ax_ins_tensor.text(start_x + train_w/2, y_pos + 0.04, "$W_{train}$", ha='center', fontsize=8, fontweight='bold', color=c_dark)
    ax_ins_tensor.text(start_x + train_w + gap_w/2, y_pos + 0.04, "$H$", ha='center', fontsize=8, fontweight='bold', color=c_penalty)
    ax_ins_tensor.text(pred_start + pred_w/2, y_pos + 0.04, "$W_{pred}$", ha='center', fontsize=8, fontweight='bold', color=c_primary)

ax_ins_tensor.text(0.5, 0.09, "All localized windows are packed into a single Batched Matrix Multiplication", ha='center', fontsize=8.5, color='#7f8c8d')

msg = r"Theoretical Rule: By explicitly fitting an isolated structural penalty ($W \cdot P$) dynamically over rolling residuals, the framework neutralizes concept drift instantly without catastrophic forgetting."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "adaptive_tam_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()