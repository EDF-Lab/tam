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
c_primary = '#34495e'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "MLOps: Empirical Evaluation & Diagnostics", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "BenchmarkTracker: Scale-Independent Metrics & Production Safeguards", fontsize=14, color='#7f8c8d', va='top')

# =======================================================
# 1. SMAPE | Box: Y=0.60 to 0.84 (H:0.24)
# =======================================================
draw_container(ax, 0.04, 0.60, 0.92, 0.24, edgecolor='#bdc3c7', facecolor='#f8f9fa')
ax.text(0.06, 0.79, "1. The Target-Zero Problem & SMAPE", fontsize=15, fontweight='bold', color=c_dark)
ax.text(0.06, 0.74, "• Standard MAPE explodes to infinity if the true target is zero, instantly crashing automated pipelines.", fontsize=12, color=c_primary)
ax.text(0.06, 0.69, r"• TAM defaults to Symmetric MAPE (SMAPE) to bound maximum error to exactly $200\%$.", fontsize=12, color=c_primary)
ax.text(0.06, 0.64, r"$\mathrm{SMAPE} = \frac{100}{N} \sum \frac{|Y_i - \hat{Y}_i|}{(|Y_i| + |\hat{Y}_i|)/2}$", fontsize=14, color='#e74c3c')

# =======================================================
# 2. Temporal Degradation | Box: Y=0.32 to 0.56 (H:0.24)
# =======================================================
draw_container(ax, 0.04, 0.32, 0.92, 0.24, edgecolor='#bdc3c7', facecolor='#f8f9fa')
ax.text(0.06, 0.51, "2. Temporal Degradation Tracking", fontsize=15, fontweight='bold', color=c_dark)
ax.text(0.06, 0.46, "• Splits the out-of-sample test array chronologically ($\mathcal{H}_1$ and $\mathcal{H}_2$).", fontsize=12, color=c_primary)
ax.text(0.06, 0.41, "• If the Degradation Ratio reaches +20%, it signals the need to trigger AdaptiveTAM or KalmanTAM.", fontsize=12, color=c_primary)
ax.text(0.06, 0.36, r"$\mathrm{Degradation} (\%) = \left( \frac{\mathrm{RMSE}_{\mathcal{H}_2} - \mathrm{RMSE}_{\mathcal{H}_1}}{\mathrm{RMSE}_{\mathcal{H}_1}} \right) \times 100$", fontsize=14, color='#e74c3c')

# =======================================================
# 3. Residual Autocorrelation | Box: Y=0.04 to 0.28 (H:0.24)
# =======================================================
draw_container(ax, 0.04, 0.04, 0.92, 0.24, edgecolor='#bdc3c7', facecolor='#f8f9fa')
ax.text(0.06, 0.23, "3. Residual Autocorrelation (Durbin-Watson Proxy)", fontsize=15, fontweight='bold', color=c_dark)
ax.text(0.06, 0.18, "• If residuals are serially correlated, the model is missing a critical time-dependent feature.", fontsize=12, color=c_primary)
ax.text(0.06, 0.13, r"• Computes the Lag-1 Autocorrelation ($\rho_1$) as a highly efficient proxy for the canonical Durbin-Watson test.", fontsize=12, color=c_primary)
ax.text(0.06, 0.08, r"$\rho_1 = \frac{\mathrm{Cov}(\epsilon_t, \epsilon_{t-1})}{\mathrm{Var}(\epsilon_t)} \quad \rightarrow \quad DW \approx 2(1 - \rho_1)$", fontsize=14, color='#e74c3c')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(os.path.join(output_folder, "mlops_evaluation_slide.png"), dpi=300, bbox_inches='tight')
plt.close()