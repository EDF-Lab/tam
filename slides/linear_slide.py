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

c_primary = "#0984e3"
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

ax.text(0.04, 0.94, "The Linear Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Dot Product Kernel Projection & Regularized Global Intercept Integration", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "l(x, scaled=3.14, ap=-30)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Dot Product Kernel Projection", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Projects raw continuous inputs directly into the Primal space manifold." + "\n" +
        r"• Strictly corresponds to the homogeneous Dot Product (Linear) Kernel regime." + "\n" +
        r"  Bypasses scaling limits to evaluate massive continuous trajectories efficiently.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• scaled: Optional baseline scalar factor (s) | ap: Structural penalty power", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.43, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.57, r"$\mathbf{Linear\ Feature\ Map\ Equation:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.53, r"$\phi_{lin}(x) = s \cdot x \quad \mathrm{with} \quad s = \pi \quad \text{(WeaKL legacy default)}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.48, r"$\mathbf{Global\ Matrix\ Intercept\ Offset:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.44, r"$\phi_{offset}(x) = 1 \quad \mathrm{with} \quad \lambda_{offset} \to 0 \quad \text{(Improper Prior)}$", fontsize=10, color=c_dark)

ax_ins_harmonic = ax.inset_axes([x_left + 0.02, 0.08, 0.40, 0.32])
x_plot = np.linspace(-1, 1, 100)
ax_ins_harmonic.plot(x_plot, np.pi * x_plot, color=c_primary, lw=2.5, label=r'Scaled Slope ($\pi \cdot x$)')
ax_ins_harmonic.plot(x_plot, np.ones_like(x_plot), color=c_dark, lw=1.5, linestyle='--', label='Intercept Offset (1)')
ax_ins_harmonic.set_title("Primal Feature Projections", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_harmonic.legend(loc='lower right', fontsize=8, frameon=True)
ax_ins_harmonic.tick_params(labelsize=8)
ax_ins_harmonic.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Regularized Isotropic Identity Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{lin} = \lambda I \quad \mathrm{with} \quad \lambda = 10^{ap}$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• Standard OLS leaves intercepts unpenalized to maintain translation invariance." + "\n" +
        r"• TAM embeds the bias directly as a regularized weight via an improper prior." + "\n" +
        r"• Maintains absolute hardware tensor uniformity across backend execution blocks," + "\n" +
        r"  preventing structural divergence to optimize GPU cache allocation bounds.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_penalty = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
lambda_vals = np.logspace(-4, 2, 100)
weights_shrinkage = 1.0 / (1.0 + lambda_vals)
ax_ins_penalty.plot(lambda_vals, weights_shrinkage, color=c_penalty, lw=2.5)
ax_ins_penalty.set_xscale('log')
ax_ins_penalty.set_title(r"Isotropic Ridge Coefficient Shrinkage", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_penalty.set_xlabel(r"Regularization Multiplier ($\lambda$)", fontsize=8)
ax_ins_penalty.set_ylabel("Normalized Weight Magnitude", fontsize=8)
ax_ins_penalty.tick_params(labelsize=8)
ax_ins_penalty.grid(True, which="both", alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Matrix Uniformity & OOM Prevention", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Traditional frameworks separate fixed intercepts from continuous smooth lines." + "\n" +
        r"• TAM processes the offset and slope inside the same global quadratic solver pass." + "\n" +
        r"• Maximizes hardware efficiency by feeding a standard, homogenous matrix block." + "\n" +
        r"  Bypasses custom indexing loops to eliminate intermediate VRAM fragmentation leaks.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Incorporating the global bias directly into the penalized linear system guarantees maximum matrix uniformity and clean hardware caching loops."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "linear_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()