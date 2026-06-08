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

c_primary = '#e67e22'
c_dark = '#2c3e50'
c_accent = '#f39c12'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Spline Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Penalized B-Splines (P-Splines) for Smooth Non-Linear Approximations", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "s(x, k=10, deg=3, p=2, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Continuous vs. Discrete Splines", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Classical exact smoothing splines operate in an infinite dual RKHS space." + "\n" +
        r"• This generates dense matrices requiring an intractable $\mathcal{O}(N^3)$ inversion." + "\n" +
        r"• TAM utilizes discrete P-splines to truncate the problem into a finite primal space.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• k: Number of fixed knots | deg: Polynomial degree | p: Difference penalty order", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.43, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.57, r"$\mathbf{Recursive\ Cox-de\ Boor\ Formula:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.53, r"$B_{i,m}(x) = \frac{x - \tau_i}{\tau_{i+m-1} - \tau_i} B_{i,m-1} + \frac{\tau_{i+m} - x}{\tau_{i+m} - \tau_{i+1}} B_{i+1,m-1}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.48, r"$\mathbf{Explicit\ Finite-Dimensional\ Feature\ Map:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.44, r"$\phi_{spline}(x) = [B_{1,m}(x), \dots, B_{K,m}(x)]^\top$", fontsize=10, color=c_dark)

ax_ins_harmonic = ax.inset_axes([x_left + 0.02, 0.08, 0.40, 0.32])
x_plot = np.linspace(-1, 1, 300)
def spline_bump(x, center, width=0.5):
    return np.maximum(0, 1 - ((x - center) / width) ** 2) ** 3

b1 = spline_bump(x_plot, -0.4)
b2 = spline_bump(x_plot, 0.0)
b3 = spline_bump(x_plot, 0.4)
composite = 0.5 * b1 + 0.9 * b2 + 0.3 * b3

ax_ins_harmonic.plot(x_plot, b1, 'k--', alpha=0.25)
ax_ins_harmonic.plot(x_plot, b2, 'k--', alpha=0.25)
ax_ins_harmonic.plot(x_plot, b3, 'k--', alpha=0.25)
ax_ins_harmonic.plot(x_plot, composite, color=c_primary, lw=2.5, label=r'Composite Spline')
ax_ins_harmonic.set_title("B-Spline Basis Components", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_harmonic.legend(loc='upper right', fontsize=8, frameon=True)
ax_ins_harmonic.tick_params(labelsize=8)
ax_ins_harmonic.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. The Discrete Difference Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{spline} = \lambda (\Delta_d^\top \Delta_d)$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• Avoids computationally wasteful integration of continuous derivatives." + "\n" +
        r"• Applies a discrete difference matrix $\Delta_d$ directly to adjacent coefficients." + "\n" +
        r"• Proxy strictly conserves moments (mean, variance) without boundary bias," + "\n" +
        r"  providing stable regularization block-diagonally in the global system.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_penalty = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
x_smooth = np.linspace(-1, 1, 200)
y_noisy = np.sin(np.pi * x_smooth) + 0.15 * np.sin(12 * np.pi * x_smooth)
y_fitted_smooth = np.sin(np.pi * x_smooth)

ax_ins_penalty.plot(x_smooth, y_noisy, color='#bdc3c7', alpha=0.6, label='Unpenalized Overfit')
ax_ins_penalty.plot(x_smooth, y_fitted_smooth, color=c_penalty, lw=2.2, label='Penalized Smooth ($\lambda$)')
ax_ins_penalty.set_title("Roughness Penalty Effect", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_penalty.legend(fontsize=8, loc='lower center')
ax_ins_penalty.tick_params(labelsize=8)
ax_ins_penalty.grid(True, alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Boundary Extrapolation & Engineering Novelty", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Polynomial expansions suffer from Runge's phenomenon outside the domain." + "\n" +
        r"• TAM computes the exact Jacobian at boundaries via PyTorch finite differences." + "\n" +
        r"• Projects a stable, first-order Taylor linear extension infinitely out-of-distribution." + "\n" +
        r"• Decouples knot dimension from scale ($K \ll N$) to form matrix-free operations," + "\n" +
        r"  bypassing traditional CPU storage limits to execute natively on GPUs.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Spline effects use discrete difference matrices as perfect algebraic proxies for continuous derivative integrals. Linearity is forced out-of-bounds."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "spline_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()