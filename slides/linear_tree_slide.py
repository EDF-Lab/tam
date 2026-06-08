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

c_primary = '#8e44ad'
c_dark = '#2c3e50'
c_accent = '#9b59b6'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Linear Tree Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Varying-Coefficient Models & Piecewise Continuous Regressions", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "lt(x_part, slope='x_slope', max_leaves=8, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.81, "1. Varying-Coefficient Networks", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.77, 
        r"• Standard trees construct piecewise-constant surfaces (rigid step-functions)." + "\n" +
        r"• TAM elevates this into a rigorous Varying-Coefficient model by decoupling the" + "\n" +
        r"  variables defining geography ($x_{part}$) from the continuous local slopes ($x_{slope}$).",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax.text(x_left + 0.02, 0.65, r"2. Feature Map ($\Phi$) & Tensor Product", fontsize=16, fontweight='bold', color=c_dark)

draw_container(ax, x_left + 0.02, 0.38, 0.40, 0.25, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_left + 0.03, 0.60, "Primal Feature Mapping (Concatenation):", fontsize=9.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.55, r"$\phi_{lt}(x_{p}, x_{s}) = \left[ \phi_{tree\_base}(x_{p}), \quad \phi_{tree\_slope}(x_{p}) \otimes \phi_{lin}(x_{s}) \right]^\top$", fontsize=12, color=c_dark)

ax.text(x_left + 0.03, 0.49, "Architectural Guardrails (Stability):", fontsize=9.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.45, 
        r"• Collinearity Prevention: Strictly enforces `n_trees=1` to prevent" + "\n" +
        r"  catastrophic multicollinearity caused by overlapping local slopes." + "\n" +
        r"• Anti-Starvation: `max_leaves` forces even quantile thresholds.", 
        fontsize=9.5, color='#c0392b', linespacing=1.4, va='top')

ax_lt = ax.inset_axes([x_left + 0.04, 0.08, 0.36, 0.26])
ax_lt.set_xlim(0, 10)
ax_lt.set_ylim(-1.5, 2.5)
ax_lt.set_xticks([])
ax_lt.set_yticks([])

x_plot = np.linspace(0, 10, 300)
y_true = np.sin(x_plot * 0.8) + x_plot * 0.15

x_bins = [0, 2.5, 5.0, 7.5, 10]
y_bins = [0.2, 0.8, -0.2, -0.6, 1.0, 1.0]
ax_lt.step(x_bins + [10], y_bins, where='post', color='#bdc3c7', lw=2, linestyle='--', label="Base Tree (Constant)")

x_lt1, y_lt1 = [0, 2.5], [0.1, 0.9]
x_lt2, y_lt2 = [2.5, 5.0], [0.9, -0.1]
x_lt3, y_lt3 = [5.0, 7.5], [-0.1, -0.8]
x_lt4, y_lt4 = [7.5, 10.0], [-0.8, 1.2]
ax_lt.plot(x_lt1, y_lt1, color=c_primary, lw=2.5, label="Linear Tree (Slopes)")
ax_lt.plot(x_lt2, y_lt2, color=c_primary, lw=2.5)
ax_lt.plot(x_lt3, y_lt3, color=c_primary, lw=2.5)
ax_lt.plot(x_lt4, y_lt4, color=c_primary, lw=2.5)

ax_lt.plot(x_plot, y_true, color=c_dark, lw=1.5, alpha=0.5, label="True Surface")

for xb in x_bins[1:-1]:
    ax_lt.axvline(xb, color=c_dark, linestyle=':', alpha=0.3)

ax_lt.set_title("Base Intercepts + Local Tensor Slopes", fontsize=10, fontweight='bold', color=c_dark)
ax_lt.legend(loc='lower center', fontsize=7.5, ncol=3, frameon=True)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.81, "3. Block-Diagonal Regularization ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.77, r"$P_{lt} = \mathrm{diag}(P_{tree\_base}, \ P_{cross})$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.73, 
        r"• Formulated as a block-diagonal encapsulation of its internal components." + "\n" +
        r"• This anisotropic structure guarantees that the varying slopes ($\beta_1(x_{part})$)" + "\n" +
        r"  are penalized independently from the local spatial intercepts ($\beta_0(x_{part})$).",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax_ins_sparse = ax.inset_axes([x_right + 0.04, 0.39, 0.36, 0.21])
ax_ins_sparse.axis('off')

n1 = 6 
n2 = 6 
sparse_mat = np.zeros((n1+n2, n1+n2))

for i in range(n1):
    sparse_mat[i, i] = 1

for i in range(n2):
    sparse_mat[n1+i, n1+i] = 1

ax_ins_sparse.spy(sparse_mat, markersize=5, color=c_penalty)
ax_ins_sparse.set_title("Sparse Global Penalty Assembly", fontsize=10, fontweight='bold', color=c_dark)

rect_base = patches.Rectangle((-0.5, -0.5), n1, n1, linewidth=1.5, edgecolor=c_dark, facecolor='none', linestyle='--')
ax_ins_sparse.add_patch(rect_base)
ax_ins_sparse.text(n1/2 - 0.5, n1 + 0.5, r"$P_{tree\_base}$", ha='center', va='top', fontsize=10, fontweight='bold', color=c_dark)

rect_cross = patches.Rectangle((n1 - 0.5, n1 - 0.5), n2, n2, linewidth=1.5, edgecolor=c_primary, facecolor='none', linestyle='--')
ax_ins_sparse.add_patch(rect_cross)
ax_ins_sparse.text(n1 + n2/2 - 0.5, n1 - 1.0, r"$P_{cross}$", ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_primary)

ax.text(x_right + 0.02, 0.34, "4. Resolving MOB Matrix Singularities", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.30, 
        r"• The Flaw: Classical Model-Based Recursive Partitioning (MOB) fits unpenalized" + "\n" +
        r"  local OLS regressions inside every leaf. If a leaf is starved of data or lacks" + "\n" +
        r"  variance in $x_{slope}$, the matrix $(X^\top X)$ becomes singular, crashing the model." + "\n" +
        r"• The Solution: TAM formulates the VC model strictly via the Primal tensor product." + "\n" +
        r"  The structural penalty $P_{cross}$ smoothly shrinks starved unstable gradients" + "\n" +
        r"  exactly to zero, falling back safely on the global linear trend.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

msg = r"Theoretical Rule: By concatenating a base tree with a tensor product of a slope tree and a continuous variable, the algorithm computes stable varying-coefficients across structural breaks."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "linear_tree_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()