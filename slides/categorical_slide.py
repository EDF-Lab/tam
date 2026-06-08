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

c_primary = '#e84393'
c_dark = '#2c3e50'
c_accent = '#fd79a8'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Categorical Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Discrete Topological Embeddings & Structured RKHS Penalization Systems", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.85, 0.41, 0.10, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.91, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.87, "c(x, n_cat=7, topo='ordinal', ap=-5)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Discrete Structural Embedding", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Maps discrete non-continuous spaces directly into valid vector blocks." + "\n" +
        r"• Bridges the gap between discrete categorical sets and continuous analytics." + "\n" +
        r"  Solves qualitative features concurrently inside the global Primal Ridge solver.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Topological Projections ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• topo: nominal / ordinal / fourier | n_cat: Number of discrete categorical bins", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.43, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.56, r"$\mathbf{Euclidean\ Indicator\ (Nominal\ /\ Ordinal):}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.52, r"$\phi_{cat}(x) = \left[ I(x=c_1), \ I(x=c_2), \ \dots, \ I(x=c_K) \right]^T$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.47, r"$\mathbf{Continuous\ Angular\ Cycle\ (Fourier):}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.43, r"$\phi_{cat}(x) = \left[ \cos\left(\frac{k \pi x}{2}\right), \ \sin\left(\frac{k \pi x}{2}\right) \right]_{k=1}^m$", fontsize=10, color=c_dark)

ax_ins_harmonic = ax.inset_axes([x_left + 0.02, 0.08, 0.40, 0.31])
cats = np.arange(1, 8)
y_nominal = np.array([0.3, 0.9, -0.5, 0.2, 0.8, -0.4, 0.4])
y_ordinal = np.array([0.0, 0.15, 0.32, 0.50, 0.65, 0.72, 0.78])
x_dense = np.linspace(1, 7, 200)
y_fourier = 0.5 * np.sin((x_dense - 1) * np.pi / 3) + 0.2

ax_ins_harmonic.plot(cats, y_nominal, color='#bdc3c7', marker='o', linestyle=':', label='Nominal (Jagged Profile)')
ax_ins_harmonic.plot(cats, y_ordinal, color=c_penalty, marker='s', linestyle='-', lw=2, label='Ordinal (Smooth Transitions)')
ax_ins_harmonic.plot(x_dense, y_fourier, color=c_primary, lw=2, label='Fourier (Continuous Wave Profile)')
ax_ins_harmonic.set_title("Fitted Category Levels Topology Comparison", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_harmonic.set_xticks(cats)
ax_ins_harmonic.set_xticklabels([f'c{i}' for i in cats])
ax_ins_harmonic.set_ylim(-1.0, 1.3)
ax_ins_harmonic.legend(loc='upper left', fontsize=7.5, frameon=True)
ax_ins_harmonic.tick_params(labelsize=8)
ax_ins_harmonic.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Structured Penalization Matrices ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{nominal} = \lambda I \quad \mathbf{\oplus} \quad P_{ordinal} = \lambda D^T D$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• Nominal: Isotropic identity structures shrink sparse category variances." + "\n" +
        r"• Ordinal: First-order finite difference blocks force adjacent cluster smoothing." + "\n" +
        r"• Fourier: Trigonometric Sobolev weight filters eliminate discrete ringing noise," + "\n" +
        r"  satisfying Mercer conditions to guarantee valid RKHS eligibility boundaries.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_penalty = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
x_indices = np.arange(1, 6)
y_p_nominal = np.ones(5)
y_p_fourier = 1.0 + (x_indices ** 2)
ax_ins_penalty.plot(x_indices, y_p_nominal, color='#bdc3c7', lw=1.5, linestyle='--', marker='o', label='Nominal Penalty (Flat diag)')
ax_ins_penalty.plot(x_indices, y_p_fourier, color=c_primary, lw=2.2, marker='^', label='Fourier Sobolev Penalty (Decay)')
ax_ins_penalty.set_title("Structural Regularization Diagonals", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_penalty.set_xticks(x_indices)
ax_ins_penalty.legend(fontsize=8, loc='upper left')
ax_ins_penalty.tick_params(labelsize=8)
ax_ins_penalty.grid(True, alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Unified Design Block System", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Classical setups isolate string categories using iterative backfitting paths." + "\n" +
        r"• TAM evaluates qualitative arrays natively via functional one-hot structures." + "\n" +
        r"• Concatenates discrete feature blocks cleanly beside smooth continuous curves." + "\n" +
        r"  Resolves qualitative and numeric variables together in a single closed form.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Enforcing structured penalization matrices preserves the intrinsic geometric hierarchy of discrete qualitative sets without sacrificing optimization linearity."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "categorical_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()