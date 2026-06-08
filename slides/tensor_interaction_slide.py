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

c_primary = '#d35400'
c_dark = '#2c3e50'
c_accent = '#e67e22'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Tensor Product Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Multi-Dimensional Feature Interactions & Anisotropic Scale-Invariant Smoothing Space", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "te(s(x1, k=10, ap=-5), f(x2, m=5, ap=-10), ap=-30)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Space Fusion & Kronecker Mapping", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Fuses independent marginal spaces to model complex multivariate response surfaces." + "\n" +
        r"• Creates a joint hypothesis space via row-wise Kronecker product operations." + "\n" +
        r"  Allows heterogeneous continuous smoothers to interact explicitly inside the Primal space.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• marginal bases: s(), f(), l() or n() | ap: Global interaction penalty multiplier", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.41, 0.40, 0.19, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.57, r"$\mathbf{Marginal\ Basis\ Vectors\ Definition:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.54, r"$\phi_1 = [h_{1,1}, \dots, h_{1,D_1}] \quad \phi_2 = [h_{2,1}, \dots, h_{2,D_2}]$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.50, r"$\mathbf{Kronecker\ Tensor\ Product\ Development\ (\otimes):}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.46, r"$\phi_{cross}(x_1, x_2) = \phi_1(x_1) \otimes \phi_2(x_2)$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.43, r"$= [h_{1,1}h_{2,1}, \ h_{1,1}h_{2,2}, \ \dots, \ h_{1,D_1}h_{2,D_2}]$", fontsize=10, color=c_dark)

ax_ins_interaction = ax.inset_axes([x_left + 0.02, 0.08, 0.40, 0.30])
x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
Z_surface = (1.0 - X_mesh**2) * np.sin(np.pi * Y_mesh)
contour = ax_ins_interaction.contourf(X_mesh, Y_mesh, Z_surface, cmap='GnBu', levels=15, alpha=0.8)
ax_ins_interaction.set_title("Fused B-Spline × Fourier Surface $\Phi_{cross}$", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_interaction.tick_params(labelsize=8)
ax_ins_interaction.grid(True, alpha=0.2)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Anisotropic Scale-Invariant Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{cross} = \lambda \sum_{i=1}^{K_{te}} \left( I_1 \otimes \dots \otimes P_i \otimes \dots \otimes I_{K_{te}} \right)$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.69, r"Note: $P_i = \lambda_i P_{effect_i}$ (Marginal penalties arrive pre-scaled via internal 'ap')", fontsize=9.5, color=c_primary, style='italic')
ax.text(x_right + 0.02, 0.60, 
        r"• Avoids structural failure of thin-plate splines on heterogeneous physical units." + "\n" +
        r"• Constructs directional penalty blocks to independently constrain coordinate roughness." + "\n" +
        r"• Satisfies Mercer conditions for a valid Reproducing Kernel Hilbert Space (RKHS)," + "\n" +
        r"  ensuring full compatibility with the Generalized Representer Theorem framework.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_sparsity = ax.inset_axes([x_right + 0.04, 0.34, 0.32, 0.20])
mock_matrix = np.zeros((12, 12))
for b in range(3):
    mock_matrix[b*4:(b+1)*4, b*4:(b+1)*4] = 1.0
for i in range(12):
    mock_matrix[i, i] = 2.0
ax_ins_sparsity.imshow(mock_matrix, cmap='Reds', interpolation='none', alpha=0.8)
ax_ins_sparsity.set_title(r"Anisotropic Penalty Structure $(D_1 \cdot D_2)^2$", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_sparsity.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax_ins_sparsity.grid(False)

ax.text(x_right + 0.02, 0.24, "4. Structural Safety & Fallback Guarantees", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Multiplicative coefficient expansions typically exhaust physical VRAM profiles." + "\n" +
        r"• TAM bypasses direct matrix storage via the matrix-free Sparse CG solver." + "\n" +
        r"• The anisotropic penalty dominates rank-deficient data-starved local spaces." + "\n" +
        r"  Unstable gradients shrink exactly to zero, safely falling back on global trends.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Pointwise multiplication of valid reproducing kernels preserves global convexity while anisotropic smoothing enforces structural safety."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "tensor_interaction_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()