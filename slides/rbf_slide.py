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

c_primary = '#ff7675'
c_dark = '#2c3e50'
c_accent = '#fab1a0'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Radial Basis Function (RBF) Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Centroid-Based Isotropic Kernels & The Matérn Screening Effect", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.85, 0.41, 0.10, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.91, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.87, "rbf(x, n_centers=50, gamma=0.1, nu=1.5, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Centroid Prototype Projection", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Standard Gaussian Processes evaluate all $N$ pairs, creating an $\mathcal{O}(N^3)$ bottleneck." + "\n" +
        r"• TAM explicitly truncates this by evaluating distance-based similarity against a fixed" + "\n" +
        r"  set of $M$ strategic prototypes, projecting a finite $\mathcal{O}(NM^2)$ Primal block.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping ($\Phi$) & Kernel Selection", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• n_centers (M): Fixed prototypes | gamma ($\gamma$): Length-scale | nu ($\nu$): Smoothness", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.43, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.56, r"Explicit Primal Feature Mapping:", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.52, r"$\phi_{rbf}(x) = \left[ K(x, c_1), \dots, K(x, c_M) \right]^T$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.47, r"Gaussian (Squared Exp) vs. Matérn Kernel Families:", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.43, r"$K_{Gauss}(x, c) = \exp(-\gamma ||x - c||_2^2) \quad \leftrightarrow \quad K_{Matern}(x, c)$", fontsize=10, color=c_dark)

ax_ins_kernels = ax.inset_axes([x_left + 0.02, 0.07, 0.40, 0.33])
dist = np.linspace(-3, 3, 200)
k_gauss = np.exp(-1.0 * dist**2)
d_scaled = np.sqrt(3) * np.abs(dist)
k_matern_15 = (1 + d_scaled) * np.exp(-d_scaled)
k_matern_05 = np.exp(-np.abs(dist))

ax_ins_kernels.plot(dist, k_gauss, color=c_dark, lw=2.5, linestyle='--', label=r'Gaussian (Infinitely Smooth)')
ax_ins_kernels.plot(dist, k_matern_15, color=c_primary, lw=2, label=r'Matérn ($\nu=1.5$, Once Differentiable)')
ax_ins_kernels.plot(dist, k_matern_05, color=c_accent, lw=1.5, alpha=0.8, label=r'Matérn ($\nu=0.5$, Rough/Chaotic)')
ax_ins_kernels.set_title("Kernel Spatial Correlation Profiles", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_kernels.legend(loc='upper right', fontsize=7.5, frameon=True)
ax_ins_kernels.tick_params(labelsize=8)
ax_ins_kernels.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. The Screening Effect & Structural Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{rbf} = \lambda I_M$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• The Gaussian kernel's spectral density decays exponentially, violating the" + "\n" +
        r"  mathematical conditions required for localized prototype screening." + "\n" +
        r"• The Matérn family's algebraic decay mathematically guarantees the Screening Effect," + "\n" +
        r"  validating sparse centroid truncation without severe approximation collapse.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_spectral = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
omega = np.logspace(0, 1.5, 100)
spec_gauss = np.exp(-0.5 * omega**2)
spec_matern = (1 + omega**2)**(-2.0)

ax_ins_spectral.plot(omega, spec_gauss, color=c_dark, lw=2, linestyle='--', label='Gaussian (Exponential Decay)')
ax_ins_spectral.plot(omega, spec_matern, color=c_penalty, lw=2.5, label='Matérn (Algebraic Decay)')
ax_ins_spectral.set_xscale('log')
ax_ins_spectral.set_yscale('log')
ax_ins_spectral.set_ylim(1e-10, 2)
ax_ins_spectral.set_title("Spectral Density Decay (Frequency Domain)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_spectral.set_xlabel(r"Frequency ($\omega$)", fontsize=8)
ax_ins_spectral.set_ylabel("Power", fontsize=8)
ax_ins_spectral.legend(loc='lower left', fontsize=7.5, frameon=True)
ax_ins_spectral.tick_params(labelsize=8)
ax_ins_spectral.grid(True, which='both', alpha=0.2)

ax.text(x_right + 0.02, 0.30, "4. Eradicating Non-Convex Optimization", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Classical RBF networks learn centroid locations and bandwidths via gradient descent," + "\n" +
        r"  creating an unstable loss landscape heavily prone to pathological saddle points." + "\n" +
        r"• TAM fixes the prototypes geographically and computes pairwise Euclidean distances" + "\n" +
        r"  natively on the GPU. The RBF network is flattened into a strictly convex projection.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: A valid Matérn projection evaluated over a strictly regularized finite RKHS ($P = \lambda I_M$) guarantees global optimality in a single algebraic step."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "rbf_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()