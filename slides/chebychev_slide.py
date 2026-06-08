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

c_primary = '#3f51b5'
c_dark = '#2c3e50'
c_accent = "#7007c5"
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Chebyshev Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Orthogonal Minimax Polynomials & Recurrent Stable Tensorization", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.85, 0.41, 0.10, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.91, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.87, "p(x, deg=10, s=2.0, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Runge's Phenomenon & Minimax Approximation", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Standard monomial bases ($x^n$) create ill-conditioned Vandermonde matrices." + "\n" +
        r"• Uniform polynomial interpolation systematically triggers Runge's phenomenon." + "\n" +
        r"• TAM projects into orthogonal Chebyshev polynomials of the first kind ($T_n(x)$)," + "\n" +
        r"  guaranteeing near-optimal minimax approximation without boundary oscillation.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.65, r"2. Primal Mapping Function ($\Phi$) & Constraints", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.60, r"• deg: Maximum polynomial degree ($D$) | s: Smoothness filter parameter", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.41, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.54, r"Exact Three-Term Recurrence Relation:", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.50, r"$T_0(x) = 1, \quad T_1(x) = x, \quad T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.45, r"Explicit Primal Feature Mapping ($\Phi$):", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.42, r"$\phi_{chebyshev}(x) = \left[ T_n(x) \right]_{n=1}^D$", fontsize=10, color=c_dark)

ax_ins_signal = ax.inset_axes([x_left + 0.02, 0.07, 0.40, 0.32])
x_plot = np.linspace(-1, 1, 200)
T1 = x_plot
T2 = 2 * x_plot**2 - 1
T3 = 4 * x_plot**3 - 3 * x_plot
T4 = 8 * x_plot**4 - 8 * x_plot**2 + 1

ax_ins_signal.plot(x_plot, T1, color='#bdc3c7', lw=1.5, linestyle='--', label=r'$T_1(x)$')
ax_ins_signal.plot(x_plot, T2, color=c_primary, lw=2.0, alpha=0.7, label=r'$T_2(x)$')
ax_ins_signal.plot(x_plot, T3, color=c_accent, lw=2.0, alpha=0.9, label=r'$T_3(x)$')
ax_ins_signal.plot(x_plot, T4, color=c_dark, lw=2.0, label=r'$T_4(x)$')
ax_ins_signal.set_title("Chebyshev Minimax Orthogonal Basis", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_signal.legend(loc='lower right', fontsize=7.5, frameon=True, ncol=2)
ax_ins_signal.tick_params(labelsize=8)
ax_ins_signal.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Degree-Dependent Sobolev Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{chebyshev} = \lambda \cdot \mathrm{diag}\left( n^{2s} \right)_{n=1}^D$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• High-degree polynomials are highly prone to overfitting empirical noise." + "\n" +
        r"• TAM applies a strictly diagonal, degree-dependent geometric penalty multiplier." + "\n" +
        r"• Operationally acts as a low-pass filter, heavily penalizing high-frequency" + "\n" +
        r"  wiggles in favor of smooth, structurally stable secular macro-trends.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_zplane = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
n_degrees = np.arange(1, 16)
p_s1 = n_degrees**(2 * 1.0)
p_s2 = n_degrees**(2 * 1.5)
p_s3 = n_degrees**(2 * 2.0)

ax_ins_zplane.plot(n_degrees, p_s1, color='#bdc3c7', marker='o', lw=1.5, label='s = 1.0')
ax_ins_zplane.plot(n_degrees, p_s2, color=c_accent, marker='s', lw=2.0, label='s = 1.5')
ax_ins_zplane.plot(n_degrees, p_s3, color=c_penalty, marker='^', lw=2.5, label='s = 2.0')
ax_ins_zplane.set_yscale('log')
ax_ins_zplane.set_title(r"Sobolev Penalty Low-Pass Filter ($n^{2s}$)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_zplane.set_xlabel("Polynomial Degree (n)", fontsize=8)
ax_ins_zplane.set_ylabel("Penalty Multiplier (Log)", fontsize=8)
ax_ins_zplane.legend(loc='lower right', fontsize=7.5, frameon=True)
ax_ins_zplane.tick_params(labelsize=8)
ax_ins_zplane.grid(True, which='both', alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Tensor Stability & RKHS Eligibility", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• The input $x$ is affinely transformed to $[-1, 1]$ to preserve orthogonality." + "\n" +
        r"• By computing the recursion entirely in-place on GPU tensors, the engine bypasses" + "\n" +
        r"  dense intermediate matrices and catastrophic floating-point cancellation." + "\n" +
        r"• Because $n^{2s} > 0$ strictly for $n \geq 1$, the matrix is positive definite," + "\n" +
        r"  satisfying Mercer's conditions for a valid Reproducing Kernel Hilbert Space.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Recurrent tensorization avoids floating-point collapse, while Sobolev geometric scaling filters high-frequency polynomial variance."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "chebyshev_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()