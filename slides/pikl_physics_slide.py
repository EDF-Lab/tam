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

c_primary = '#f1c40f'
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

ax.text(0.04, 0.94, "The Physics Effect (PIKL)", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Physics-Informed Kernel Learning & Exact Analytical Constraints", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "phys(t, basis='fourier', D0=k, D1=c, D2=m, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.81, "1. The Optimization Failure of PINNs", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.77, 
        r"• Standard PINNs rely on gradient descent to balance empirical loss with a PDE residual." + "\n" +
        r"• This creates notoriously pathological optimization landscapes (gradient stiffness)." + "\n" +
        r"• TAM abandons gradient descent, formatting the continuous differential equation" + "\n" +
        r"  strictly as an exact, convex penalty mapped directly into the Primal space.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax.text(x_left + 0.02, 0.63, r"2. Feature Map Equivalence ($\Phi_{phys} = \Phi_{base}$)", fontsize=16, fontweight='bold', color=c_dark)

draw_container(ax, x_left + 0.02, 0.41, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.54, "The Capacity vs. Rules Separation:", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.49, r"The physics effect does not require a novel feature map. The basis (Spline,", fontsize=11, color=c_dark)
ax.text(x_left + 0.03, 0.45, r"Fourier, Neural) provides the capacity to fit the data ($\Phi$), while the", fontsize=11, color=c_dark)
ax.text(x_left + 0.03, 0.41, r"stiffness matrix ($P_{phys}$) defines the strict physical rules it must obey.", fontsize=11, color=c_dark)

ax_ins_signal = ax.inset_axes([x_left + 0.03, 0.08, 0.38, 0.28])
t = np.linspace(0, 10, 200)
true_signal = np.exp(-0.2 * t) * np.cos(2 * t)
noisy_data = true_signal + np.random.normal(0, 0.15, size=t.shape)

ax_ins_signal.scatter(t[::4], noisy_data[::4], color='#bdc3c7', s=15, label='Noisy Sensors')
ax_ins_signal.plot(t, true_signal, color=c_primary, lw=2.5, label='Exact PIKL Resolution')
ax_ins_signal.set_title("Damped Harmonic Oscillator Extraction", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_signal.legend(loc='upper right', fontsize=8, frameon=True)
ax_ins_signal.set_xticks([])
ax_ins_signal.set_yticks([])

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.81, "3. The Physical Stiffness Matrix ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.77, r"$\int |\mathcal{L}(f)|^2 dt \approx \theta^\top P_{phys} \theta \quad \rightarrow \quad P_{phys} = \lambda L_{op}^\top L_{op}$", fontsize=13, color=c_dark)
ax.text(x_right + 0.02, 0.71, 
        r"• The continuous integral of the physical residual reduces analytically to a quadratic form." + "\n" +
        r"• By replacing the standard statistical smoothing penalty with $P_{phys}$, the solver" + "\n" +
        r"  forces the Primal coefficients into the manifold defined by the differential equation.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax_ins_resonance = ax.inset_axes([x_right + 0.04, 0.39, 0.36, 0.20])
omega = np.linspace(0, 4, 200)
m, c, k = 1.0, 0.3, 4.0
penalty_curve = (-m * omega**2 + k)**2 + (c * omega)**2

ax_ins_resonance.plot(omega, penalty_curve, color=c_penalty, lw=2)
ax_ins_resonance.axvline(2.0, color=c_dark, linestyle='--', lw=1.5)
ax_ins_resonance.text(2.1, 10, r"$\omega_0 = \sqrt{k/m}$", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_resonance.scatter([2.0], [c**2 * 4], color=c_dark, s=50, zorder=5)

ax_ins_resonance.set_title(r"Fourier Spectral Penalty Profile: $P_{jj}(\omega_j)$", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_resonance.set_xlabel(r"Frequency ($\omega$)", fontsize=8)
ax_ins_resonance.set_ylabel("Penalty Weight", fontsize=8)
ax_ins_resonance.set_xticks([])
ax_ins_resonance.set_yticks([])

ax.text(x_right + 0.02, 0.34, "4. Basis-Specific Matrix Translations", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.30, 
        r"• Fourier (Spectral Diagonalization): Complex exponentials are eigenfunctions." + "\n" +
        r"  $P_{phys}$ is purely diagonal. Penalty approaches zero at the physical resonant frequency." + "\n" +
        r"• Splines (Discrete Difference): Maps continuous derivatives to discrete $\Delta$ matrices," + "\n" +
        r"  creating a highly banded, sparse positive semi-definite Gramian matrix." + "\n" +
        r"• Neural (Autograd Gramian): For deep NNGPs, TAM analytically pre-computes" + "\n" +
        r"  the exact Jacobian via PyTorch autograd over the empirical grid, freezing the geometry.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

msg = r"Theoretical Rule: By replacing statistical smoothing with an analytical stiffness matrix, the convex solver strictly forces the hypothesis into the exact manifold defined by the physical PDE."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "pikl_physics_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()