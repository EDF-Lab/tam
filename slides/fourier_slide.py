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

c_primary = '#3498db'
c_dark = '#2c3e50'
c_accent = '#85c1e9'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Fourier Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Sobolev Spectral Basis for Rigid Seasonalities & Irregular Grids", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "f(x, m=10, s=2, cyclic=True, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Evolution from Complex to Real RKHS", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Legacy architectures (WeaKL) rely on complex exponentials ($e^{ikx}$) with no cyclic possibility." + "\n" +
        r"• TAM maps this into an exact, real-valued trigonometric basis ($\cos, \sin$)," + "\n" +
        r"  bypassing complex autograd limits to maximize native GPU tensor throughput.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• m: Harmonic frequencies | s: Smoothness parameter | cyclic: Boundary toggle", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.44, 0.40, 0.16, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.56, r"$\mathbf{cyclic=True}$ (Strict Periodic Boundary):", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.53, r"$\phi_{fourier}(x) = \left[ \cos(k \pi x), \sin(k \pi x) \right]_{k=1}^m$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.48, r"$\mathbf{cyclic=False}$ (Relaxed Wave Boundaries):", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.45, r"$\phi_{fourier}(x) = \left[ \cos\left(\frac{k \pi x}{2}\right), \sin\left(\frac{k \pi x}{2}\right) \right]_{k=1}^m$", fontsize=10, color=c_dark)

ax_ins_harmonic = ax.inset_axes([x_left + 0.02, 0.10, 0.40, 0.32])
x_plot = np.linspace(-1, 1, 300)
f_fundamental = np.sin(np.pi * x_plot)
f_harmonic = 0.3 * np.cos(4 * np.pi * x_plot)
f_composite = f_fundamental + f_harmonic

ax_ins_harmonic.plot(x_plot, f_fundamental, 'k--', alpha=0.3, label=r'Base wave ($k=1$)')
ax_ins_harmonic.plot(x_plot, f_harmonic, 'r--', alpha=0.3, label=r'Harmonic ($k=4$)')
ax_ins_harmonic.plot(x_plot, f_composite, color=c_primary, lw=2.5, label=r'Composite $\Phi$')
ax_ins_harmonic.set_title("Spectral Basis Assembly", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_harmonic.legend(loc='lower center', fontsize=8, frameon=True, ncol=3)
ax_ins_harmonic.tick_params(labelsize=8)
ax_ins_harmonic.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Rigorous Sobolev Roughness Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{fourier} = \lambda \cdot \mathrm{diag}\left( [1 + k^{2s}]_{k=1}^m \oplus [1 + k^{2s}]_{k=1}^m \right)$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• High-frequency wiggles ($k \gg 1$) are penalized exponentially by order $s$." + "\n" +
        r"• Mathematically eliminates overfitting to high-frequency stochastic noise." + "\n" +
        r"• Matrix is strictly positive definite: $1 + k^{2s} > 0$, guaranteeing an empty" + "\n" +
        r"  null space and ensuring perfectly stable, full-rank primal inversions.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_penalty = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
k_arr = np.arange(1, 11)
s_values = [1, 2, 3]
colors = ['#bdc3c7', '#e74c3c', '#c0392b']
for s_val, col in zip(s_values, colors):
    ax_ins_penalty.plot(k_arr, 1 + k_arr**(2*s_val), marker='o', color=col, label=f's={s_val}')
ax_ins_penalty.set_yscale('log')
ax_ins_penalty.set_title(r"Roughness Penalty Multiplier ($1 + k^{2s}$)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_penalty.set_xlabel("Harmonic Frequency Index (k)", fontsize=8)
ax_ins_penalty.set_ylabel("Penalty Weight (Log)", fontsize=8)
ax_ins_penalty.legend(fontsize=8)
ax_ins_penalty.tick_params(labelsize=8)
ax_ins_penalty.grid(True, which="both", alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Breaking the Grid Lock (FFT Avoidance)", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Classical spectral methods rely on Fast Fourier Transforms (FFT)," + "\n" +
        r"  which strictly demand equidistant grids and break down under missing data." + "\n" +
        r"• TAM evaluates continuous trigonometric tensors directly on continuous, irregular x." + "\n" +
        r"• The diagonal structure of $P_{fourier}$ introduces zero memory allocation overhead." + "\n" +
        r"  Ultra-expressive dictionaries ($m > 100$) collapse into an $\mathcal{O}(ND^2)$ certainty.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Scalar multiplier lambda ($\lambda = 10^{ap}$) controls global regularization. Higher Sobolev orders (s) protect boundaries from Gibbs ringing."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "fourier_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()