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

c_primary = '#1abc9c'
c_dark = '#2c3e50'
c_accent = "#13ca32"
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Wavelet Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Continuous Wavelet Dictionary for Local Transients & Structural Shocks", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "w(x, n_scales=5, n_locations=20, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Time-Frequency Localization", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Global Fourier harmonics struggle with non-stationary singularities and anomalies." + "\n" +
        r"• TAM builds a continuous dictionary using localized wavelets with compact support." + "\n" +
        r"  This isolates sudden structural breaks without inducing distant Gibbs ringing.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Primal Mapping Function ($\Phi$) & Hyperparameters", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• n_scales (a): Log-dilations | n_locations (b): Linear translation centers", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.43, 0.40, 0.17, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.57, r"$\mathbf{Ricker\ Mother\ Wavelet:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.53, r"$\psi(t) = (1 - t^2) e^{-t^2 / 2} \quad \mathrm{with} \quad t = \frac{x - b_j}{a_i}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.48, r"$\mathbf{Explicit\ L_2\ Scaled\ Feature\ Map:}$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.44, r"$\phi_{wavelet}(x) = \left[ \frac{1}{\sqrt{a_i}} \psi\left(\frac{x - b_j}{a_i}\right) \right]_{i=1, \dots, n_{scales}}^{j=1, \dots, n_{locations}}$", fontsize=10, color=c_dark)

ax_ins_harmonic = ax.inset_axes([x_left + 0.02, 0.08, 0.40, 0.32])
x_plot = np.linspace(-1, 1, 400)
def ricker_func(x, a, b):
    t = (x - b) / a
    return (1.0 - t**2) * np.exp(-t**2 / 2.0) / np.sqrt(a)

y_wide = ricker_func(x_plot, 0.4, -0.2)
y_narrow = ricker_func(x_plot, 0.12, 0.4)

ax_ins_harmonic.plot(x_plot, y_wide, color=c_primary, lw=2, label=r'Low-Freq (a=0.4, b=-0.2)')
ax_ins_harmonic.plot(x_plot, y_narrow, color=c_accent, lw=2, label=r'High-Freq (a=0.12, b=0.4)')
ax_ins_harmonic.set_title("Time-Frequency Dictionary Components", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_harmonic.legend(loc='upper left', fontsize=7.5, frameon=True)
ax_ins_harmonic.tick_params(labelsize=8)
ax_ins_harmonic.grid(True, alpha=0.3)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Scale-Dependent Structural Penalty ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{wavelet} = \lambda \cdot \mathrm{diag}\left( \frac{1}{a_i^2} \right)$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• Employs a continuous algebraic proxy to match Oracle Wavelet Shrinkage." + "\n" +
        r"• Multiplier is inversely proportional to squared scale, penalizing fine noise." + "\n" +
        r"• Positive definite diagonal isolates null space to safeguard global convexity," + "\n" +
        r"  allowing safe concurrent optimization alongside global polynomials and splines.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_penalty = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
scales_arr = np.logspace(-2, 0, 50)
penalty_val = 1.0 / (scales_arr**2)

ax_ins_penalty.plot(scales_arr, penalty_val, color=c_penalty, lw=2.5)
ax_ins_penalty.set_xscale('log')
ax_ins_penalty.set_yscale('log')
ax_ins_penalty.set_title(r"Shrinkage Penalty Blow-up ($1 / a_i^2$)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_penalty.set_xlabel("Wavelet Dilation Scale (a)", fontsize=8)
ax_ins_penalty.set_ylabel("Penalty Magnitude", fontsize=8)
ax_ins_penalty.tick_params(labelsize=8)
ax_ins_penalty.grid(True, which="both", alpha=0.3)

ax.text(x_right + 0.02, 0.30, "4. Continuous Geometry over Discrete Pipelines", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Traditional setups rely on complex, multi-stage discrete filtering pipelines." + "\n" +
        r"• TAM evaluates continuous analytic wavelets directly on the input tensor manifold." + "\n" +
        r"• High-frequency features light up exclusively when backed by loss gradients." + "\n" +
        r"  Bypasses sequential sub-band filters to solve system in one algebraic pass.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: Constant spectral energy is maintained via the $1/\sqrt{a_i}$ factor. Scale-dependent penalties suppress high-frequency stochastic noise."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "wavelet_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()