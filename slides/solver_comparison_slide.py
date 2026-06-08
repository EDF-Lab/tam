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

c_dark = '#2c3e50'
c_mgcv = '#e67e22'
c_tam = '#27ae60'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "StaticTAM: The Primal Solver", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Algorithmic Architecture: mgcv vs. tam", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.46, 0.85, 0.50, 0.11, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.48, 0.93, "The core TAM innovation lies in the 14 Mathematical Effects compared to mgcv & WeaKL", fontsize=9.5, color='#95a5a6', fontweight='bold')
ax.text(0.48, 0.90, "Offset • Linear • Categorical • Chebyshev • Fourier • Splines • Wavelets", fontsize=10.5, color='#f1c40f', fontweight='bold')
ax.text(0.48, 0.88, "NEPT (Neural) • Trees • Linear Trees • Tensor • RBF • PID • PIKL (Physics)", fontsize=10.5, color='#f1c40f', fontweight='bold')

draw_container(ax, 0.04, 0.685, 0.92, 0.16, edgecolor=c_dark, facecolor=c_bg_card)
ax.text(0.06, 0.81, r"The Unified Goal: Solve $Y = \mu(X) + \epsilon$", fontsize=16, fontweight='bold', color=c_dark)
ax.text(0.06, 0.76, r"$Y$: Target $\quad|\quad X$: Input Covariates $\quad|\quad \mu(\cdot)$: Prediction Function $\quad|\quad \epsilon$: Noise", fontsize=12, color='#34495e')
ax.text(0.06, 0.71, r"$\mu(X) = \sum_{l=1}^L h_l(X) \quad \mathrm{with} \quad h_l(X) = \sum_{i=1}^{I_l} \phi_{l,i}(X) \theta_{l,i} \quad \mathrm{and} \quad D = \sum_{l=1}^L I_l$", fontsize=12, color=c_dark)
ax.text(0.55, 0.81, "The Primal Projection:", fontsize=12, fontweight='bold', color=c_dark)
ax.text(0.55, 0.77, 
        r"• Bypass the $\mathcal{O}(N^3)$ computational trap of full kernel methods." + "\n" +
        r"• Project the functional problem onto a finite primal basis of $D$ linear" + "\n" +
        r"  parameters, where $N$ represents the number of observations.", 
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

x_left = 0.04
draw_container(ax, x_left, 0.08, 0.44, 0.60, edgecolor=c_mgcv)

ax.text(x_left + 0.02, 0.64, "mgcv (Probabilistic & Iterative)", fontsize=18, fontweight='bold', color=c_mgcv)
ax.text(x_left + 0.30, 0.64, r"$\mathcal{O}(I \times (ND^2 + D^3))$", fontsize=14, fontweight='bold', color=c_mgcv)

ax.text(x_left + 0.02, 0.58, "Objective (Deviance & Penalty):", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.52, r"$\min_{\theta} D(Y, \Phi\theta) + \|M\theta\|_2^2$", fontsize=15, color=c_dark)

ax.text(x_left + 0.02, 0.45, "I steps of Iterative PIRLS:", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.39, r"$\hat{\theta}^{(t+1)} = (\Phi^\top W^{(t)} \Phi + nP)^{-1} \Phi^\top W^{(t)} z^{(t)}$", fontsize=14, color=c_dark)

ax.text(x_left + 0.02, 0.32, "Pseudo-data ($z$) & Weights ($W$) updates:", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.28, r"$g(\mu) = \eta \quad \mathrm{where} \quad \eta = \Phi\theta$", fontsize=11, color=c_dark)
ax.text(x_left + 0.02, 0.24, r"$z^{(t)} = \eta^{(t)} + (Y - \mu^{(t)}) g'(\mu^{(t)})$", fontsize=12, color=c_dark)
ax.text(x_left + 0.02, 0.19, r"$W_{ii}^{(t)} = 1 / \left( \mathrm{Var}(Y_i) [g'(\mu_i^{(t)})]^2 \right)$", fontsize=12, color=c_dark)

draw_container(ax, x_left + 0.22, 0.09, 0.20, 0.07, edgecolor='#bdc3c7', facecolor=c_bg_card)
ax.text(x_left + 0.23, 0.135, "Simon N. Wood", fontsize=8, color='#7f8c8d', fontweight='bold')
ax.text(x_left + 0.23, 0.105, "Generalized additive models. CRC Press, 2017.", fontsize=8, color='#7f8c8d')

x_right = 0.52
draw_container(ax, x_right, 0.08, 0.44, 0.60, edgecolor=c_tam)

ax.text(x_right + 0.02, 0.64, "tam (Algebraic & Scalable)", fontsize=18, fontweight='bold', color=c_tam)
ax.text(x_right + 0.31, 0.64, r"$\mathcal{O}(ND^2 + D^3)$", fontsize=14, fontweight='bold', color=c_tam)

ax.text(x_right + 0.02, 0.58, "Objective (L2 & Penalty):", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.51, r"$\min_{\theta} \frac{1}{N} \sum_{j=1}^N \|\Lambda (\Phi_j \theta - Y_j)\|_2^2 + \|M\theta\|_2^2$", fontsize=15, color=c_dark)

ax.text(x_right + 0.02, 0.44, "Exact Single-Step Resolution:", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.37, r"$\hat{\theta} = (\Phi^\top \Lambda^\top \Lambda \Phi + nP)^{-1} \Phi^\top \Lambda^\top \Lambda Y$", fontsize=15, color=c_tam, fontweight='bold')

ax.text(x_right + 0.02, 0.30, "Temporal Masking & Group-Chunking:", fontsize=11, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.26, r"• $\Lambda$ acts as an exact isolation mask.", fontsize=11.5, color=c_dark)
ax.text(x_right + 0.02, 0.22, r"• Drives the hardware Group-Chunking approach.", fontsize=11.5, color=c_dark)
ax.text(x_right + 0.02, 0.18, r"• Safely pads asynchronous/missing sequences.", fontsize=11.5, color=c_dark)

draw_container(ax, x_right + 0.02, 0.085, 0.40, 0.085, edgecolor='#e2e8f0', facecolor='#eefaf3')
ax.text(x_right + 0.03, 0.15, '"The strength of WeaKL lies in its exact computation [...]', fontsize=8.5, color=c_dark, style='italic')
ax.text(x_right + 0.03, 0.125, 'it is free from optimization errors. It relies solely on linear algebra, taking advantage of GPU programming [...]', fontsize=8.5, color=c_dark, style='italic')
ax.text(x_right + 0.03, 0.10, "N. Doumèche and F. Bach and E. Bedek and G. Biau, C. Boyer, Y. Goude (arXiv:2502.10485)", fontsize=8, color='#7f8c8d', fontweight='bold')

ax.text(0.5, 0.04, "Conclusion: A Complementary Approach", ha='center', fontsize=14, fontweight='bold', color=c_dark)
ax.text(0.5, 0.01, "mgcv provides superior probabilistic comprehension for complex distributions, while tam delivers massive GPU scaling for Gaussian Gigadata.", ha='center', fontsize=11, color='#34495e')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "solver_comparison_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()