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

c_primary = '#6c5ce7'
c_dark = '#2c3e50'
c_accent = '#a29bfe'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Neural Effect (NEPT)", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Neural Explicit Primal Tensorization for Compositional Deep Representations", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.85, 0.41, 0.10, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.91, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.87, "n(x1, others='x2|x3', n_neurons=500, act='relu', n_hidden_layers=2, ap=-3)", fontsize=10, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.80, "1. Explicit Primal Tensorization", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.72, 
        r"• Translates intractable infinite-width Dual kernel networks into explicit tensors." + "\n" +
        r"• Normalizes and extracts deep compositional structures into a bounded Primal block." + "\n" +
        r"  Bypasses the cubic scaling limits of Gaussian Processes to compute on millions of points.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax.text(x_left + 0.02, 0.67, r"2. Stochastic Initialization & Mapping ($\Phi$)", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_left + 0.02, 0.62, r"• n_neurons ($N_L$): Layer width | act: relu / cos / tanh | n_hidden_layers: Depth", fontsize=10, color='#7f8c8d', fontweight='bold')

draw_container(ax, x_left + 0.02, 0.38, 0.40, 0.22, edgecolor='#e2e8f0', facecolor=c_bg_card)
ax.text(x_left + 0.03, 0.56, "Stochastic Initialization (Weights & Biases):", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.52, r"$W^{(1)} \sim \mathcal{N}(0, 1) \quad W^{(l \geq 2)} \sim \mathcal{N}\left(0, \frac{1}{N_{l-1}}\right)$", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.48, r"$b^{(l)} \sim \mathcal{U}[0, 2\pi]$ (act=cos) or $\mathcal{N}(0, 1)$ (act=relu/tanh)", fontsize=10, color=c_dark)
ax.text(x_left + 0.03, 0.43, "Feature Extraction ($\Phi$) & Global Readout ($\theta$):", fontsize=10, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.39, r"$\phi_{neural}(X) = \frac{1}{\sqrt{N_L}} z^{(L)} \in \mathbb{R}^{N_{total} \times N_L} \quad \rightarrow \quad $" + "Learns " + r"$\theta \in \mathbb{R}^{N_L}$", fontsize=10, color=c_dark)

ax_ins_net = ax.inset_axes([x_left + 0.02, 0.06, 0.40, 0.30])
ax_ins_net.axis('off')
in_y = np.linspace(0.2, 0.8, 3)
h1_y = np.linspace(0.1, 0.9, 5)
hL_y = np.linspace(0.1, 0.9, 5)

for iy, y_i in enumerate(in_y):
    ax_ins_net.scatter(0.1, y_i, s=100, color=c_dark, zorder=3)
    ax_ins_net.text(0.04, y_i, f'$x_{iy+1}$', va='center', ha='right', fontsize=10, fontweight='bold')
    for y_h1 in h1_y:
        ax_ins_net.plot([0.1, 0.35], [y_i, y_h1], color='#cbd5e1', lw=0.8, alpha=0.7)

for y_h1 in h1_y:
    ax_ins_net.scatter(0.35, y_h1, s=60, color=c_primary, alpha=0.5, zorder=3)
    ax_ins_net.plot([0.35, 0.55], [y_h1, y_h1], color='#94a3b8', lw=1.5, linestyle=':')
ax_ins_net.text(0.35, 0.97, r'$\sigma(X W^{(1)} + b^{(1)})$', ha='center', fontsize=9, color=c_dark, fontweight='bold')

for y_hL in hL_y:
    ax_ins_net.scatter(0.55, y_hL, s=60, color=c_primary, zorder=3)
    ax_ins_net.plot([0.55, 0.80], [y_hL, 0.50], color=c_penalty, lw=1.5, alpha=0.8, zorder=1)
ax_ins_net.text(0.55, 0.97, r'$\sigma(Z W^{(L)} + b^{(L)})$', ha='center', fontsize=9, color=c_dark, fontweight='bold')

ax_ins_net.scatter(0.80, 0.50, s=120, color=c_penalty, zorder=3)
ax_ins_net.text(0.85, 0.50, r'$\hat{y} = \phi_{neural} \theta$', va='center', ha='left', fontsize=11, fontweight='bold', color=c_penalty)

ax_ins_net.text(0.67, 0.75, r'$\theta_1$', fontsize=9, color=c_penalty, fontweight='bold')
ax_ins_net.text(0.67, 0.25, r'$\theta_{N_L}$', fontsize=9, color=c_penalty, fontweight='bold')

rect_frozen = patches.Rectangle((0.23, 0.02), 0.42, 0.91, linewidth=1, edgecolor='#94a3b8', facecolor='#e2e8f0', alpha=0.4, zorder=2)
ax_ins_net.add_patch(rect_frozen)
ax_ins_net.text(0.44, 0.05, "FROZEN MULTI-LAYER REPS", ha='center', fontsize=7.5, fontweight='bold', color='#64748b')
ax_ins_net.set_xlim(0, 1)
ax_ins_net.set_ylim(0, 1)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.80, "3. Readout Ridge Regularization ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.74, r"$P_{neural} = \lambda I_{N_L} \quad \mathrm{with} \quad \lambda = 10^{ap}$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.65, 
        r"• Learning occurs exclusively at the readout layer via the vector $\theta \in \mathbb{R}^{N_L}$." + "\n" +
        r"• The unconstrained non-convex neural surface collapses into a quadratic form." + "\n" +
        r"• Enforces strict isotropic identity blocks to safeguard continuous regularized norms," + "\n" +
        r"  guaranteeing full operational convergence inside a valid global RKHS framework.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

ax_ins_interaction = ax.inset_axes([x_right + 0.04, 0.38, 0.36, 0.24])
x_mesh = np.linspace(-2, 2, 50)
y_mesh = np.linspace(-2, 2, 50)
X_m, Y_m = np.meshgrid(x_mesh, y_mesh)
Z_convex = X_m**2 + Y_m**2
ax_ins_interaction.contourf(X_m, Y_m, Z_convex, cmap='Purples', levels=15, alpha=0.8)
ax_ins_interaction.set_title("Strictly Convex Loss Profile (Zero Saddle Points)", fontsize=10, fontweight='bold', color=c_dark)
ax_ins_interaction.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax_ins_interaction.grid(False)

ax.text(x_right + 0.02, 0.30, "4. Eradicating Backpropagation Pathologies", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.12, 
        r"• Standard deep networks suffer from optimization stiffness and vanishing gradients." + "\n" +
        r"• TAM bypasses joint gradient step updates via static Primal matrix blocks." + "\n" +
        r"• Combines randomized feature expressivity side-by-side with continuous splines." + "\n" +
        r"  Transforms stochastic heuristic training into an exact linear algebra certainty.",
        fontsize=10.5, color='#34495e', linespacing=1.3)

msg = r"Theoretical Rule: By freezing deep random weights scaled to the NNGP prior limit, the neural optimization surface becomes strictly convex and solvable in a single algebraic pass."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "nept_neural_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.close()