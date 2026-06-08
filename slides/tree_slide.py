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

c_primary = '#16a085'
c_dark = '#2c3e50'
c_accent = '#1abc9c'
c_penalty = '#c0392b'
c_bg_card = '#f8f9fa'

def draw_container(ax_parent, x, y, width, height, edgecolor, facecolor='#ffffff'):
    box = patches.FancyBboxPatch(
        (x, y), width, height, 
        boxstyle="round,pad=0.0,rounding_size=0.015", 
        linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=1
    )
    ax_parent.add_patch(box)

ax.text(0.04, 0.94, "The Tree Effect", fontsize=28, fontweight='bold', color=c_dark, va='top')
ax.text(0.04, 0.89, "Oblivious Random Forests & Smooth Kernel Convergence", fontsize=14, color='#7f8c8d', va='top')

draw_container(ax, 0.55, 0.86, 0.41, 0.09, edgecolor='#bdc3c7', facecolor='#2c3e50')
ax.text(0.57, 0.92, "Declarative Formula API Syntax:", fontsize=10, color='#95a5a6', family='sans-serif', fontweight='bold')
ax.text(0.57, 0.88, "t(x1, others='x2|x3', n_trees=200, max_depth=6, ap=-3)", fontsize=13, color='#f1c40f', family='monospace', fontweight='bold')

x_left = 0.04
draw_container(ax, x_left, 0.05, 0.44, 0.78, edgecolor=c_primary)

ax.text(x_left + 0.02, 0.81, "1. The Random Engine", fontsize=16, fontweight='bold', color=c_primary)
ax.text(x_left + 0.02, 0.77, 
        r"• Cuts are blindly generated via PyTorch's uniform distribution $\mathcal{U}[X_{min}, X_{max}]$." + "\n" +
        r"• Every single tree receives a completely unique, independent array of random cuts." + "\n" +
        r"• Oblivious Trees (symmetric splits) eliminate GPU thread divergence.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax.text(x_left + 0.02, 0.65, r"2. Feature Map ($\Phi$) & Notation Translation", fontsize=16, fontweight='bold', color=c_dark)

draw_container(ax, x_left + 0.02, 0.38, 0.40, 0.25, edgecolor='#e2e8f0', facecolor=c_bg_card)

ax.text(x_left + 0.03, 0.60, r"Sparse Indicator Bit-String ($\phi$):", fontsize=9.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.56, r"$\phi_{tree}(x) = \frac{1}{\sqrt{B}} \left[ I(x \in R_{1,1}), \dots, I(x \in R_{B,M_B}) \right]^\top$", fontsize=11, color=c_dark)
ax.text(x_left + 0.03, 0.51, "Global Tensor Assembly & Shape:", fontsize=9.5, fontweight='bold', color=c_dark)
ax.text(x_left + 0.03, 0.47, r"$\Phi_{tree} = \frac{1}{\sqrt{B}} \left[ \Phi^{(1)} \mid \Phi^{(2)} \mid \dots \mid \Phi^{(B)} \right] \in \mathbb{R}^{N \times (B \times L)}$", fontsize=11, color=c_dark)

ax.text(x_left + 0.03, 0.42, "N: Dataset Rows", fontsize=9.5, fontweight='bold', color='#c0392b')
ax.text(x_left + 0.16, 0.42, "B: n_trees", fontsize=9.5, fontweight='bold', color='#c0392b')
ax.text(x_left + 0.26, 0.42, r"M_B: Terminal Leaf Index ($M_B = L$)", fontsize=9.5, fontweight='bold', color='#c0392b')
ax.text(x_left + 0.03, 0.39, r"L: Leaves per tree $\rightarrow$ strictly defined as $2^{max\_depth}$ or max_leaves", fontsize=9.5, fontweight='bold', color='#c0392b')

ax_1tree = ax.inset_axes([x_left + 0.03, 0.07, 0.17, 0.28])
ax_1tree.set_xlim(0, 10)
ax_1tree.set_ylim(-1.5, 1.5)
ax_1tree.set_xticks([])
ax_1tree.set_yticks([])
x_plot = np.linspace(0, 10, 200)
y_true = np.sin(x_plot)
ax_1tree.plot(x_plot, y_true, color='#bdc3c7', alpha=0.4, lw=2)
ax_1tree.step([0, 2, 3.5, 6, 8.5, 10], [0, 0.9, -0.3, -0.9, 0.7, 0], where='post', color=c_penalty, lw=2)
ax_1tree.set_title("1 Tree: Rigid Staircase", fontsize=10, fontweight='bold', color=c_dark)

ax_200tree = ax.inset_axes([x_left + 0.25, 0.07, 0.17, 0.28])
ax_200tree.set_xlim(0, 10)
ax_200tree.set_ylim(-1.5, 1.5)
ax_200tree.set_xticks([])
ax_200tree.set_yticks([])
ax_200tree.plot(x_plot, y_true, color='#bdc3c7', alpha=0.4, lw=2)
noise_curve = y_true + np.random.normal(0, 0.01, size=200)
ax_200tree.plot(x_plot, noise_curve, color=c_primary, lw=2.5)
ax_200tree.set_title("200 Trees: Smooth Kernel", fontsize=10, fontweight='bold', color=c_dark)

x_right = 0.52
draw_container(ax, x_right, 0.05, 0.44, 0.78, edgecolor=c_penalty)

ax.text(x_right + 0.02, 0.81, "3. Isotropic Ridge Regularization ($P$)", fontsize=16, fontweight='bold', color=c_penalty)
ax.text(x_right + 0.02, 0.77, r"$P_{tree} = \lambda I_{(B \times L)}$", fontsize=14, color=c_dark)
ax.text(x_right + 0.02, 0.73, 
        r"• Since leaf assignments function as discrete, non-ordinal categorical bins, they are" + "\n" +
        r"  optimally bounded by an isotropic $L_2$ shrinkage block across all features." + "\n" +
        r"• Strictly limits the statistical weight of deep, isolated regions (variance control).",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

ax_ins_sparse = ax.inset_axes([x_right + 0.04, 0.39, 0.36, 0.21])
ax_ins_sparse.axis('off')

b_size = 5
n_blocks = 4
sparse_mat = np.zeros((b_size*n_blocks, b_size*n_blocks))
for i in range(n_blocks):
    start = i * b_size
    for j in range(b_size):
        sparse_mat[start+j, start+j] = 1

ax_ins_sparse.spy(sparse_mat, markersize=4, color=c_penalty)
ax_ins_sparse.set_title(r"Sparse COO Identity Tensor ($\lambda I$)", fontsize=10, fontweight='bold', color=c_dark)

for i in range(n_blocks):
    rect = patches.Rectangle((i*b_size - 0.5, i*b_size - 0.5), b_size, b_size, linewidth=1, edgecolor='#bdc3c7', facecolor='none', linestyle='--')
    ax_ins_sparse.add_patch(rect)

ax.text(x_right + 0.02, 0.34, "4. Anti-Starvation & Memory Safety", fontsize=16, fontweight='bold', color=c_dark)
ax.text(x_right + 0.02, 0.30, 
        r"• Anti-Starvation: For n_trees=1, purely random cuts risk microscopic 0-data" + "\n" +
        r"  gaps. TAM bypasses randomness here, forcing evenly spaced `torch.linspace`." + "\n" +
        r"• Kernel: Combining 200 trees averages the random bins into a smooth continuous" + "\n" +
        r"  function. Evaluates the exact probability that $x$ and $x'$ share a bin." + "\n" +
        r"• Memory: To prevent GPU VRAM exhaustion, $P_{tree}$ is strictly materialized" + "\n" +
        r"  as a `torch.sparse_coo_tensor`, injecting millions of constraints at 0 memory cost.",
        fontsize=10.5, color='#34495e', linespacing=1.4, va='top')

msg = r"Theoretical Rule: A single randomized tree generates a rigid block step-function. Combining hundreds of randomized trees mathematically converges into a smooth continuous kernel."
ax.text(0.5, 0.01, msg, ha='center', va='bottom', fontsize=11, style='italic', color='#ffffff',
        bbox=dict(boxstyle="round,pad=0.4", fc=c_dark, ec="#1a252f", lw=1))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

output_path = os.path.join(output_folder, "tree_effect_slide.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Patched slide visual saved seamlessly at: {output_path}")

plt.close()