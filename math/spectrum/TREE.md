# The Tree Effect (Random Forests & Binning Features)

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

## Formula Definition

Classical continuous bases (such as polynomials or Fourier harmonics) often struggle to capture sharp, irregular, and highly non-linear threshold effects. To capture these discontinuous jump structures, the input space $\mathcal{X}$ is recursively partitioned into a set of $M$ disjoint terminal regions (leaves) {cite:p}`breiman2017classification`.

The TAM framework specifically utilizes **Oblivious Random Trees** rather than standard CART trees. This structural choice maximizes GPU vectorization and mathematically eliminates the thread divergence that paralyzes standard decision trees on hardware accelerators {cite:p}`prokhorenkova2018catboost`. For an ensemble consisting of $B$ independent trees, the explicit Primal feature mapping $\phi_{tree}(x)$ is constructed by concatenating the sparse binary indicator vectors of every tree.

To ensure the Reproducing Kernel Hilbert Space (RKHS) norm remains bounded as the ensemble grows, the concatenated vector is strictly scaled by $1/\sqrt{B}$:

$$\phi_{tree}(x) = \frac{1}{\sqrt{B}} \left[ I(x \in R_{1,1}), \dots, I(x \in R_{1,M_1}), \dots, I(x \in R_{B,M_B}) \right]^\top$$

### The Dual-Architecture: Oblivious vs. Flat Histograms

Depending on the hyperparameters provided, this module mathematically bifurcates into two distinct architectures:

1. **Oblivious Random Trees (`max_depth`):** Generates symmetric, multi-dimensional binary splits yielding $2^D$ leaves. Unlike standard CART algorithms that use greedy data-dependent optimization, this architecture generates purely random hypotheses: split features are sampled uniformly, and split thresholds are drawn from a uniform distribution $t \sim \mathcal{U}[-1, 1]$. This is ideal for spatial interactions and mathematically eliminates the thread divergence that paralyzes standard decision trees on GPUs.
2. **Flat N-ary Histograms (`max_leaves`):** Generates flat, 1-Dimensional piecewise splits. For 1D spaces, purely random Monte Carlo splits frequently create starved leaves (zero empirical data), destroying the local covariance matrix. This architecture triggers an **Anti-Starvation Protocol**: it bypasses random sampling entirely in favor of deterministic, mathematically even partitions (linearly spaced across the $[-1, 1]$ domain) to guarantee a full-rank linear system.

## The Primal Matrix Schema & Hyperparameter Topology

To fully understand the hardware geometry, we must decompose how the user-defined hyperparameters strictly map to the global Design Matrix ($\Phi$) and the Penalty Matrix ($P$). Let $B$ be the number of trees, and $L$ be the total leaves per tree.

**1. The Design Matrix Schema ($\Phi$):**
The global matrix horizontally concatenates $B$ independent One-Hot encoded blocks, scaled globally:


$$\Phi_{tree} = \frac{1}{\sqrt{B}} \left[ \Phi^{(1)} \mid \Phi^{(2)} \mid \dots \mid \Phi^{(B)} \right] \in \mathbb{R}^{N \times (B \times L)}$$

**2. The Hyperparameter Routing:**

* **`n_trees` ($B$):** Dictates the horizontal block expansion of $\Phi$. It strictly triggers the $\frac{1}{\sqrt{B}}$ scaling factor across all non-zero entries to ensure RKHS convergence.
* **`max_depth` ($D$):** Forces the Oblivious architecture. The algorithm samples $D$ random thresholds uniformly. The mapping evaluates the boolean path, outputting a One-Hot vector per tree of exactly $L = 2^D$ columns.
* **`max_leaves` ($L$):** Forces the Flat Histogram architecture. It bypasses Monte Carlo sampling to generate $L$ deterministically spaced intervals (via quantiles or linspace), yielding $L$ columns per tree.
* **`additional_features`:** Transforms the 1D interval splits into a multi-dimensional checkerboard of spatial bounding boxes by allowing the random split sampler to cycle through multiple feature axes (e.g., creating a 2D spatial grid for Latitude/Longitude).

## Optimal Penalization

Because an observation $x$ falls into exactly one leaf per tree, the resulting Primal feature evaluation $\phi_{tree}(x)$ is a highly sparse Euclidean bit-string. Since the leaf assignments function as discrete, non-ordinal categorical bins, their coefficients are optimally bounded by an isotropic Ridge penalty block applied across all $B \times L$ total features.

Governed by the hyperparameter **`lambda` ($\lambda$)**, the penalty matrix is mathematically defined as:

$$P_{tree} = \lambda I_{(B \times L)}$$

Applying this isotropic $L_2$ shrinkage strictly prevents isolated, deep leaves containing very few empirical samples from acquiring disproportionately large statistical weights, thereby controlling the structural variance of the tree ensemble.

**Hardware Translation:** Because $B \times L$ can easily reach into the millions, materializing this identity matrix densely would cause immediate VRAM exhaustion. Therefore, the framework strictly initializes $P_{tree}$ natively as a `torch.sparse_coo_tensor`, ensuring this massive penalty block consumes virtually zero memory while seamlessly integrating into the Sparse Conjugate Gradient solver.

## RKHS Eligibility, Kernel Equivalence & Sparsity Adaptation

In classical statistical software, tree predictions are strictly evaluated as algorithmic jump logic. However, projecting shift-invariant kernels into finite Primal spaces using randomized Monte Carlo sampling is a mathematically proven technique for scalable kernel machine resolution {cite:p}`rahimi2007random`.

Building on this, Scornet (2016) provides the rigorous functional analysis proving that infinite random forests are strictly equivalent to specific Kernel estimators {cite:p}`scornet2016random`. Within TAM, the `TreeEffect` operates exactly as a finite-dimensional Primal realization of this Random Forest Kernel. Taking the dot product of two scaled bit-strings over an infinite ensemble evaluates the exact probability that two points $x$ and $x'$ fall into the identical bin. The sparse random bins approximate a Laplacian-like continuous similarity metric, converging at $\mathcal{O}(1/B)$-orders of magnitude faster than standard random Fourier features {cite:p}`wu2016revisiting`.

Furthermore, the theoretical analysis of Breiman's original algorithm proves that this specific feature map inherently **adapts to sparsity** {cite:p}`biau2012analysis`. The statistical convergence rate of the tree ensemble depends strictly on the number of strong, active features rather than the ambient dimension of the noise. This guarantees that the TAM solver remains statistically efficient even when the spatial dictionary is bombarded with irrelevant exogenous variables.

Therefore, equipping this sparse indicator bit-string with a strictly positive isotropic penalty ($P_{tree} = \lambda I_{(B \times L)}$) rigorously embeds the discrete tree ensemble into a valid, finite-dimensional RKHS. 

**Memory Management:** To bypass absolute hardware limitations when the total number of leaves ($B \times L$) reaches into the millions, the framework natively constructs this block-diagonal penalty matrix as a sparse Coordinate (`COO`) tensor. This architectural decision explicitly prevents the dense GPU VRAM exhaustion that routinely paralyzes classical exact solvers.

## Theoretical Integration

In standard machine learning architectures (Random Forests or Gradient Boosting), tree predictions are either greedily averaged or sequentially boosted {cite:p}`hastie2009elements`. These classical ensemble algorithms are notoriously incapable of extrapolating linear trends and routinely fail to isolate smooth, global structural effects.

By severing the tree from its traditional greedy estimator and treating the ensemble purely as a topological feature generator (a sparse Primal dictionary $\Phi_{tree}$), the TAM framework fundamentally re-architects the Random Forest. Solved using exact Primal linear algebra, it flawlessly synchronizes the highly localized, non-linear jump logic of the forest with the global, analytical continuity of classical mathematical functions.