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

1. **Oblivious Random Trees (`max_depth`):** Generates symmetric, multi-dimensional binary splits yielding $2^D$ leaves. Unlike standard CART algorithms that use greedy data-dependent optimization, this architecture generates data-independent topological hypotheses: split features are sampled uniformly, and split thresholds are drawn from statistical distributions (either uniform spatial grids or density-adaptive quantiles). This is ideal for spatial interactions and mathematically eliminates the thread divergence that paralyzes standard decision trees on GPUs.
2. **Flat N-ary Histograms (`max_leaves`):** Generates flat, 1-Dimensional piecewise splits. For 1D spaces, pure Monte Carlo splits frequently create starved leaves (zero empirical data), destroying the local covariance matrix. This architecture triggers an **Anti-Starvation Protocol**: it bypasses random sampling entirely in favor of deterministic, mathematically even partitions (via strict linear spacing or exact quantiles) to guarantee a full-rank linear system.

## The Primal Matrix Schema & Hyperparameter Topology

To fully understand the hardware geometry, we must decompose how the user-defined hyperparameters strictly map to the global Design Matrix ($\Phi$) and the Penalty Matrix ($P$). Let $B$ be the number of trees, and $L$ be the total leaves per tree.

**1. The Design Matrix Schema ($\Phi$):**
The global matrix horizontally concatenates $B$ independent One-Hot encoded blocks, scaled globally:

$$\Phi_{tree} = \frac{1}{\sqrt{B}} \left[ \Phi^{(1)} \mid \Phi^{(2)} \mid \dots \mid \Phi^{(B)} \right] \in \mathbb{R}^{N \times (B \times L)}$$

**2. The Hyperparameter Routing:**

* **`n_trees` ($B$):** Dictates the horizontal block expansion of $\Phi$. It strictly triggers the $\frac{1}{\sqrt{B}}$ scaling factor across all non-zero entries to ensure RKHS convergence.
* **`max_depth` ($D$):** Forces the Oblivious architecture. The algorithm samples $D$ random thresholds. The mapping evaluates the boolean path, outputting a One-Hot vector per tree of exactly $L = 2^D$ columns.
* **`max_leaves` ($L$):** Forces the Flat Histogram architecture. It bypasses Monte Carlo sampling to generate $L$ deterministically spaced intervals, yielding $L$ columns per tree.
* **`additional_features`:** Transforms the 1D interval splits into a multi-dimensional checkerboard of spatial bounding boxes by allowing the random split sampler to cycle through multiple feature axes.
* **`split_strategy`:** Defines the threshold sampling distribution. `'uniform'` creates mathematically orthogonal, shift-invariant Cartesian grids. `'quantile'` applies the empirical Probability Integral Transform to create density-adaptive partitions that perfectly balance sample distributions across all leaves.
* **`sp_alpha` ($\alpha_{sp}$):** Controls the Anisotropic Sparsity-Adaptive Penalty, scaling the $L_2$ shrinkage inversely to the empirical data density of each specific leaf.

## Empirical Sparsity-Adaptive Penalization (Anisotropic Ridge)

A fundamental mathematical clash exists between the dense representations of continuous bases and the severe geometric sparsity of random forests. Because a single data point activates exactly one leaf per tree, classical exact solvers operating with a global isotropic Ridge penalty ($P = \lambda I$) tend to aggressively over-penalize the fragmented tree leaves, artificially collapsing the ensemble's predictive variance. 

To achieve statistical equivalence with continuous modules, TAM replaces the isotropic penalty with an **Anisotropic Sparsity-Adaptive Penalty**. During the initial forward pass, the framework evaluates the routing logic on the training manifold to capture the exact empirical sample count $C_i$ landing in each specific leaf $i$. The base penalty $\lambda_p$ is then dynamically scaled for each individual leaf feature along the diagonal matrix:

$$[P_{tree}]_{ii} = \lambda_p \cdot \left( \frac{C_i + \epsilon}{\bar{C}} \right)^{-\alpha_{sp}}$$

where $\bar{C}$ is the mean sample count across all leaves, and $\epsilon$ is a microscopic smoothing constant to prevent division by zero.

* When $\alpha_{sp} = 0$, the scaling factor collapses to $1$, strictly recovering the classical isotropic Ridge penalty ($\lambda_p I$) applied uniformly across the entire forest.
* When $\alpha_{sp} > 0$, the penalty dynamically adapts to the data density. Highly populated leaves receive standard or reduced shrinkage, confidently fitting strong signals. Conversely, starved edge-boundary leaves receive geometrically massive penalties. This prevents the sparse basis from fitting empirical noise without artificially depressing the global tree effect.

**Interaction with Split Strategy:** This penalty perfectly complements the topological splitting strategy. When using the `quantile` strategy, data is distributed evenly across all leaves ($C_i \approx \bar{C}$), causing the anisotropic penalty to elegantly collapse back into a stable isotropic penalty. When using the `uniform` strategy, the anisotropic penalty actively suppresses the naturally starved edge regions, directly preventing catastrophic rank deficiency and test drift.

**Hardware Translation:** Because $B \times L$ can easily reach into the millions, materializing this diagonal matrix densely would cause immediate VRAM exhaustion. Therefore, the framework strictly initializes $P_{tree}$ natively as a `torch.sparse_coo_tensor`, consuming virtually zero memory while seamlessly integrating into the Sparse Conjugate Gradient solver.

## RKHS Eligibility, Kernel Equivalence & Sparsity Adaptation

In classical statistical software, tree predictions are strictly evaluated as algorithmic jump logic. However, projecting shift-invariant kernels into finite Primal spaces using randomized Monte Carlo sampling is a mathematically proven technique for scalable kernel machine resolution {cite:p}`rahimi2007random`.

Building on this, Scornet (2016) provides the rigorous functional analysis proving that infinite random forests are strictly equivalent to specific Kernel estimators {cite:p}`scornet2016random`. Within TAM, the `TreeEffect` operates exactly as a finite-dimensional Primal realization of this Random Forest Kernel. Taking the dot product of two scaled bit-strings over an infinite ensemble evaluates the exact probability that two points $x$ and $x'$ fall into the identical bin. The sparse random bins approximate a Laplacian-like continuous similarity metric, converging at $\mathcal{O}(1/B)$-orders of magnitude faster than standard random Fourier features {cite:p}`wu2016revisiting`.

Furthermore, the theoretical analysis of Breiman's original algorithm proves that this specific feature map inherently **adapts to sparsity** {cite:p}`biau2012analysis`. The statistical convergence rate of the tree ensemble depends strictly on the number of strong, active features rather than the ambient dimension of the noise. This guarantees that the TAM solver remains statistically efficient even when the spatial dictionary is bombarded with irrelevant exogenous variables.

Therefore, equipping this sparse indicator bit-string with a strictly positive, sparsity-adaptive penalty rigorously embeds the discrete tree ensemble into a valid, finite-dimensional RKHS. 

## Theoretical Integration

In standard machine learning architectures (Random Forests or Gradient Boosting), tree predictions are either greedily averaged or sequentially boosted {cite:p}`hastie2009elements`. These classical ensemble algorithms are notoriously incapable of extrapolating linear trends and routinely fail to isolate smooth, global structural effects.

By severing the tree from its traditional greedy estimator and treating the ensemble purely as a topological feature generator (a sparse Primal dictionary $\Phi_{tree}$), the TAM framework fundamentally re-architects the Random Forest. Solved using exact Primal linear algebra, it flawlessly synchronizes the highly localized, non-linear jump logic of the forest with the global, analytical continuity of classical mathematical functions.
