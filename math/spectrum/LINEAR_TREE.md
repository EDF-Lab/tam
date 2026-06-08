# The Linear Tree Effect (Varying-Coefficient Models)

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

## Formula Definition: Varying-Coefficient Networks

By definition, a standard `TreeEffect` constructs a piecewise-constant response surface. It perfectly isolates spatial jumps and structural breaks but cannot calculate local gradients or continuous linear slopes within its terminal leaves.

To build **Piecewise Continuous Regressions**, the TAM framework elevates the "Linear Tree" concept into a rigorous **Varying-Coefficient (VC) Model** {cite:p}`hastie1993varying`. The `LinearTreeEffect` (`lt`) explicitly decouples the variables defining the **partitions/geography** ($x_{part}$) from the continuous variables defining the **local slopes** ($x_{slope}$).

Mathematically, it encapsulates a base tree (acting as the local intercept $\beta_0(x_{part})$) and a tensor product of a slope tree with a linear effect (acting as the varying gradient $\beta_1(x_{part})$). The explicit Primal feature mapping geometrically concatenates these two spaces:

$$\phi_{lt}(x_{part}, x_{slope}) = \left[ \phi_{tree_{base}}(x_{part}), \quad \phi_{tree_{slope}}(x_{part}) \otimes \phi_{lin}(x_{slope}) \right]^\top$$

## Optimal Penalization

To regularize this composite structure, the penalty matrix is constructed as a block-diagonal encapsulation of its internal components.

The local intercepts are bounded by the standard isotropic Ridge penalty of the base tree ($P_{tree_{base}}$), while the local slopes are penalized by the anisotropic Kronecker penalty of the tensor product ($P_{cross}$):

$$P_{lt} = \begin{bmatrix} P_{tree_{base}} & 0 \\ 0 & P_{cross} \end{bmatrix} = \text{diag}(P_{tree_{base}}, P_{cross})$$

This anisotropic structure guarantees that the varying coefficients ($\beta_1(x_{part})$) can be penalized independently from the local intercepts ($\beta_0(x_{part})$), ensuring optimal scale-invariant shrinkage.

## Theoretical Critique: Resolving MOB Singularities

In classical statistical literature, "Linear Trees" (such as M5 or MOB) rely on **Model-Based Recursive Partitioning** {cite:p}`zeileis2008model`. These algorithms attempt to fit an unpenalized local Ordinary Least Squares (OLS) regression strictly inside every individual leaf.

**The Singularity Flaw:** As Zeileis et al. highlight, this recursive partitioning frequently triggers catastrophic matrix singularities. If a specific spatial partition (leaf) is starved of temporal data or lacks variance in the $x_{slope}$ dimension, its local covariance matrix ($X^\top X$) becomes non-invertible, crashing the algorithm {cite:p}`zeileis2008model`.

**The TAM Solution:** By formulating the varying-coefficient model globally via the Primal tensor product ($\Phi_{tree} \otimes \Phi_{lin}$), TAM entirely bypasses the localized OLS problem. The explicit structural penalty $P_{cross}$ strictly dominates any empty or low-variance subspace. If a geographic region lacks sufficient data to resolve a local slope, the solver smoothly shrinks that unstable local gradient exactly to zero. The model safely falls back exclusively on the global linear trend, mathematically guaranteeing a globally full-rank system without algorithmic crashes.

## RKHS Eligibility

The theoretical validity of this effect relies on two fundamental closure properties of Reproducing Kernel Hilbert Spaces (RKHS) proven by Aronszajn {cite:p}`aronszajn1950theory`.

First, the Kronecker product ($\otimes$) of the slope tree and the linear projection rigorously generates a valid piecewise-continuous RKHS (closure under pointwise multiplication). Second, concatenating this joint space with the base tree space via a direct sum ($\oplus$) maintains the strict positive definiteness of the global space. Therefore, the `LinearTreeEffect` operates mathematically as a perfectly valid, finite-dimensional RKHS.

## Architectural Guardrails

When declaring a Linear Tree, the framework enforces several architectural guardrails to guarantee mathematical stability during the exact Primal Sparse Conjugate Gradient resolution:

1. **Collinearity Prevention (`n_trees=1`):** The framework strictly forces the number of trees to 1. Using a massive random forest ($B \gg 1$) to calculate overlapping, localized linear slopes would create catastrophic multicollinearity inside the solver.
2. **Anti-Starvation Protocol (`max_leaves`):** For 1D Piecewise Regressions (e.g., `lt(x, max_leaves=8)`), purely random binary splits create microscopic leaves. By using `max_leaves`, the framework bypasses Monte Carlo sampling, enforcing perfectly even quantile thresholds across the domain to maximize local data density.
3. **Spatial Local Slopes (`max_depth`):** For multi-dimensional interactions (e.g., `lt(lat, others='lon', slope='temp', max_depth=4)`), it uses an oblivious tree on the coordinates to construct a spatial grid, applying the Kronecker product to fit a unique, regularized varying-coefficient for temperature inside every geographic zone.

### Universal Extrapolation

Because TAM isolates the algorithmic partition logic from the continuous geometric space by embedding the discrete leaves as finite Primal blocks, the `LinearTreeEffect` is mathematically capable of extrapolating outside its training distribution.

By using the `extrapolate` parameter (e.g., `'linear'` or `'saturation'`), the framework can force the piecewise linear tree to project a smooth, stable slope infinitely outside the $[-1, 1]$ bounding box, completely overcoming the classic limitation of standard Decision Trees.
