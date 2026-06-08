# The Tensor Product Effect (Interactions)

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

## Formula Definition

Additive models traditionally assume structural independence between features ($f(x_1, x_2) = f_1(x_1) + f_2(x_2)$). To rigorously model multi-dimensional interactions and complex multivariate response surfaces, the framework constructs a joint hypothesis space by fusing multiple independent marginal spaces.

To understand the interaction mathematically, we evaluate the feature map for a single observation across $K_{te}$ interacting dimensions. Let the explicit Primal feature mapping for an observation in the $k$-th marginal space with $D_k$ degrees of freedom be:

$$\phi_k(x_k) = \left[ h_{k,1}(x_k), h_{k,2}(x_k), \dots, h_{k,D_k}(x_k) \right]$$

The explicit multi-dimensional feature mapping for the joint observation $(x_1, \dots, x_{K_{te}})$ is defined by the recursive Kronecker product of these $K_{te}$ basis vectors {cite:p}`marx2005multidimensional`:

$$\phi_{cross}(x_1, \dots, x_{K_{te}}) = \phi_1(x_1) \otimes \dots \otimes \phi_{K_{te}}(x_{K_{te}})$$

For the standard two-dimensional case ($K_{te}=2$) between a space with $D_1$ degrees of freedom and a space with $D_2$ degrees of freedom, the expanded algebraic vector contains every possible pairwise combination of the individual marginal basis functions:

$$\phi_{cross}(x_1, x_2) = \left[ h_{1,1}(x_1)h_{2,1}(x_2), \dots, h_{1,1}(x_1)h_{2,D_2}(x_2), \dots, h_{1,D_1}(x_1)h_{2,D_2}(x_2) \right]$$

Or written more compactly:

$$\phi_{cross}(x_1, x_2) = \left[ h_{1,i}(x_1) h_{2,j}(x_2) \right]_{i=1, \dots, D_1}^{j=1, \dots, D_2}$$

At the global dataset level, evaluating this vector for all $N_{total}$ temporal observations constructs the complete joint design matrix $\Phi_{cross} = \Phi_1 \otimes \dots \otimes \Phi_{K_{te}}$.

## Optimal Penalization (Scale-Invariance)

Standard multivariate smoothers, such as isotropic Thin Plate Splines, enforce a uniform, rotationally invariant penalty across all dimensions. However, this geometric assumption mathematically fails when the interacting variables operate on radically heterogeneous physical units (e.g., time versus temperature or geographic coordinates versus financial prices). In these cases, a standard Euclidean distance metric becomes arbitrary and meaningless {cite:p}`wood2006low`.

To achieve optimal, scale-invariant smoothing, the framework applies an **anisotropic penalty**. The global structural penalty sub-matrix $P_{cross}$ is constructed by taking the Kronecker product of each marginal penalty matrix $P_i$ with the identity matrix $I$ of the opposing passive spaces, all regulated by a global interaction hyperparameter $\lambda$:

$$P_{cross} = \lambda \sum_{i=1}^{K_{te}} \left( I_1 \otimes \dots \otimes P_i \otimes \dots \otimes I_{K_{te}} \right)$$

Crucially, $P_i = \lambda_i P_{\text{effect}_i}$, meaning each marginal penalty matrix is natively pre-scaled by its own specific structural regularization parameter ($\lambda_i$), consistent with the primal definition of all TAM base effects. 

For the standard two-dimensional case ($K_{te}=2$), this evaluates directly to:

$$P_{cross} = \lambda (P_1 \otimes I_2 + I_1 \otimes P_2)$$

This explicit formulation reveals how the first term isolates structural roughness strictly along the $x_1$ axis (mathematically bypassing $x_2$ via the Identity matrix $I_2$), while the second term strictly penalizes $x_2$. Summing these orthogonal matrices generates a geometrically complete, scale-invariant regularization surface driven by the singular, dynamically optimized global parameter $\lambda$ {cite:p}`wood2006low`.

## The Geometric Scale of the Kronecker Product

Understanding the precise linear algebra of the tensor product is critical to understanding the hardware limitations of classical solvers. The dimension of the resulting feature space scales strictly multiplicatively, the covariance and penalty matrices $P_{cross}$ expand as the product of their squared marginal dimensions (yielding a $(D_A \times D_B) \times (D_A \times D_B)$ matrix)..

Assume an interaction between a Spline space $\Phi_A$ with $D_A = 2$ degrees of freedom and a Fourier space $\Phi_B$ with $D_B = 3$ degrees of freedom. For a single observation, the marginal row vectors are:

$$\phi_A(x_A) = \begin{bmatrix} a_1 & a_2 \end{bmatrix}$$

$$\phi_B(x_B) = \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix}$$

The row-wise Kronecker product computes every pairwise multiplication:

$$\phi_{cross}(x_A, x_B) = \phi_A(x_A) \otimes \phi_B(x_B) = \begin{bmatrix} a_1 b_1 & a_1 b_2 & a_1 b_3 & a_2 b_1 & a_2 b_2 & a_2 b_3 \end{bmatrix}$$

While the feature vector expands multiplicatively from $(2)$ and $(3)$ to $1 \times 6$, the covariance and penalty matrices $P_{cross}$ expand **quadratically**. To evaluate the first term of the anisotropic penalty ($P_A \otimes I_B$), where $P_A \in \mathbb{R}^{2 \times 2}$ and $I_B \in \mathbb{R}^{3 \times 3}$:

$$P_A \otimes I_B = \begin{bmatrix}
p_{11} & 0 & 0 & p_{12} & 0 & 0 \\
0 & p_{11} & 0 & 0 & p_{12} & 0 \\
0 & 0 & p_{11} & 0 & 0 & p_{12} \\
p_{21} & 0 & 0 & p_{22} & 0 & 0 \\
0 & p_{21} & 0 & 0 & p_{22} & 0 \\
0 & 0 & p_{21} & 0 & 0 & p_{22}
\end{bmatrix}$$

The final structural penalty matrix $P_{cross}$ fundamentally scales to $(D_A \times D_B) \times (D_A \times D_B)$.

## RKHS Eligibility and The Representer Theorem

The theoretical validity of this construction relies on the fundamental closure properties of kernels. Aronszajn formally proved that the class of positive definite reproducing kernels is strictly closed under pointwise multiplication {cite:p}`aronszajn1950theory`. Therefore, the direct tensor product of multiple valid marginal Hilbert spaces, $\mathcal{H}_{cross} = \mathcal{H}_1 \otimes \dots \otimes \mathcal{H}_N$, rigorously generates a valid, finite-dimensional product RKHS.

However, existence alone is insufficient for scalable computation. The Generalized Representer Theorem bridges this gap, proving that any regularized risk functional evaluated over this infinite-dimensional product space possesses a globally optimal solution that lies exactly within the span of the empirical finite-dimensional Primal mappings {cite:p}`scholkopf2001generalized, hofmann2008kernel`. This guarantees that TAM's truncated matrix formulations do not merely approximate the interaction, but solve it mathematically optimally.

## Neural Interactions and Marginal Clarity (Critique)

While the tensor product seamlessly fuses classical bases like splines and polynomials, applying it naively to deep neural embeddings (`NeuralEffect`) introduces severe interpretability pathologies.

If a model interacts two unconstrained neural feature maps without including their independent main effects, the resulting response surface entangles the main effect variance with pure interaction variance. To preserve strict structural auditability, we must respect the principle of **Marginal Clarity** {cite:p}`yang2021gami`.

GAMI-Net highlights that to prevent this entanglement, interaction components must be structurally orthogonalized against their main effects {cite:p}`yang2021gami`. Within the TAM Primal resolution, this means that whenever a `TensorProductEffect` involving neural networks is declared, the underlying main effects must either be explicitly defined as separate additive terms in the formula, or the interaction penalty $\lambda$ must be heavily regularized to ensure it only captures the residual joint variance that the marginal main effects fail to explain.

## Theoretical Integration & Hardware Translation

If a model interacts a Spline ($D_A = 100$) with a spatial Fourier grid ($D_B = 100$), the resulting interaction generates $10,000$ unique features. The dense Primal covariance matrix ($\Phi_{cross}^\top \Phi_{cross}$) and the penalty matrix ($P_{cross}$) will scale to $10,000 \times 10,000$. Adding a third interaction term pushes this to $1,000,000 \times 1,000,000$.

In classical statistical computing environments, resolving these high-dimensional tensor product smooths creates rank-deficient design matrices that exhaust physical RAM and require complex iterative orthogonalization routines.

By natively embedding the recursive Kronecker basis expansion $\Phi_{cross}$ and the anisotropic block penalty $P_{cross}$ directly into the global Primal architecture using highly parallelized GPU tensor algebra (`torch.kron`, `einsum`), the TAM framework bypasses these numerical bottlenecks entirely. This forces models with extreme dimensionality into the matrix-free Sparse Conjugate Gradient solver, allowing multidimensional interactions to be solved simultaneously with the rest of the GAM without triggering memory faults.

## Theoretical Extension: Piecewise Continuous Regressions (Linear Trees)

In traditional statistical learning, piecewise linear regressions (or "Linear Model Trees") require complex algorithmic overrides to handle singular covariance matrices when local partitions (leaves) contain too few temporal observations ($T$).

In the TAM Primal RKHS framework, this is resolved analytically. Let $\phi_{tree}(x_A)$ be the highly sparse, mutually exclusive indicator vector generated by a Random Binning Feature ensemble (see the Tree Effect), and let $\phi_{lin}(x_B)$ be a globally supported continuous linear projection. Their Kronecker product yields:

$$\phi_{cross}(x_A, x_B) = \phi_{tree}(x_A) \otimes \phi_{lin}(x_B)$$

Geometrically, this operation projects the continuous linear gradient strictly into the disjoint spatial support of the tree's terminal leaves. By Aronszajn's theorem, because both marginal spaces are valid Hilbert spaces, their product rigorously generates a valid piecewise-continuous RKHS.

### Structural Safety via the Global Primal Equation

Because the tensor product creates a massive number of coefficients (e.g., Total Leaves $\times$ Linear Features), standard local regressions would suffer catastrophic matrix singularity. However, because TAM solves the system globally, the resulting tensor dimension $D$ is analytically safeguarded by the core group equation:

$$\hat{\theta}_g = \left( \Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g + T \cdot P_{cross} \right)^{-1} \Phi_g^\top \Lambda_g^\top \Lambda_g Y_g$$

The anisotropic penalty $P_{cross}$ ensures absolute structural safety. If a specific spatial partition $R_{b,m}$ experiences an absence of temporal data (i.e., the corresponding rows in the masking matrix $\Lambda_g$ evaluate to zero), the explicit structural penalty matrix $P_{cross}$ strictly dominates that empty subspace. It smoothly shrinks the unstable local slope $\theta_{g, i}$ to exactly **zero**.

Consequently, the local prediction does not revert to an arbitrary localized constant; the interaction term perfectly zeroes itself out, causing the model to gracefully fall back exclusively on the **global marginal main effects** (e.g., the global intercept and global linear trend) fitted concurrently in the broader equation.

> Note: To successfully compute these local slopes without triggering matrix singularities, the underlying tree must either rely on massive overlapping ensembles ($B \gg 1$) or explicitly trigger the 1D Flat Histogram architecture using the `max_leaves` hyperparameter.