# The Exact Primal Resolution

**Navigation:**

  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/01_additive_api.md)
  * **Related topic:** [Tensorization & Broadcasting](02_tensorization.md)

## Defining the Problem: From CPU Bottlenecks to GPU Scale

Modern statistical learning is often fractured across disparate paradigms. While Deep Learning scales massively on GPUs, it relies on strictly non-convex, stochastic gradient descent heuristics. Conversely, traditional Generalized Additive Models (GAMs) offer mathematical exactness and interpretability. For instance, robust GAM solvers (like those utilizing PIRLS, as in {cite:p}`wood2025package`) can resolve Gaussian problems exactly in a single step. However, these traditional exact solvers are inherently CPU-bound. When faced with massive industrial datasets ($N_{total} > 10^6$) or highly complex tensor-product topologies ($D > 10^4$), the classical computation of the covariance matrix becomes an intractable hardware bottleneck.

The Time series Additive Model (TAM) framework bridges this gap. It proposes a rigorous algebraic unification that maintains the globally exact, closed-form resolution of standard GAMs, but re-architects the underlying linear algebra for massive, Matrix-Free GPU parallelization.

At the core of this framework lies the fundamental equation for a given group $g$:

$$Y_g = \mu(X_g) + \epsilon_g$$

Where:
* $Y_g$ represents the target variable we aim to predict for group $g$.
* $X_g$ represents the matrix of input covariates.
* $\mu(X_g)$ is the underlying true function (conditional expectation) mapping the inputs to the target.
* $\epsilon_g$ is the irreducible stochastic noise.

Estimating $\mu(X_g)$ as a single monolithic block is often intractable and lacks interpretability. To address this, the framework relies on a Generalized Additive formulation, decomposing the global function into a sum of independent functional operators.

$$\mu(X_g) = \sum_{l=1}^{L} h_l(X_g)$$

To resolve this continuous formulation computationally, each sub-function $h_l(X_g)$ is projected into a discrete feature space. Specifically, each effect $l$ is decomposed into a sum of $I_l$ explicitly evaluated basis functions multiplied by their respective coefficients:

$$h_l(X_g) = \sum_{i=1}^{I_l} \phi_{l,i}(X_g) \theta_{g, l,i}$$

The evaluation of this basis expansion over all $T$ temporal data steps for a specific independent group $g$ is then captured globally using a specific topological dictionary, known as the group Design Matrix, $\Phi_{g, l}$. The full model is evaluated as the sum of the dot products of these matrices and their corresponding coefficient vectors $\theta_{g, l}$:

$$Y_g = \sum_{l=1}^{L} (\Phi_{g, l} \theta_{g, l}) + \epsilon_g$$

-----

## RKHS Theory and The Exact Primal Equations

To stabilize the estimation of these coefficients, the framework employs biased estimation via Ridge Regression {cite:p}`hoerl1970ridge`. Following the foundational classical Representer Theorem of Kimeldorf and Wahba {cite:p}`kimeldorf1970correspondence`, minimizing a loss function weighted by observation reliability alongside a roughness penalty guarantees a globally optimal continuous function. Furthermore, modern generalizations of the representer theorem by {cite:p}`boyer2019representer` rigorously establish that regularizing these inverse problems with a strictly convex function guarantees that the solution inherently collapses into a combination of a finite number of fundamental atoms. Relying on this structural guarantee, TAM bypasses the infinite-dimensional Dual space solvers used by standard kernel machines {cite:p}`cortes1995support, vapnik2013nature` and explicitly parameterizes the problem directly within a truncated, finite-dimensional Primal space {cite:p}`bach2024learning`. Consequently, relying on the unified algebraic framework proposed by {cite:p}`doumeche2025forecasting`, the exact primal resolution yields the globally optimal coefficients for our chosen finite basis in a single highly efficient algebraic step.

The exact primal minimization problem for group $g$ is defined as:

$$\min_{\theta_g} \frac{1}{T} \sum_{j=1}^T \|\Lambda_g ( \Phi_{g, j} \theta_g - Y_{g, j} )\|_2^2 + \|M\theta_g\|_2^2$$

To find the optimal exact coefficients $\hat{\theta}_g$, the system minimizes a penalized empirical risk function. This requires defining specific matrices:

1. **The Target Vector ($Y_g$)**
2. **The Group Design Matrix ($\Phi_g$)**
3. **The Loss Weighting Matrix ($\Lambda_g$)**: A diagonal matrix encoding the relative importance of each observation (represented computationally as $L_g^\top L_g$).
4. **The Global Penalty Matrix ($P$ or $M^*M$)**: A regularization matrix that acts as a structural constraint, restricting the solution to a compact mathematical set to guarantee stability {cite:p}`tikhonov1943stability`.

  > **Implementation:** While the theoretical framework uses $\Lambda_g \in \mathbb{R}^{T \times T}$ to handle both observation weighting and the masking of padded asynchronous sequences (where $\Lambda_{g, t,t} = 0$), constructing a dense temporal diagonal matrix is inefficient on GPUs. In practice, the top-level API defaults to unweighted Ordinary Least Squares ($\Lambda_g = I$), and the temporal `NaN` masking is achieved purely via memory-efficient boolean indexing prior to the matrix multiplications. The underlying mathematical engines remain pre-architected to accept custom target-weight matrices to support future Weighted Least Squares features.

  > **Historical Context:** The framework's deliberate shift away from the Dual space is a direct response to the classical bottleneck of Support Vector Machines and traditional kernel methods {cite:p}`cortes1995support, vapnik2013nature`. Furthermore, while {cite:p}`kimeldorf1970correspondence` established the Representer Theorem for splines, it is the Generalized Representer Theorem {cite:p}`scholkopf2001generalized` that mathematically licenses the application of these empirical risk minimizations across our wider spectrum of topological penalties.

### Deriving the Closed-Form Solution

We formulate the global convex loss function $\mathcal{L}(\theta_g)$ using the Design Matrix ($\Phi_g$), the observation weighting matrix ($\Lambda_g$, computed as $L_g^\top L_g$), and the block-diagonal structural penalty matrix ($P$):

$$\mathcal{L}(\theta_g) = (\Phi_g \theta_g - Y_g)^\top \Lambda_g^\top \Lambda_g (\Phi_g \theta_g - Y_g) + T \theta_g^\top P \theta_g$$

Because this formulation is strictly convex, we can find the global minimum analytically by taking the gradient with respect to $\theta_g$ and setting it to zero:

$$\nabla_{\theta_g} \mathcal{L} = 2 \Phi_g^\top \Lambda_g^\top \Lambda_g (\Phi_g \theta_g - Y_g) + 2 T P \theta_g = 0$$

Expanding and isolating $\theta_g$:

$$\Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g \theta_g + T P \theta_g = \Phi_g^\top \Lambda_g^\top \Lambda_g Y_g$$
$$(\Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g + T P) \theta_g = \Phi_g^\top \Lambda_g^\top \Lambda_g Y_g$$

Multiplying by the inverse of the regularized covariance matrix yields the exact TAM core equation:

$$\hat{\theta}_g = \left( \Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g + T P \right)^{-1} \Phi_g^\top \Lambda_g^\top \Lambda_g Y_g$$

  > **Implementation:** See how this continuous mathematical theory is translated into OOM-safe, PyTorch tensor operations in [The Additive API & Object-Oriented Architecture](../../architecture/core/01_additive_api.md).

-----

## Safe Concatenation: Aronszajn’s Direct Sum

When building a complex model with multiple additive effects, the sub-spaces must not mathematically corrupt one another. How do we ensure that a discrete difference penalty applied to a Spline does not accidentally penalize a Fourier series?

The abstract RKHS norm $\|h_l\|_{\mathcal{H}_l}^2$ penalizes the complexity of a specific function. By defining our mapping $\Phi_l$, we algebraically reduce this continuous norm to a discrete quadratic form on the coefficients:

$$\|h_l\|_{\mathcal{H}_l}^2 = \theta_l^\top P_l \theta_l$$

Aronszajn’s Direct Sum Theorem {cite:p}`aronszajn1950theory` proves that if a global Hilbert space is the orthogonal direct sum of independent sub-spaces ($\mathcal{H} = \bigoplus_{l=1}^L \mathcal{H}_l$), the cross-terms in the inner product vanish. Therefore, the global penalty is strictly the sum of the individual spatial penalties.


### Example: Mixing Effects in the Global Matrices

To understand how the global matrices $\Phi_g$ and $P$ are structurally assembled in practice, consider a model mixing three distinct effects: a Linear (`l(x1)`), a P-Spline (`s(x2)`), and a Fourier series (`f(x3)`).

**The Global Design Matrix ($\Phi_g$):**
The individual design matrices are concatenated horizontally. If we have $T$ rows of temporal data for group $g$, $\Phi_{g, lin}$ has 1 column, $\Phi_{g, spl}$ has $I_{spl}$ columns (knots), and $\Phi_{g, four}$ has $I_{four}$ columns (harmonics). The global $\Phi_g$ has dimension $T \times (1 + I_{spl} + I_{four})$:

$$\Phi_g = \begin{bmatrix} \vert & \vert & \vert \\ \Phi_{g, lin} & \Phi_{g, spl} & \Phi_{g, four} \\ \vert & \vert & \vert \end{bmatrix}$$

**The Global Penalty Matrix ($P$):**
The structural priors are assembled into a block-diagonal matrix based on Aronszajn’s Direct Sum Theorem {cite:p}`aronszajn1950theory, bach2024learning`. The Linear effect uses a standard Ridge penalty ($P_{lin} = I$), the Spline uses a discrete difference penalty ($P_{spl} = D^\top D$), and the Fourier effect uses a diagonal Sobolev penalty ($P_{four} = S$). The global $P$ is:

$$P = \begin{bmatrix} \lambda_{lin} P_{lin} & 0 & 0 \\ 0 & \lambda_{spl} P_{spl} & 0 \\ 0 & 0 & \lambda_{four} P_{four} \end{bmatrix}$$

**The Global Coefficient Vector ($\theta_g$):**
Because these effects are structurally isolated within the penalty matrix, the solver calculates the exact global coefficients in a single algebraic operation without them mathematically corrupting one another, yielding a vertically concatenated vector:

$$\hat{\theta}_g = \begin{bmatrix} \hat{\theta}_{g, lin} \\ \hat{\theta}_{g, spl} \\ \hat{\theta}_{g, four} \end{bmatrix}$$

-----

## Computational Evolution: The Matrix-Free CG Solver

While the exact Primal shift reduces complexity from the intractable $\mathcal{O}(N_{total}^3)$ of Dual kernel methods to $\mathcal{O}(G \times T \times D^2 + G \times D^3)$, explicitly forming the dense global covariance matrix $A_g = (\Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g + T P)$ can still cause fatal Out-Of-Memory (OOM) errors on GPUs when $D$ becomes massive (e.g., interacting deep neural features).

To bypass this final physical VRAM limit, the framework deploys a **Matrix-Free Sparse Conjugate Gradient (CG)** solver.

Krylov subspace methods like CG do not require the explicit matrix $A_g$; they only require a function capable of computing the matrix-vector product $A_g v$ for a given vector $v$ {cite:p}`golub1996matrix`. By exploiting the associative property of matrix multiplication, TAM evaluates the sequence strictly from right to left:

$$A_g v = \Phi_g^\top \Big( \Lambda_g^\top \Lambda_g \big( \Phi_g v \big) \Big) + T P v$$



1. Compute $u_1 = \Phi_g v$ (Shape: $T \times 1$)
2. Compute $u_2 = \Lambda_g^\top \Lambda_g u_1$ (Shape: $T \times 1$)
3. Compute $u_3 = \Phi_g^\top u_2$ (Shape: $D \times 1$)
4. Add the penalty $u_3 + T P v$

-----

### The Three-Tier Optimization Strategy

Rather than relying solely on a separate validation set, the framework replaces standard black-box hyperparameter tuning with a direct three-speed resolution:

1. **Business Priors (Instantaneous Resolution):** Hyperparameters fixed by domain expertise are parsed directly to compute the analytic solution instantly without iterative searches.
2. **Algebraic GCV Solver (Continuous Optimization):** Exact minimization of the Generalized Cross-Validation score using the cyclic trace trick to automatically find the ideal regularization penalty $\lambda$ without a hold-out set {cite:p}`golub1979generalized`. [Follow the link for the GCV Solver mathematics.](../../math/core/05_gcv_theory.md).
3. **Grid Search (Discrete Optimization):** Multi-Start Coordinate Descent optimization for architectural hyperparameters. By cycling through one parameter axis at a time, the solver reliably navigates non-convex topological search spaces {cite:p}`wright2015coordinate`.

---

## The Spectrum of Functional Topologies

To be eligible for the global design matrix, every effect must define its Primal map $\phi(x)$, an optimal penalty $P_i$, and prove its valid RKHS embedding. The framework achieves this across a highly diverse spectrum of mathematical domains.

To have a quick look at the hyperparameters and the formulas for $\Phi$ (mapping function) and $P$ (penalty), look at the summary table in the [`THEORY`](THEORY.md) chapter.

Check the mathematical definition of the different effects at the following chapters: **[Linear](../spectrum/LINEAR.md)** / **[Fourier](../spectrum/FOURIER.md)** / **[Spline](../spectrum/SPLINES.md)** / **[Chebyshev](../spectrum/CHEBYSHEV.md)** / **[Categorical](../spectrum/CATEGORICAL.md)** / **[Interaction](../spectrum/CROSS_TENSOR.md)** / **[RBF](../spectrum/RBF.md)** / **[Neural](../spectrum/NEURAL.md)** / **[Tree](../spectrum/TREE.md)** / **[Linear Tree](../spectrum/LINEAR_TREE.md)** / **[Wavelet](../spectrum/WAVELETS.md)** / **[Physics](../spectrum/PHYSICS_PIKL.md)** / **[PID](../spectrum/PID.md)**

## Conclusion

The ultimate theoretical breakthrough of TAM is the rigid structural flattening of these bases. By formally establishing that spectral waves, algorithmic tree partitions, deep compositional features, and physical PDEs can all be simultaneously expressed as strictly penalized, finite-dimensional RKHS blocks, the framework transforms the most complex, pathologically non-convex challenges in modern machine learning into a single, perfectly exact linear algebra resolution.