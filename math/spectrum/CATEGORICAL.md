# The Categorical Effect (Discrete Topologies)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

To model a variable residing in a discrete, non-continuous space $\mathcal{X} = \{c_1, c_2, \dots, c_K\}$, the Primal feature map projects the data into a finite-dimensional basis. The TAM framework provides two fundamentally different geometric projections depending on the selected topology:

**1. Euclidean Indicator Projection (`nominal` and `ordinal`):**
If the data is purely discrete, it is embedded into a $K$-dimensional orthogonal Euclidean basis using a standard indicator (One-Hot) vector:

$$\phi_{cat}(x) = \left[ I(x=c_1), I(x=c_2), \dots, I(x=c_K) \right]^\top$$

**2. Continuous Angular Projection (`fourier`):**
If the discrete categories represent sampled points of an underlying continuous cycle (e.g., hours of the day or days of the week), the discrete input $x$ is mapped to a continuous trigonometric domain and projected using a truncated sequence of $m$ harmonics:

$$\phi_{cat}(x) = \left[ \cos\left(\frac{k \pi x}{2}\right), \sin\left(\frac{k \pi x}{2}\right) \right]_{k=1}^m$$

## Optimal Penalization

The construction of the structural penalty sub-matrix $P_{cat}$ rigorously bifurcates based on the underlying topology of the discrete space:

* **Nominal (Unordered) Categories:** If the classes lack an intrinsic hierarchy (e.g., IDs, Colors), an isotropic Ridge penalty is applied:

  $$P_{cat} = \lambda I$$
  
  This uniformly shrinks the coefficients toward zero, acting as an optimal Bayesian prior that prevents sparse or rare categories from dominating the global variance {cite:p}`wood2017generalized`.

* **Ordinal (Ordered) Categories:** If the classes possess a natural sequence (e.g., age groups), an isotropic penalty destroys this inherent topological information. Instead, we enforce smooth clustering by penalizing the transitions between adjacent categories {cite:p}`gertheiss2010sparse`. The optimal penalty is constructed using a finite difference matrix $D$:

  $$P_{cat} = \lambda D^\top D$$

  where $D \in \mathbb{R}^{(K-1) \times K}$ computes the first-order discrete differences $\beta_k - \beta_{k-1}$. Minimizing this quadratic form strictly encourages adjacent categories to share similar statistical weights.

* **Cyclical (Fourier) Categories:** If the categories are discrete representations of a continuous cycle, they inherit the exact diagonal Sobolev penalty from the Fourier effect:

  $$P_{cat} = \lambda \cdot \text{diag}\left( [1 + k^{2s}]_{k=1}^m \oplus [1 + k^{2s}]_{k=1}^m \right)$$

  This structurally filters out high-frequency ringing artifacts across the discrete bins, stabilizing time-series learning by mapping discrete dummies to continuous angular projections {cite:p}`doumeche2025forecasting`.

## RKHS Eligibility

These projections securely embed discrete categorical topologies into a continuous, finite-dimensional vector space. In the nominal case, equipping the orthogonal basis with an isotropic penalty natively generates the Kronecker delta kernel, defined as $k(x, x') = \delta_{x, x'}$. In the ordinal case, the finite difference penalty dynamically generates a structured covariance matrix where the kernel similarity decays proportionally to the ordinal distance between classes. Both satisfy all Mercer conditions for a valid Reproducing Kernel Hilbert Space (RKHS) {cite:p}`aronszajn1950theory`.

## Theoretical Integration & Hardware Translation

In classical statistical software, estimating discrete categorical effects alongside continuous non-linear smoothers often forces the algorithm to rely on partitioned design spaces and alternating iterative backfitting {cite:p}`chambers2017statistical`. 

By rigorously formalizing categorical sets as penalized finite-dimensional RKHS blocks, the TAM framework flawlessly concatenates these discrete embeddings directly beside continuous bases (such as Fourier series or Splines). This theoretically bridges discrete and continuous functional analysis, allowing both to be evaluated (`torch.nn.functional.one_hot`) and solved simultaneously within the exact, closed-form inversion of the global Primal matrix on the GPU.