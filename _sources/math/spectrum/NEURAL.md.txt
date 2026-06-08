# The Neural Effect (NEPT)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## The NEPT Concept (Neural Explicit Primal Tensorization)

The Neural effect is a globally exact, backpropagation-free generalized additive module. It resolves the computational intractability of deep kernel theories by translating them into a **Neural Explicit Primal Tensorization (NEPT)**, allowing deep neural representations to be hybridized simultaneously with classical topological effects (like splines and wavelets).

Instead of tracking continuous network evolution in an intractable Dual function space, NEPT aggressively normalizes a deep, frozen network and extracts it as a finite-dimensional Primal tensor block $\Phi_{neural} \in \mathbb{R}^{N_{total} \times D_{neural}}$. This computationally operationalizes the continuous neural network theory, which proved that an infinite-width network with an $L_2$-norm penalty on the output weights functions as an exact kernel machine {cite:p}`le2007continuous`.

## Formula Definition: Neural Explicit Primal Tensorization

To endow the framework with the deep, compositional abstraction capabilities of modern neural architectures without inheriting their optimization instabilities, NEPT projects the input data through a randomized, multi-layer feed-forward neural network. 

Let $L$ denote the total number of hidden layers (`n_hidden_layers`), and $N_l$ denote the width of the layer (`n_neurons`). The sequence of hidden weight matrices $W^{(l)}$ and biases $b^{(l)}$ are stochastically sampled and permanently frozen without backpropagation. To guarantee convergence to the Gaussian Process prior and prevent variance explosion across deep layers, the initial projection layer is drawn from a standard normal $W^{(1)} \sim \mathcal{N}(0, 1)$, while all subsequent hidden layers are rigorously scaled by the width of the preceding layer: $W^{(l)} \sim \mathcal{N}(0, 1/N_{l-1})$ for $l \ge 2$. Biases $b^{(l)}$ are assigned based on activation-specific theoretical priors: $b^{(l)} \sim \mathcal{U}[0, 2\pi]$ for the `cos` activation to instantiate exact Random Fourier Features, and $b^{(l)} \sim \mathcal{N}(0, 1)$ for the `relu` and `tanh` activations.

The explicit Primal feature mapping iteratively applies a non-linear continuous activation $\sigma(\cdot)$ across the frozen hidden layers, where $\sigma(v)$ applies one of the following element-wise transformations based on the chosen hyperparameter ($\max(0, v)$ for `relu`, $\cos(v)$ for `cos`, or $\frac{e^v - e^{-v}}{e^v + e^{-v}}$ for `tanh`):

$$z^{(0)} = x$$
$$z^{(l)} = \sigma\left( z^{(l-1)} W^{(l)} + b^{(l)} \right) \quad \text{for } l = 1, \dots, L$$

The final hidden layer is extracted as the finite-dimensional Primal block, aggressively scaled by $1/\sqrt{N_L}$ to normalize the final output variance across the arbitrarily wide random feature space:

$$\phi_{neural}(x) = \frac{1}{\sqrt{N_L}} z^{(L)}$$

Because the entirety of the deep internal representations are frozen, only the final linear readout coefficients $\theta_{neural}$ mapping $z^{(L)}$ to the target are actively learned by the global estimator.

## RKHS Eligibility & Optimal Penalization

By restricting the learning process exclusively to the final readout layer $\theta \in \mathbb{R}^D$, the notoriously highly non-convex deep neural optimization problem strictly collapses into a convex quadratic form. 

The mathematically optimal penalization is an isotropic Ridge penalty applied to the readout coefficients, structurally defined by an identity block:

$$P_{neural} = \lambda I$$

By equipping the finite-width neural projection with this strictly positive isotropic penalty and the $1/\sqrt{N_L}$ variance scaling, NEPT guarantees its operation within a valid Reproducing Kernel Hilbert Space (RKHS). This seamlessly integrates the deep representations into the globally exact primal solver alongside classical structural effects without disrupting the global convexity of the regularized normal equations.

## Structural Novelty: The 8-Paradigm Synthesis

Because NEPT rigorously flattens the deep kernel projection into the deterministic block-diagonal matrix $\Phi$, it fundamentally redefines how deep learning interacts with additive modeling. Below is the precise structural differentiation between NEPT and the 8 foundational paradigms it synthesizes:

1. **Compositional Kernels** {cite:p}`daniely2016toward`
   * **Common Point:** Both recognize that the topological architecture of a randomly initialized deep network inherently defines a highly expressive compositional basis.
   * **The NEPT Novelty:** While theoretical literature evaluates this expressivity within the abstract Dual kernel space, NEPT explicitly extracts it into a finite Primal tensor. This allows the compositional neural kernel to be algebraically concatenated alongside non-neural structural priors (like exact localized $\Phi_{wavelet}$ or $\Phi_{spline}$), an operation impossible in pure Dual theory.

2. **Neural Network Gaussian Processes (NNGP)** {cite:p}`lee2017deep`
   * **Common Point:** Both exploit the theoretical guarantee that infinitely wide, randomly initialized frozen networks converge to a valid Gaussian Process prior.
   * **The NEPT Novelty:** Exact NNGP inference requires evaluating the dense Dual covariance matrix, creating an $\mathcal{O}(N_{total}^3)$ bottleneck that paralyzes hardware on Gigadata ($N_{total} > 10^6$). NEPT systematically truncates this GP prior into the $D$-dimensional Primal space, collapsing the complexity to $\mathcal{O}(N_{total} D^2)$ and solving it instantly on GPUs via Sparse Conjugate Gradients.

3. **Neural Tangent Kernel (NTK)** {cite:p}`jacot2018neural`
   * **Common Point:** Both acknowledge the continuous kernel dynamics governing modern neural networks.
   * **The NEPT Novelty:** Exact NTK tracks the continuous gradient descent of *all* weights in function space. NEPT entirely rejects continuous weight updates. By freezing the internal layers and optimizing only the global readout $\theta$, the continuous differential flow collapses into a single-step, closed-form quadratic equation.

4. **Neural Additive Models (NAMs)** {cite:p}`agarwal2021neural`
   * **Common Point:** Both structurally isolate individual input features into independent neural sub-networks to guarantee strict interpretability.
   * **The NEPT Novelty:** NAMs attempt to optimize these sub-networks jointly using non-convex backpropagation. NEPT bypasses backpropagation completely. By flattening the features into static Primal blocks, NEPT solves the additive network exactly alongside classical statistical bases (e.g., GAM Splines), ensuring absolute mathematical convexity.

5. **N-BEATS** {cite:p}`oreshkin2020n`
   * **Common Point:** Both leverage deep hierarchical topologies to discover complex basis expansions specifically optimized for time-series forecasting.
   * **The NEPT Novelty:** N-BEATS relies on a deep sequential residual bottleneck (layer $k$ fits the error leftover by layer $k-1$). NEPT rejects sequential error fitting. By flattening all effects into a single block-diagonal global matrix $\Phi$, NEPT resolves the interactions of all bases simultaneously, preventing cascading error propagation.

6. **Efficient BackProp** {cite:p}`lecun1998efficient`
   * **Common Point:** Both works target the pathological optimization landscapes inherent in deep neural architectures (ill-conditioning, saddle points, vanishing gradients).
   * **The NEPT Novelty:** Classical deep learning mitigates these pathologies through stochastic heuristics (initialization, learning rate schedules). NEPT *eradicates the problem entirely*. The exact global Ridge penalty $P = \lambda I$ mathematically guarantees a globally optimal loss surface with zero saddle points, transforming deep learning optimization into an exact linear algebra certainty.

7. **Extreme Learning Machines (ELMs) & Random Vector Functional Links (RVFLs)** {cite:p}`pao1994learning, huang2006extreme`
   * **Common Point:** Both architectures project inputs through randomly initialized hidden layers, freeze those internal weights, and exclusively optimize the final linear readout layer using least squares or Ridge regression.
   * **The NEPT Novelty:** Classical ELMs and RVFLs are historically viewed as standalone, heuristic randomized architectures. NEPT explicitly re-contextualizes this frozen mechanism through the rigorous lens of modern NNGP theory. By enforcing the $1/\sqrt{N_L}$ variance scaling and projecting the readout strictly as an isotropic Ridge penalty ($P = \lambda I$) within the global block-diagonal design matrix $\Phi$, NEPT elevates the ELM from a stochastic heuristic into a mathematically guaranteed RKHS topology embedded safely within a continuous generalized additive framework.

8. **Continuous Neural Networks & Exact Kernel Machines** {cite:p}`le2007continuous`
   * **Common Point:** Both frameworks mathematically establish that freezing an infinite-width neural network and applying an $L_2$-norm regularization penalty to the output weights creates an exact kernel machine.
   * **The NEPT Novelty:** Analytical proofs for continuous neural networks fundamentally relied on uniform priors across a single hidden layer to maintain tractability in the uncountably infinite Dual space. NEPT abandons these single-layer uniform bounds in favor of deep Gaussian representations ($W^{(l)} \sim \mathcal{N}(0, 1/N_{l-1})$) and terminal variance scaling. Furthermore, NEPT explicitly truncates this theoretical guarantee into a highly optimized, finite-dimensional Primal block, completely bypassing the $\mathcal{O}(N_{total}^3)$ bottleneck of Dual space evaluation.