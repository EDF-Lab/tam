# Physics-Informed Kernel Learning (PIKL)

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

The TAM framework integrates prior physical knowledge directly into the statistical learning process. Unlike standard Physics-Informed Neural Networks (PINNs), which rely on non-convex optimization via stochastic gradient descent {cite:p}`raissi2019physics`, this framework formulates the physical constraint as an exact, convex penalty mapped directly into the Primal space {cite:p}`doumeche2025physics`.

## Theoretical Critique: The Optimization Failure of PINNs

The canonical approach to physics-informed machine learning, PINNs {cite:p}`raissi2019physics`, attempts to enforce differential equations by adding the PDE residual to the neural network's empirical loss function. However, this creates a notoriously pathological optimization landscape.

As noted in the PIKL literature {cite:p}`doumeche2025physics`, enforcing physical laws via backpropagation requires computing continuous second or third-order derivatives of the network with respect to its inputs during every training step. This process is computationally expensive, highly sensitive to initialization, and frequently succumbs to gradient pathologies (e.g., stiffness between the data loss and the physics loss), leading to convergence failures. By abandoning gradient descent entirely in favor of an exact, closed-form kernel regression solver, TAM perfectly guarantees global optimality and eliminates the need for iterative physics-loss balancing.

## The Theoretical Justification: From RKHS to the Primal Space

The connections between partial differential equations and kernel methods are mathematically well-established in functional analysis and meshless methods {cite:p}`schaback2006kernel`. Building on this foundation, Doumèche et al. proved that for any linear partial differential equation, the Physics-Informed Kernel Learning (PIKL) problem can be solved exactly by redefining the norm of the Reproducing Kernel Hilbert Space (RKHS) {cite:p}`doumeche2025physics`.

To embed a physical linear operator $\mathcal{L}$, the regularized empirical risk minimization problem is formulated as:

$$\min_{f \in \mathcal{H}} \sum_{i=1}^N (y_i - f(t_i))^2 + \lambda \int |\mathcal{L}(f)|^2 dt$$

By the Representer Theorem, the optimal function $f$ lies in a finite-dimensional span of the data. By explicitly projecting this into the TAM Primal space via the feature map $f(t) = \Phi(t) \theta$, the continuous integral of the physical residual reduces analytically to a strict quadratic form:

$$\int |\mathcal{L}(f)|^2 dt \approx \theta^\top P \theta$$

where $P$ is the explicit "Stiffness Matrix" encoding the differential equation.

## The Mathematical Generalization of the Stiffness Matrix

The construction of the penalty matrix $P$ depends strictly on how the continuous operator $\mathcal{L}$ interacts with the chosen functional basis.

### 1. The Fourier Basis (Spectral Diagonalization)

Doumèche originally focused on Fourier expansions because complex exponentials are the natural eigenfunctions of constant-coefficient differential operators {cite:p}`doumeche2025physics`.

Applying the continuous operator $\mathcal{L} = \sum w_n \frac{d^n}{dt^n}$ to a Fourier basis element $e^{i\omega t}$ yields:

$$\mathcal{L}(e^{i\omega t}) = \left( \sum_{n} w_n (i\omega)^n \right) e^{i\omega t}$$

The term in the parentheses is the characteristic polynomial of the PDE. Because the operator diagonalizes perfectly in the frequency domain, the penalty matrix $P$ is purely diagonal. For each frequency $\omega_j$, the penalty is the squared modulus of this polynomial:

$$P_{jj} = \lambda \left| \sum_{n} w_n (i \omega_j)^n \right|^2$$

This creates a perfect spectral filter: frequencies that naturally satisfy the PDE receive near-zero penalty, while frequencies that violate the physics are heavily suppressed.

### 2. P-Splines (Discrete Difference Operators)

To generalize this to strictly local bases like B-Splines, the framework maps continuous derivatives to discrete matrix operators. Let $I$ be the identity matrix and $D^{(k)}$ be the $k$-th order discrete difference operator acting on the spline coefficients $\theta$.

The continuous derivative of the spline function $f^{(k)}(t)$ is approximated by $D^{(k)} \theta$. The combined discrete differential operator becomes:

$$L_{op} = \sum_{k=0}^{n} w_k D^{(k)}$$

The physical residual is the Euclidean norm of this transformed coefficient vector, resulting in the Gramian penalty matrix:

$$P = \lambda L_{op}^\top L_{op}$$

Because $D^{(k)}$ matrices are highly banded and sparse, $P$ remains a sparse positive semi-definite matrix, preserving the $\mathcal{O}(ND)$ computational efficiency of local splines.

### 3. Neural Random Features (NNGP & RFF)

Applying physical PDE constraints to deep neural representations requires us to distinguish between two distinct topological architectures within the `NeuralEffect`:

**A. Shallow Random Fourier Features (Analytical Diagonalization)**
If the neural topology is restricted to a single hidden layer utilizing sinusoidal activations, the network operates strictly as a Random Fourier Feature (RFF) expansion: $\phi_j(t) = \cos(W_j t + b_j)$, where $W_j$ are fixed random weights drawn from a specific prior. 

Because the activation is purely harmonic and single-layered, taking the $n$-th continuous derivative analytically brings down $n$ powers of the known random weight $W_j$. Similar to the exact Fourier basis, these shallow neurons are independent, making the physical operator perfectly diagonal:

$$P_{jj} = \lambda \left| \sum_{n} w_n W_j^n \right|^2$$

This brilliantly enforces physics directly at the initialization phase. Random neurons whose sampled weights $W_j$ naturally align with the roots of the PDE's characteristic equation receive near-zero penalty and are preserved by the solver, while incompatible neurons are heavily suppressed.

**B. Deep Explicit Primal Tensorization (Autograd Gramian)**
If the framework utilizes the deep, multi-layer NNGP approximation formalized in the `NeuralEffect` (where $\phi_{neural}(x) = h^{(L)}(x)$ via sequential non-linearities like SiLU or Tanh), the simple analytical polynomial shortcut fails due to the complex chain rule required across $L$ layers. 

For deep EPT networks, the TAM framework abandons the diagonal approximation and instead computes the exact physical residual over the empirical grid. Let $\Phi_{neural}$ be the continuous design matrix of the frozen deep network. The continuous differential operator $L_{op}(\Phi_{neural})$ is computed analytically using PyTorch's native hardware-accelerated automatic differentiation (`torch.autograd.functional.jacobian` and `hessian`) with respect to the input $t$. 

The penalty matrix is then constructed as the exact, dense empirical Gramian of these gradients:

$$P = \lambda L_{op}(\Phi_{neural})^\top L_{op}(\Phi_{neural})$$

While this results in a dense penalty block, because the deep weights are strictly frozen, this Jacobian projection is computed only **once** prior to training. This allows deep, highly compositional neural embeddings to be constrained by exact differential equations and solved optimally via Sparse Conjugate Gradients without requiring iterative, unstable backpropagation during the fitting phase.

## A Concrete Physics Example: The Damped Harmonic Oscillator

To demonstrate the power of this explicit matrix formulation, consider the canonical Damped Harmonic Oscillator:

$$m \frac{d^2f}{dt^2} + c \frac{df}{dt} + k f = 0$$

In standard PINNs, enforcing this requires computing second-order gradients via backpropagation during every training step, which is notoriously unstable {cite:p}`raissi2019physics`.

In the TAM framework, we translate this into the linear operator $\mathcal{L}$ with weights $w_0 = k$, $w_1 = c$, and $w_2 = m$.

If we evaluate this using the Fourier basis, the characteristic polynomial penalty $P_{jj}$ at frequency $\omega_j$ becomes:

$$P_{jj} = \lambda \left( (-m \omega_j^2 + k)^2 + (c \omega_j)^2 \right)$$

This equation reveals the exact mathematical behavior of the solver. The penalty $P_{jj}$ reaches its absolute minimum when $-m \omega_j^2 + k = 0$.
This means the matrix inversion mathematically forces the statistical model to select the physical resonant frequency:

$$\omega_0 = \sqrt{\frac{k}{m}}$$

The framework analytically computes this optimal physical geometry once, solving the constrained system in a single step using the formula syntax: `phys(time, basis='fourier', D0=k, D1=c, D2=m)`.

## What exactly is the feature map $\Phi_{phys}(x)$?

A critical realization in the TAM architecture is that a "Physics Effect" does not require a novel feature map.

Mathematically:

$$\Phi_{phys}(x) = \Phi_{base}(x)$$

Whether the chosen basis is a Spline, a Fourier series, or a Neural RFF, the design matrix $\Phi(x)$ evaluated on the empirical data remains exactly identical to the standard implementation of that basis.

The physics constraint is enforced entirely in the Dual-to-Primal inversion:

$$\hat{\theta} = \left( \Phi^\top \Phi + N \cdot P_{phys} \right)^{-1} \Phi^\top Y$$

By substituting the standard statistical smoothing penalty with the analytical physical stiffness matrix $P_{phys}$, the solver forces the geometric prior of the coefficients $\theta$ into the specific manifold defined by the differential equation. The feature map defines the capacity of the model, while $P_{phys}$ defines the strict physical rules it must obey.