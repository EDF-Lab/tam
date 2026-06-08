# The Theory of N-Dim Broadcasting and Normalization

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/02_data_pipeline.md)
  * **Related topic:** [The Exact Primal Resolution](01_primal_model.md)
  
This chapter details the mathematical and geometric preparation of data before its injection into the Primal solver. To guarantee computational efficiency on hardware accelerators (GPUs) and the numerical stability of the regression, raw data (heterogeneous time series) must be projected into a strict tensor space and a normalized continuous domain.

---

## Group Normalization and the $[-1, 1]$ Domain

In the TAM framework, input variables are not injected at their raw scale. They undergo a rigorous affine transformation to bound them within the $[-1, 1]$ interval. Unlike standard statistical scaling (zero mean, unit variance) or $[0, 1]$ min-max scaling, the $[-1, 1]$ domain is a fundamental mathematical prerequisite for several topological reasons:

1. **Polynomial Orthogonality & Minimax Properties:** The standard Chebyshev polynomials are fundamentally orthogonal on $[-1, 1]$ with respect to the weight $(1-x^2)^{-1/2}$ {cite:p}`rivlin1990chebyshev`. Their unique minimax property, acting as the monic polynomial of least deviation from zero, is strictly confined to this interval. Attempting to extrapolate outside $[-1, 1]$ destroys this optimal convergence due to the rapid exponential growth of the polynomials, while inside the interval, evaluating at Chebyshev nodes is necessary to suppress the Runge phenomenon associated with uniform grids.

2. **Numerical Stability and Matrix Conditioning:** While LeCun’s theoretical analyses and heuristics {cite:p}`lecun1998efficient` were originally aimed at accelerating gradient descent, the mathematical requirement for a well-conditioned covariance matrix remains absolute for exact Primal inversion. A zero-centered, bounded domain restricts the condition number $\kappa(A)$ (the ratio of maximum to minimum eigenvalues) of the global design matrix $(\Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g)$. This prevents catastrophic floating-point cancellation during GPU matrix inversion and drastically accelerates the convergence rate of the sparse Conjugate Gradient solver. Furthermore, when utilizing Random Fourier Features to approximate shift-invariant kernels via explicit $\cos(\omega^\top X + b)$ mappings {cite:p}`rahimi2007random`, restricting inputs to $[-1, 1]$ prevents numerical overflow inside the trigonometric functions. This hardware-aware scaling is required to preserve the integrity of the approximated spectral measure.


The transformation is applied independently for each independent group $g \in G$ to preserve local dynamics. Let $X_g$ be the raw feature vector for a specific group. The framework calculates the local amplitude $A_g$ and the local center $C_g$:

$$A_g = \max(X_g) - \min(X_g)$$
$$C_g = \frac{\max(X_g) + \min(X_g)}{2}$$

The normalized projection $\tilde{X}_g$ is then computed as:

$$\tilde{X}_g = \frac{X_g - C_g}{A_g / 2.0}$$

*(Note: If a feature is constant, meaning $A_g = 0$, the amplitude is strictly overridden to $1.0$ to prevent division by zero, mapping the constant entirely to $0.0$.)*

---

## The 3D Tensor Manifold

Once normalized, the data is removed from its flat tabular format (e.g., Pandas DataFrames) and reshaped into a rigid 3D tensorial structure. This allows the mathematical dispatcher to solve independent equations simultaneously using PyTorch's batched linear algebra functions. 

The resulting raw input tensor $\mathcal{X}$ possesses the strictly defined dimensions $(G, T, F)$, which is subsequently mapped into the Primal Design Matrix $\boldsymbol{\Phi}$ of dimensions $(G, T, D)$, where:

* $G$: The total number of independent groups (e.g., unique smart meters, individual patients, or geographical regions).
* $T$: The number of temporal observations (time steps) per group.
* $F$: The number of raw exogenous input features.
* $D$: The Primal Dimension (the total number of evaluated functional bases or degrees of freedom).

### The Transitivity Shield and Masking ($\Lambda_g$)

This rigid matrix representation strictly requires that all groups possess the exact same number of temporal observations $T$. Because real-world time series are often asynchronous, of varying lengths, or entirely cross-sectional, the framework employs an automated "Transitivity Shield".

Before tensorization, the framework balances the groups. During the training phase (`fit`), the algorithm drops surplus rows, forcing all groups to match the temporal length of the shortest group. During the inference phase (`predict`), the framework switches to a "fill" strategy, padding the tensors with `NaN`s to align the manifold without discarding any user data.

Mathematically, these padded `NaN`s are functionally isolated via the diagonal observation weighting matrix $\Lambda_g \in \mathbb{R}^{T \times T}$ introduced in the core equations. If a time step $t$ is padded, its corresponding diagonal entry is set to zero ($\Lambda_{g, t,t} = 0$). This strictly annihilates the invalid timestamp during the inner product accumulation $\Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g$, maintaining the GPU memory contiguity without corrupting the exact Primal resolution.

> *(Note: To prevent IEEE 754 floating-point `NaN` propagation where $0 \times \text{NaN} = \text{NaN}$, the PyTorch implementation strictly replaces padded features with safe dummy values, or uses boolean index masking, prior to the $\Lambda_g$ dot product).*

---

## Adaptive Windowing (Rolling Origin)

For meta-learning algorithms dealing with data drift and the violation of exchangeability in time series, classic Train/Test split evaluation, and static data normalization, is insufficient, as standard global methods fundamentally assume a stationary process {cite:p}`gamakumara2023conditional`. To address this, the framework models time as a continuous flow using a closed-form, tensorial alternative to iterative Kalman filters for managing structural breaks, formalizing the online adaptation approach {cite:p}`doumeche2025forecasting`.

Rather than using a simple sliding window, the Adaptive solver coordinates three specific temporal hyperparameters:
* $W_{train}$ (Learning Size): The strict number of historical steps used to train the system.
* $W_{pred}$ (Window Size): The number of steps the model will continuously predict before triggering an update.
* $H$ (Horizon): The required forecasting horizon offset.

Mathematically, for a given simulation step $t$, the training subset $\mathcal{D}_t$ is defined by reversing the horizon offset and extracting the historical learning size:

$$\mathcal{D}_t = \{ (X_{t-W_{train}-H+1 : t-H+1},\; Y_{t-W_{train}-H+1 : t-H+1}) \}$$

And the prediction is evaluated purely on the forward window:

$$\mathcal{P}_t = \{ X_{t : t+W_{pred}} \}$$

Rather than iterating temporally (which is computationally slow in Python), the framework vectorizes these subsets. It calculates the reverse chronological start indices and constructs integer offset tensors (`train_offsets` and `predict_offsets`). Using advanced PyTorch tensor indexing, it gathers all training and prediction windows across the entire timeline in a single, hardware-accelerated memory operation. 

This transforms the continuous time dimension into a massive, discrete batch dimension, delivering perfectly aligned subsets directly into the exact Primal or Conjugate Gradient solver.

## Boundary Conditions and OOD Extrapolation

Because the global `[-1, 1]` affine normalization relies on historical minimums and maximums, extreme Out-Of-Distribution (OOD) shocks (e.g., an unprecedented heatwave) will push normalized inputs beyond the $\pm 1.0$ limit. If left unmanaged, this causes "Topology Roulette": Chebyshev polynomials would explode, Fourier series would dangerously wrap around their cyclical torus, and Splines would decay to zero (blinding the model).

To guarantee mathematical safety in production, the framework implements a **Universal Extrapolation Wrapper**. Rather than failing unpredictably, the user can explicitly dictate the OOD boundary physics for any effect (via `extrapolate='continue' | 'constant' | 'linear' | 'saturation'`).

### The Mathematical Formulation (Finite Differences)

When a raw input feature vector $x \in \mathbb{R}^F$ (where $F$ is the dimension of the inputs for a specific effect) exceeds the $[-1, 1]^F$ hypercube, the wrapper intercepts the raw feature map $\Phi(x)$. It projects the point back to the nearest safe boundary, defined as $x_{bound} = \text{clamp}(x, -1, 1)$.

It then calculates the outward Euclidean distance $d = \|x - x_{bound}\|_2$ and the normalized outward unit vector $u = \frac{x - x_{bound}}{d}$.

To compute the directional derivative (the slope) exactly at the boundary without triggering OOD instability, the framework takes a microscopic numerical step ($\epsilon = 10^{-5}$) *strictly backward* into the safe hypercube:

$$D_u \Phi_{bound} \approx \frac{\Phi(x_{bound}) - \Phi(x_{bound} - \epsilon \cdot u)}{\epsilon}$$

Using this safe boundary value and directional derivative, the framework applies a First-Order Taylor Expansion modified by the requested extrapolation mode:

1. **`continue` (Raw Evaluation):** The framework bypasses the wrapper and evaluates the raw topological equation.

$$\hat{\Phi}(x) = \Phi(x)$$

> *(Warning: Unsafe for oscillatory or polynomial bases).*

2. **`constant` (Plateau / Hard Clamp):** The boundary slope is forced to zero, safely capping the forecasting momentum. Perfect for physical systems with hard capacity limits.

$$\hat{\Phi}(x) = \Phi(x_{bound})$$

3. **`linear` (Natural Extension):** Freezes the exact slope evaluated at the boundary and projects it to infinity. This universal First-Order Taylor Expansion forces any evaluated effect to behave identically to a classical Natural Cubic Spline, which is mathematically constrained to be linear beyond its boundary knots {cite:p}`hastie2009elements`.

$$\hat{\Phi}(x) = \Phi(x_{bound}) + d \cdot D_u \Phi_{bound}$$

4. **`saturation` (Asymptotic Damping):** Applies an exponential decay term ($\lambda = 2.0$) to the linear distance, forcing the slope to gracefully asymptote to a fixed ceiling. This provides the directionality of the `linear` mode but guarantees the ultimate stability of the `constant` mode.

$$\hat{\Phi}(x) = \Phi(x_{bound}) + D_u \Phi_{bound} \left( \frac{1 - e^{-\lambda d}}{\lambda} \right)$$

> 💡 **Theoretical Breakthrough: Extrapolating Trees**
> Crucially, because TAM isolates the algorithmic partition logic (the Tree) from the continuous geometric space (the RKHS) by embedding the discrete leaf partitions as Random Binning Features {cite:p}`wu2016revisiting`, this OOD Extrapolation applies to Decision Trees natively. Unlike standard Random Forests or Gradient Boosting, which are structurally constrained to piecewise-constant predictions and thus mathematically incapable of extrapolating beyond their observed training targets, the native `LinearTreeEffect` leverages this exact continuous geometry to project a smooth, saturated, or linear slope infinitely outside the $[-1, 1]$ bounding box.