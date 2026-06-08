
# The Spectral Dictionary: PyTorch Implementations & Factory Assembly

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/01_primal_model.md)
  
**Core Interface:** `_base_effects.py`
**Factory Builder:** `_factory.py`

This document details the complete software engineering pipeline behind the TAM mathematical dictionary. For each effect, it exposes:
1. **The Factory Mapping**: How the text formula is parsed and instantiated.
2. **The Feature Map**: How the mathematical projection $\Phi(X)$ is computed.
3. **The Penalty Matrix**: How the structural constraint $P$ is built.

## The Base Contract and Factory Assembly

Every mathematical basis inherits from the abstract `BaseEffect` class. This ensures the core solver remains agnostic to the underlying topology, rigorously satisfying Aronszajn's conditions for a valid Reproducing Kernel Hilbert Space (RKHS) {cite:p}`aronszajn1950theory`.

```{literalinclude} ../../../../src/tam/model/spectrum/_base_effects.py
:language: python
:start-after: "#: <abstract_methods>"
:end-before: "#: </abstract_methods>"
```

The factory merges all distinct penalty matrices into a global block-diagonal system. To prevent memory exhaustion, it automatically coalesces dense blocks into a sparse coordinate tensor to safely bypass GPU VRAM limits.



```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <build_penalty>"
:end-before: "#: </build_penalty>"
```

---

## Linear 

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_linear>"
:end-before: "#: </parse_linear>"
```

**2. Feature Map & 3. Penalty Matrix**
The linear effect applies direct spatial scaling, and its penalty is a standard Ridge ($L_2$) scalar to stabilize the coefficients {cite:p}`hoerl1970ridge`.
```{literalinclude} ../../../../src/tam/model/spectrum/_linear.py
:language: python
:start-after: "#: <linear_effect>"
:end-before: "#: </linear_effect>"
```

---

## Fourier

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_fourier>"
:end-before: "#: </parse_fourier>"
```

**2. Feature Map**
Evaluates batched trigonometric functions, dynamically scaling angular frequencies based on the `cyclic` boundary flag to prevent endpoint distortions.
```{literalinclude} ../../../../src/tam/model/spectrum/_fourier.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Builds the purely diagonal Sobolev norm penalty scaled by $k^{2s}$, acting as an exact analytical low-pass filter {cite:p}`doumeche2025forecasting`.
```{literalinclude} ../../../../src/tam/model/spectrum/_fourier.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## P-Splines

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_spline>"
:end-before: "#: </parse_spline>"
```

**2. Feature Map**
Executes the recursive Cox-de Boor algorithm dynamically on the GPU.



```{literalinclude} ../../../../src/tam/model/spectrum/_spline.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Constructs the structural penalty natively using consecutive finite difference matrices `torch.diff(I)`.
```{literalinclude} ../../../../src/tam/model/spectrum/_spline.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Chebyshev

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_chebyshev>"
:end-before: "#: </parse_chebyshev>"
```

**2. Feature Map**
Evaluates the stable Chebyshev recurrence relation iteratively in-place. This bounds the global polynomial interpolation and explicitly mitigates Runge's phenomenon {cite:p}`rivlin1990chebyshev`.
```{literalinclude} ../../../../src/tam/model/spectrum/_chebyshev.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Applies a diagonal smoothness penalty mirroring the Fourier Sobolev norm.
```{literalinclude} ../../../../src/tam/model/spectrum/_chebyshev.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Categorical

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_categorical>"
:end-before: "#: </parse_categorical>"
```

**2. Feature Map**
Utilizes native one-hot embedding routing.
```{literalinclude} ../../../../src/tam/model/spectrum/_categorical.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Routes structurally: isotropic Ridge for `nominal`, and finite difference $D^\top D$ for `ordinal` features to enforce sequential class transitions.
```{literalinclude} ../../../../src/tam/model/spectrum/_categorical.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Radial Basis Function (RBF)

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_rbf>"
:end-before: "#: </parse_rbf>"
```

**2. Feature Map**
Computes pairwise Euclidean distances to centroids natively via `torch.cdist` before applying the Gaussian or Matérn kernel {cite:p}`williams2006gaussian`.
```{literalinclude} ../../../../src/tam/model/spectrum/_rbf.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Applies an isotropic identity penalty over the spatial prototypes.
```{literalinclude} ../../../../src/tam/model/spectrum/_rbf.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Wavelets

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_wavelet>"
:end-before: "#: </parse_wavelet>"
```

**2. Feature Map**
Leverages massive `.unsqueeze()` tensor broadcasting to compute the entire time-scale grid for the Ricker wavelet in a single pass {cite:p}`torrence1998practical`.
```{literalinclude} ../../../../src/tam/model/spectrum/_wavelet.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Enforces the sparsity prior via a scale-dependent diagonal "whitening" penalty to isolate transient shocks {cite:p}`donoho1994ideal`.
```{literalinclude} ../../../../src/tam/model/spectrum/_wavelet.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Neural

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_neural>"
:end-before: "#: </parse_neural>"
```

**2. Feature Map**
Projects features through a frozen Neural Network.



```{literalinclude} ../../../../src/tam/model/spectrum/_neural.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Applies an isotropic Ridge penalty ($L_2$) to the final linear readout coefficients.
```{literalinclude} ../../../../src/tam/model/spectrum/_neural.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Tree / Random Forest

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_tree>"
:end-before: "#: </parse_tree>"
```

**2. Feature Map**
Converts Oblivious Trees into sparse Euclidean bit-strings, modeling Random Binning Features {cite:p}`wu2016revisiting`. Scales the output destructively in-place using `.mul_()` to safely bypass memory limits.



```{literalinclude} ../../../../src/tam/model/spectrum/_tree.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Safely instantiates the Ridge penalty over the terminal leaves strictly as a sparse COO tensor.
```{literalinclude} ../../../../src/tam/model/spectrum/_tree.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Tensor Product

**1. Factory Parsing**
Parses the `te()` token recursively to instantiate the nested sub-effects.
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_tensor>"
:end-before: "#: </parse_tensor>"
```

**2. Feature Map**
Generates the interaction surface using row-wise Kronecker products via `torch.einsum`.



```{literalinclude} ../../../../src/tam/model/spectrum/_tensor.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Preserves marginal regularization independence via the Kronecker sum.
```{literalinclude} ../../../../src/tam/model/spectrum/_tensor.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

---

## Universal Physics (PIKL)

**1. Factory Parsing**
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_physics>"
:end-before: "#: </parse_physics>"
```

**2. Feature Map**
Delegates the feature extraction entirely to the underlying differentiable topology (Spline, Fourier, Neural).
```{literalinclude} ../../../../src/tam/model/spectrum/_physics.py
:language: python
:start-after: "#: <init_physics>"
:end-before: "#: </init_physics>"
```

**3. Penalty Matrix**
Enforces the explicit linear differential operator by constructing the analytic stiffness matrix $P$, circumventing the instability of PINNs via exact Physics-Informed Kernel Learning {cite:p}`doumeche2025physics`.
```{literalinclude} ../../../../src/tam/model/spectrum/_physics.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

## PID (Autoregressive Control)

### The PID effect code

**1. Factory Parsing**
Parses the `pid()` token to extract the target lag feature, the look-back window for the rolling integral, and the artificial stiffness multiplier for the derivative action.
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_pid>"
:end-before: "#: </parse_pid>"
```

**2. Feature Map**
Projects endogenous target lags into a discrete Proportional-Integral-Derivative control space. It executes strictly via in-place, vectorized tensor operations (using `torch.diff` for the derivative and a padded `cumsum` for the integral rolling mean) to prevent VRAM-heavy matrix cloning.
```{literalinclude} ../../../../src/tam/model/spectrum/_pid.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

**3. Penalty Matrix**
Constructs a $3 \times 3$ diagonal stiffness matrix. Uniquely applies an artificial structural boost (`d_penalty_multiplier`) to the derivative term to rigorously enforce low-pass filtering and prevent the amplification of high-frequency stochastic noise {cite:p}`aastrom2021feedback`.
```{literalinclude} ../../../../src/tam/model/spectrum/_pid.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```

### Control Diagnostics (Bode Stability)

**Core Interface:** `bode.py`

Because the `pid()` effect introduces an endogenous closed-loop dynamical system, the framework provides a native Control Diagnostics suite. This module extracts the learned structural weights and maps them to the frequency domain via the Z-transform, explicitly generating Bode plots and proving Bounded-Input Bounded-Output (BIBO) stability. 

**1. Public API**
Exposes the `plot_bode` method, which dynamically routes requests for both static PID constraints and localized Gain-Scheduled (Tensor Product) controllers.
```{literalinclude} ../../../../src/tam/model/bode.py
:language: python
:start-after: "#: <plot_bode_api>"
:end-before: "#: </plot_bode_api>"
```

**2. Physical Unscaling**
Reverses the Primal feature map normalization (the "Normalization Illusion") to extract the exact physical control weights ($K_p, K_i, K_d$). It rigorously slices the multi-dimensional tensor product space if conditioned on exogenous variables (like temperature).
```{literalinclude} ../../../../src/tam/model/bode.py
:language: python
:start-after: "#: <helpers_extraction>"
:end-before: "#: </helpers_extraction>"
```

**3. Digital Filter Construction & Rendering**
Builds the discrete transfer function polynomial, dynamically distributing the integral term over the rolling window. It renders the Bode phase and gain margins and automatically verifies that all complex poles remain strictly inside the unit circle.
```{literalinclude} ../../../../src/tam/model/bode.py
:language: python
:start-after: "#: <render_bode>"
:end-before: "#: </render_bode>"
```


## Linear Tree (Varying-Coefficient Trees)

**Related Theory:** [See the Mathematical Definition](../../math/spectrum/LINEAR_TREE.md)

**1. Factory Parsing**
The parser explicitly extracts the local slope feature and enforces `n_trees=1` to mathematically prevent overlapping collinearity during the Primal resolution. It then directly instantiates the native composite effect.
```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_linear_tree>"
:end-before: "#: </parse_linear_tree>"
```

**2. Feature Map**
Natively encapsulates the base `TreeEffect` (local intercept) and computes the Kronecker product of the slope tree with the `LinearEffect` to generate the localized gradients.

```{literalinclude} ../../../../src/tam/model/spectrum/_linear_tree.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"

```

**3. Penalty Matrix**
Constructs the block-diagonal encapsulation of the isotropic tree penalty and the anisotropic tensor penalty, safely coalescing them into a sparse COO tensor.

```{literalinclude} ../../../../src/tam/model/spectrum/_linear_tree.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"

```
