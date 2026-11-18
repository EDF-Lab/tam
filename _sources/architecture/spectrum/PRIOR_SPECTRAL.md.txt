# Prior Spectral Effect: PyTorch Architecture & Implementation

**Navigation:**
  * **Mathematical Theory:** [See PRIOR_SPECTRAL.md](../../math/spectrum/PRIOR_SPECTRAL.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

The Prior Spectral effect is exposed in formulas via the `pspec()` token:

```python
# Univariate Prior Spectral effect with Matern-3/2 prior
formula = "Y ~ pspec(X, k=64, prior='matern', nu=1.5, length_scale=1.0)"

# Empirical periodogram prior estimated on short context
formula = "Y ~ pspec(time, k=32, prior='empirical')"

# Multivariate interaction with spatial covariates
formula = "Y ~ pspec(lat, others='lon', k=64, prior='matern', nu=2.5)"

# Tensor product interaction
formula = "Y ~ te(pspec(X1, k=16), s(X2, k=16))"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Primary feature column name. |
| `k` | `int` | `64` | Number of frequencies $M$. Generates $2M$ feature columns ($M$ cosines, $M$ sines). |
| `prior` | `str` or `Callable` | `'matern'` | Spectral prior: `'matern'`, `'lorentzian'`, `'empirical'`, or custom inverse-CDF callable. |
| `nu` | `float` | `1.5` | Matérn smoothness parameter ($\nu=0.5$ is Lorentzian/Cauchy). |
| `length_scale` | `float` | `1.0` | Characteristic length scale $\ell$. |
| `sobolev_order` | `float` | `1.0` | Sobolev roughness penalty order $s$. Weights scale as $(1 + \|\omega_m\|_2^2)^s$. |
| `qmc` | `str` | `'halton'` | Low-discrepancy sequence: `'halton'`, `'sobol'`, or `'mc'`. |
| `seed` | `int` | `42` | Seed for low-discrepancy sampling. |
| `others` | `str` or `list` | `None` | Secondary feature names for multivariate inputs (separated by `\|`). |
| `ap` | `float` | `default_alpha_p` | Log10 of the regularization parameter $\lambda_p = 10^{\text{ap}}$. |
| `extrapolate` | `str` | `'continue'` | OOD behavior beyond $[-1, 1]$ (`'continue'`, `'constant'`, `'linear'`). |

## Complexity & Performance

- **Memory Complexity**: The feature map requires $\mathcal{O}(T \cdot 2M)$ floats. The diagonal penalty matrix requires $\mathcal{O}(2M)$ non-zero storage.
- **Compute Complexity**: Feature map evaluation executes in $\mathcal{O}(T \cdot M \cdot d)$ via single tensor matrix multiplication on CPU or GPU.
- **Solver Invariance**: Solved directly within TAM's closed-form $\mathcal{O}(T D^2)$ P-WLS step, fully compatible with GCV and REML tuning.

## State, Complexity and Limits

- **Class and file**: `PriorSpectralEffect`, `src/tam/model/spectrum/_prior_spectral.py`. Label: (BETA).
- **State**: none. **Layout**: any shape; `(B, T, d)` with `others=`. **Time and memory**: $\mathcal{O}(B\,T\,k\,d)$.
- **Frozen at initialization**: the frequencies (for `prior='empirical'` from the periodogram of the training values, groups concatenated).
- **Extrapolation**: all modes of the router are available.
- **Limits**: the empirical periodogram of concatenated groups sees the junctions between groups as discontinuities; for panels with very different series give the prior explicitly.

## Code

Feature map of `PriorSpectralEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_prior_spectral.py
:language: python
:start-after: "#: <pspec_feature_map>"
:end-before: "#: </pspec_feature_map>"
```

Penalty of `PriorSpectralEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_prior_spectral.py
:language: python
:start-after: "#: <pspec_penalty_matrix>"
:end-before: "#: </pspec_penalty_matrix>"
```

Parsing of the `pspec(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_prior_spectral>"
:end-before: "#: </parse_prior_spectral>"
```
