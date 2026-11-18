# HiPPO State Space Model Effect: PyTorch Architecture & Implementation

**Navigation:**
  * **Mathematical Theory:** [See HIPPO_SSM.md](../../math/spectrum/HIPPO_SSM.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

The HiPPO effect is exposed in formulas via the `hippo()` token:

```python
# Univariate HiPPO memory with translated Legendre (sliding window)
formula = "Y ~ hippo(temperature, k=16, measure='legt', dt=0.05)"

# Full-history scaled Legendre with ZOH discretization
formula = "Y ~ hippo(temperature, k=12, measure='legs', dt=0.02, discretization='zoh')"

# Combined with non-linear spline and tensor interaction
formula = "Y ~ s(temperature, k=10) + hippo(temperature, k=12) + te(hippo(temperature, k=8), c(day_type))"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Primary feature column name. |
| `k` | `int` | `16` | State dimension / order of orthogonal polynomial projection $N$. |
| `measure` | `str` | `'legt'` | HiPPO polynomial measure: `'legt'` (sliding window), `'legs'` (scale-invariant), or `'lagt'` (Laguerre). |
| `dt` | `float` | `0.05` | Discretization time-step $\Delta$. |
| `discretization` | `str` | `'bilinear'` | Discretization method: `'bilinear'` (Tustin transform) or `'zoh'` (Zero-Order Hold via Van Loan matrix exponential). |
| `degree_pen` | `float` | `1.0` | Polynomial Dirichlet degree penalty exponent $p$ in $\operatorname{diag}(1 + (n(n+1))^p)$. |
| `ap` | `float` | `default_alpha_p` | Log10 of the regularization parameter $\lambda_p = 10^{\text{ap}}$. |
| `extrapolate` | `str` | `'continue'` | OOD extrapolation strategy beyond $[-1, 1]$. |

## Complexity & Performance

- **Memory Complexity**: The feature map requires $\mathcal{O}(T \cdot N)$ floats. The diagonal penalty matrix requires $\mathcal{O}(N)$ storage.
- **Compute Complexity**: Feature map evaluation executes in $\mathcal{O}(N T \log T)$ via parallelized 1D causal zero-padded FFT convolutions (`torch.fft.rfft` / `torch.fft.irfft`) across all groups.
- **State Continuation**: The terminal state $s_{T-1} \in \mathbb{R}^{G \times N}$ is cached upon completion of training, allowing subsequent batch and online prediction frames to continue the recurrence smoothly without boundary resets.

## State, Complexity and Limits

- **Class and file**: `HiPPOEffect`, `src/tam/model/spectrum/_hippo.py`. Label: (BETA).
- **State**: `{"c": (B, k)}`. **Layout**: `(B, T)` in, `(B, T, k)` out.
- **Time and memory**: $\mathcal{O}(B\,T\,k\log T)$ with the FFT convolution (passes of 64 steps or more) plus $\mathcal{O}(T\,k^2)$ for the kernel; the loop is used below 64 steps.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction.
- **Limits**: one input column; the kernel is built for the length of the pass, so very long passes (hundreds of thousands of steps) are better cut in chunks continued with `continue_state=True`.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `HiPPOEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_hippo.py
:language: python
:start-after: "#: <hippo_feature_map>"
:end-before: "#: </hippo_feature_map>"
```

Penalty of `HiPPOEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_hippo.py
:language: python
:start-after: "#: <hippo_penalty_matrix>"
:end-before: "#: </hippo_penalty_matrix>"
```

Parsing of the `hippo(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_hippo>"
:end-before: "#: </parse_hippo>"
```
