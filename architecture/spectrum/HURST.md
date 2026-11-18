# Hurst Fractional Gaussian Noise Basis: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See HURST.md](../../math/spectrum/HURST.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

The Hurst memory basis is exposed via `hurst()`:

```python
# Fractional Gaussian noise Hurst memory basis
formula = "Y ~ hurst(Load_d1, k=12, H=0.75)"
```

## Parameter Specification

### `HurstEffect` (`hurst()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k` | `int` | `16` | Number of columns ($k/2$ sine and $k/2$ cosine harmonics); harmonic $j$ is penalized by $j^{2H-1}$, the inverse spectrum of fractional Gaussian noise. |
| `H` | `float` | `0.75` | Hurst exponent $H \in (0, 1)$. |
| `period` | `float` | `None` | Optional fixed cycle period. |

## Complexity & Performance

- Pointwise in the value: $\mathcal{O}(B\,T\,k)$ time and memory; the training span is frozen at initialization.

## State, Complexity and Limits

- **Class and file**: `HurstEffect`, `src/tam/model/spectrum/_hurst.py`; formula token `hurst()`. Label: (BETA).
- **State**: none. **Layout**: `(B, T)` or `(T,)`; one column.
- **Extrapolation**: all modes of the router are available.
- **Limits**: the penalty is a prior, not an estimate of the memory of the series; $H$ is clipped to $[0.05, 0.95]$.

## Code

Feature map of `HurstEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_hurst.py
:language: python
:start-after: "#: <hurst_feature_map>"
:end-before: "#: </hurst_feature_map>"
```

Penalty of `HurstEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_hurst.py
:language: python
:start-after: "#: <hurst_penalty_matrix>"
:end-before: "#: </hurst_penalty_matrix>"
```

Parsing of the `hurst(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_hurst>"
:end-before: "#: </parse_hurst>"
```
