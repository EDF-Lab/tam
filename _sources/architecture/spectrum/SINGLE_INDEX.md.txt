# Penalized Single-Index Distributed Lags: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See SINGLE_INDEX.md](../../math/spectrum/SINGLE_INDEX.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Single-index distributed lags are exposed via `si()`:

```python
# Semiparametric Single-Index distributed-lag B-spline
formula = "Y ~ si(temperature, lag_window=12, k=6)"
```

## Parameter Specification

### `SingleIndexEffect` (`si()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `others` | `str` or `list` | `None` | Multivariate covariates (mutually exclusive with `lag_window`). |
| `lag_window` | `int` | `None` | Distributed-lag window length for univariate inputs. |
| `k` | `int` | `10` | Univariate outer B-spline knots on normalized index $u_t = X_t^\top \alpha$. |

## Complexity & Performance

- **Boundary Safety**: `SingleIndexEffect` caches `self.u_bounds` at fit time and applies linear boundary extrapolation outside $[-1, 1]$.

## State, Complexity and Limits

- **Class and file**: `SingleIndexEffect`, `src/tam/model/spectrum/_single_index.py`. Label: (BETA).
- **State**: `{"hist": (B, lag_window - 1, 1)}` for the distributed-lag form only (`stateful` is true only then); none for `others=`.
- **Layout**: `(B, T, d)` (`keeps_feature_dim`); a batch of one column may also be given as `(B, T)`.
- **Time and memory**: $\mathcal{O}(B\,T\,L)$ for the lag form (one windowed dot product per step), $\mathcal{O}(B\,T\,d)$ for the block; no loop over groups.
- **Extrapolation**: the lag form refuses `'linear'` and `'saturation'` at construction (they would need a finite difference on isolated steps). The spline itself continues linearly outside the index range.
- **Limits**: $\alpha$ is fixed before the solve: the coefficients are linear, the index is not re-fitted.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `SingleIndexEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_single_index.py
:language: python
:start-after: "#: <si_feature_map>"
:end-before: "#: </si_feature_map>"
```

Penalty of `SingleIndexEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_single_index.py
:language: python
:start-after: "#: <si_penalty_matrix>"
:end-before: "#: </si_penalty_matrix>"
```

Parsing of the `si(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_single_index>"
:end-before: "#: </parse_single_index>"
```
