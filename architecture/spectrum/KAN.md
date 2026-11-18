# Kolmogorov-Arnold Network Edge Splines: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See KAN.md](../../math/spectrum/KAN.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Kolmogorov-Arnold edge splines are exposed via `kan()`:

```python
# Kolmogorov-Arnold Network edge splines with scale-stable inner nodes
formula = "Y ~ kan(x1, others='x2|x3', n_nodes=6, k_inner=6, k_outer=10)"
```

## Parameter Specification

### `KANEffect` (`kan()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `others` | `str` or `list` | `None` | Secondary features separated by `\|`. |
| `n_nodes` | `int` | `8` | Number of inner aggregation nodes $Q$. |
| `k_inner` | `int` | `6` | Inner edge spline knots per feature. |
| `k_outer` | `int` | `8` | Outer univariate spline knots per node. |
| `edge_pen` | `float` | `0.1` | Inner-edge roughness multiplier $\alpha_{\text{edge}}$ weighting $\mathcal{E}_q$. |

## Complexity & Performance

- **Linear Parameter Scaling**: `KANEffect` scales as $\mathcal{O}(Q k_{\text{outer}})$ coefficients instead of $\mathcal{O}(k^d)$ tensor products, with per-feature standardization and frozen $(\hat{\mu}_q, \hat{\sigma}_q)$ pre-tanh normalization preventing saturation.

## State, Complexity and Limits

- **Class and file**: `KANEffect`, `src/tam/model/spectrum/_kan.py`. Label: (BETA).
- **State**: none (`stateful = False`).
- **Layout**: `(B, T, d)` or `(N, d)` rows (a tensor-product margin); one column may also be given as `(B, T)`. One row of the input gives one row of features.
- **Time and memory**: $\mathcal{O}(B\,T\,Q\,(d\,k_{\text{in}} + k_{\text{out}}))$; the groups are rows of one batch.
- **Extrapolation**: all modes of the router are available (the effect has no memory).
- **Limits**: the inner edges are drawn from `seed` and not trained: two seeds give two different, equally valid, feature maps; use `seed` to reproduce a fit.
- **Penalty ridge**: each node block is `weight * D2.T @ D2 + 1e-6 * I` (`eps_ridge` in `build_penalty_matrix`). The ridge is kept on purpose: it makes the penalty positive definite so that the REML log-determinant is defined, and it is part of the numbers of the benchmarks.

## Code

Feature map of `KANEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_kan.py
:language: python
:start-after: "#: <kan_feature_map>"
:end-before: "#: </kan_feature_map>"
```

Penalty of `KANEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_kan.py
:language: python
:start-after: "#: <kan_penalty_matrix>"
:end-before: "#: </kan_penalty_matrix>"
```

Parsing of the `kan(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_kan>"
:end-before: "#: </parse_kan>"
```
