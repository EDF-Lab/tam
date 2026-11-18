# Singular Spectrum Analysis: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See SSA.md](../../math/spectrum/SSA.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Singular Spectrum Analysis is exposed via `ssa()`:

```python
# Singular Spectrum Analysis (SSA) causal Hankel EOF decomposition
formula = "Y ~ ssa(x1, window=72, k=4)"
```

## Parameter Specification

### `SSAEffect` (`ssa()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `window` | `int` | `48` | Hankel embedding window length $L$. |
| `k` | `int` | `10` | Top empirical orthogonal functions (EOFs) retained. |

## Complexity & Performance

- $\mathcal{O}(B\,T\,L\,k)$ time and memory (one windowed matrix product); the EOFs come from one $L \times L$ eigendecomposition at initialization.

## State, Complexity and Limits

- **Class and file**: `SSAEffect`, `src/tam/model/spectrum/_ssa.py`; formula token `ssa()`. Label: (BETA).
- **State**: `{"hist": (B, window - 1, 1)}`. **Layout**: `(B, T)`, one column.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction.
- **Limits**: the EOFs are those of the training period; a regime with another dominant period projects on them with a small amplitude.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `SSAEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_ssa.py
:language: python
:start-after: "#: <ssa_feature_map>"
:end-before: "#: </ssa_feature_map>"
```

Penalty of `SSAEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_ssa.py
:language: python
:start-after: "#: <ssa_penalty_matrix>"
:end-before: "#: </ssa_penalty_matrix>"
```

Parsing of the `ssa(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_ssa>"
:end-before: "#: </parse_ssa>"
```
