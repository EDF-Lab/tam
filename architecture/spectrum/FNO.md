# Fourier Neural Operator: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See FNO.md](../../math/spectrum/FNO.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

The Fourier Neural Operator block is exposed via `fno()`:

```python
# Resolution-invariant 1D Fourier Neural Operator spectral block
formula = "Y ~ fno(temperature, modes=6, width=6)"
```

## Parameter Specification

### `OperatorEffect` (`fno()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `modes` | `int` | `12` | Truncated Fourier frequency modes $k_{\max}$. |
| `width` | `int` | `8` | Number of spectral channels (and of columns). |
| `window` | `int` | `32` | Causal sliding-window length (`W` is accepted as an alias). |
| `causal` | `bool` | `True` | Read each step through its causal window (`True`) or the whole pass at once (`False`). |

## Complexity & Performance

- **Spectral synthesis**: `irfft` over the truncated modes, in $\mathcal{O}(B\,T\,W\log W)$ for the causal form and $\mathcal{O}(B\,T\log T)$ for the non-causal one.

## State, Complexity and Limits

- **Class and file**: `OperatorEffect`, `src/tam/model/spectrum/_fno.py`; formula token `fno()`. Label: (BETA).
- **State**: `{"hist": (B, window - 1, 1)}` for the causal form; none for `causal=False`. **Layout**: `(B, T)`, one column.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction for the causal form.
- **Limits**: the operator weights are random and fixed (`seed`): two seeds give two different feature maps.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `OperatorEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_fno.py
:language: python
:start-after: "#: <fno_feature_map>"
:end-before: "#: </fno_feature_map>"
```

Penalty of `OperatorEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_fno.py
:language: python
:start-after: "#: <fno_penalty_matrix>"
:end-before: "#: </fno_penalty_matrix>"
```

Parsing of the `fno(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_fno>"
:end-before: "#: </parse_fno>"
```
