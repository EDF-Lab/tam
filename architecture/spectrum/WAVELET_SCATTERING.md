# Wavelet Scattering Transform: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See WAVELET_SCATTERING.md](../../math/spectrum/WAVELET_SCATTERING.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

```python
formula = "Y ~ wst(temperature, J=3, Q=1, max_order=2, causal=True)"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Input signal column name. |
| `J` | `int` | `3` | Number of dyadic wavelet scales. |
| `Q` | `int` | `1` | Wavelets per octave. |
| `max_order` | `int` | `2` | Maximum scattering cascade order ($0$, $1$, or $2$). |
| `causal` | `bool` | `True` | Enforce strictly causal Morlet/Gabor wavelet convolutions. |
| `order_pen` | `float` | `4.0` | Exponential penalty multiplier across scattering orders. |

## Complexity & Performance

- **Causality & State Continuation**: the causal form carries the last `n_hist` samples of every group (`state_in` / `_state_out`) with exact continuation.
- **Compute**: a cascade of direct 1D convolutions in $\mathcal{O}(B\,T\,\ell)$ per channel ($\ell$ = filter length).

## State, Complexity and Limits

- **Class and file**: `ScatteringEffect`, `src/tam/model/spectrum/_scattering.py`; formula token `wst()`. Label: (BETA).
- **State**: `{"hist": (B, n_hist, 1)}` for the causal form; none for `causal=False`. **Layout**: `(B, T)`, one column.
- **Memory of the cascade**: `n_hist = (len(phi) - 1) + max_order * (len(psi_max) - 1)`.
- **Penalty**: diagonal order weights plus, on the order-1 block, the second-difference coupling along the octaves (needs three octaves or more).
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction for the causal form.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `ScatteringEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_scattering.py
:language: python
:start-after: "#: <wst_feature_map>"
:end-before: "#: </wst_feature_map>"
```

Penalty of `ScatteringEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_scattering.py
:language: python
:start-after: "#: <wst_penalty_matrix>"
:end-before: "#: </wst_penalty_matrix>"
```

Parsing of the `wst(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_scattering>"
:end-before: "#: </parse_scattering>"
```
