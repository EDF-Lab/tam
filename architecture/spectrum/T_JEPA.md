# Temporal Joint-Embedding Predictive Architecture (T-JEPA): PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See T_JEPA.md](../../math/spectrum/T_JEPA.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

```python
formula = "Y ~ tjepa(temperature, others='temp_smooth|temp_max', latent_dim=4, window=16, readout='linear')"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `others` | `str` or `list` | `None` | Additional exogenous channels separated by `\|`. |
| `latent_dim` | `int` | `4` | Latent embedding dimension $d$. |
| `window` | `int` | `16` | Causal context window length $w$. |
| `horizon` | `int` | `4` | Self-supervised target prediction offset $\delta$. |
| `readout` | `str` | `'spline'` | Readout map: `'linear'` or `'spline'`. |
| `epochs` | `int` | `30` | Self-supervised VICReg pre-training epochs on exogenous inputs only. |

## Complexity & Performance

- **Zero Target Leakage**: Pre-trains the context encoder, EMA target encoder ($\tau = 0.99$), and predictor strictly on exogenous covariate streams without ever reading target $y$, caching training standardization statistics $(\hat{\mu}_x, \hat{\sigma}_x)$ and carrying trailing $w - 1$ samples across splits.

## State, Complexity and Limits

- **Class and file**: `TJEPAEffect`, `src/tam/model/spectrum/_tjepa.py`. Label: (BETA).
- **State**: `{"hist": (B, window - 1, D)}`. **Layout**: `(B, T, D)` (`keeps_feature_dim`); one column may also be given as `(B, T)`.
- **Modes**: `mode="svd"` (default) and `mode="vicreg"`.

  | `mode` | encoder | fit cost | deterministic | arguments read |
  |---|---|---|---|---|
  | `svd` | linear projection on the predictive modes of the context windows, then `tanh` (`_fit_tjepa_svd`) | two SVDs, $\mathcal{O}(n\,(wD)^2)$ with $n$ window pairs | yes: bit-identical for the same data | `latent_dim`, `window`, `horizon`; `seed` only when there are fewer than 16 window pairs |
  | `vicreg` | two-layer GELU encoder, moving-average target encoder and predictor trained with AdamW (`_fit_tjepa_encoder`) | $\mathcal{O}(\text{epochs}\cdot n\,wD)$ | for a seed, on one machine | those above and `epochs` (the optimizer settings are fixed) |

- **Runtime**: measured on the CPU of the development PC with $T = 2500$ rows in each of $B = 2$ groups, $d = 4$, $w = 16$ and 2 columns: `svd` 0.01 s, `vicreg` with 30 epochs 6 s. On the national load benchmark the fit of the `vicreg` effect took 88 s against 4 s for the spline reference. A feature map costs $\mathcal{O}(B\,T\,w\,D)$ in both modes.
- **Why `svd` stays reproducible inside P-WLS**: the encoder is a pure function of the training windows (no random initialization, no stochastic optimizer), so the design matrix, and therefore the penalized solution, is the same at every fit up to the rounding of the linear algebra; with `vicreg` it depends on the seed and on the order of the floating-point operations.
- **Errors**: a group shorter than `window + horizon` rows raises a `ValueError` at initialization.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction.
- **Reproducibility**: the seed fixes the training; the global random state is restored.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `TJEPAEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_tjepa.py
:language: python
:start-after: "#: <tjepa_feature_map>"
:end-before: "#: </tjepa_feature_map>"
```

Penalty of `TJEPAEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_tjepa.py
:language: python
:start-after: "#: <tjepa_penalty_matrix>"
:end-before: "#: </tjepa_penalty_matrix>"
```

Parsing of the `tjepa(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_tjepa>"
:end-before: "#: </parse_tjepa>"
```
