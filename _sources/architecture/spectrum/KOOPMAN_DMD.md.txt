# Koopman Kernel DMD: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See KOOPMAN_DMD.md](../../math/spectrum/KOOPMAN_DMD.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

```python
formula = "Y ~ koopman(temperature, k=18, delay_embed=8, n_lift=64)"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Input dynamical observable column. |
| `k` | `int` | `12` | Number of Koopman eigenmodes retained. |
| `n_lift` | `int` | `64` | Random Fourier dictionary dimension for EDMD lifting. |
| `delay_embed` | `int` | `1` | Hankel time-delay embedding depth. |
| `alpha_res` | `float` | `10.0` | Spectral residual penalty weight. |
| `alpha_unit` | `float` | `10.0` | Unit-circle eigenvalue deviation penalty weight. |

## Complexity & Performance

- **Closed-Form EDMD**: Estimates the finite-dimensional Koopman generator $K_{\text{EDMD}}$ via Ridge regression on lifted delay coordinates and carries the trailing delay rows of every group across splits via `state_in` / `_state_out`.

## State, Complexity and Limits

- **Class and file**: `KoopmanEffect`, `src/tam/model/spectrum/_koopman.py`. Label: (BETA).
- **State**: `{"hist": (B, delay_embed - 1, d)}` when `delay_embed > 1` (`stateful`), none otherwise. **Layout**: `(B, T, d)`; `others=` adds columns to the delay embedding.
- **Columns**: exactly `k`, always. `get_n_coeffs()` does not depend on the data.
- **Time and memory**: the fit is one SVD of at most 20 000 snapshot rows; a feature map costs $\mathcal{O}(B\,T\,(d\,d_H)^2)$ for the quadratic lift.
- **Device**: the dictionary is drawn on the CPU in float64 and moved to the device of the input.
- **Extrapolation**: with a delay embedding `'linear'` and `'saturation'` are refused at construction.
- **Limits**: very short series (fewer than three steps per group) fall back to the first `k` dictionary directions.
- **Penalty terms**: the weight of every dictionary column is $1 + \alpha_{\text{res}}\, r^2 + \alpha_{\text{unit}}\,(|\mu|^2 - 1)^2$, where $r^2$ is the ResDMD residual of the mode (how badly its eigenpair satisfies the transition equation on the training snapshots) and $|\mu|$ the modulus of its eigenvalue; the defaults are `alpha_res = alpha_unit = 10`. Both terms add to the unit baseline, so a mode that is exact and on the unit circle has weight 1 and a poor mode is shrunk by up to a factor that grows linearly in `alpha_res` and quadratically in the distance of $|\mu|^2$ from 1. Pairs of complex-conjugate modes share their weight. The two constants are fixed by the user: the smoothing criterion (GCV or REML) tunes the single scale $\lambda$ that multiplies the whole penalty, not the two constants.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `KoopmanEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_koopman.py
:language: python
:start-after: "#: <koopman_feature_map>"
:end-before: "#: </koopman_feature_map>"
```

Penalty of `KoopmanEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_koopman.py
:language: python
:start-after: "#: <koopman_penalty_matrix>"
:end-before: "#: </koopman_penalty_matrix>"
```

Parsing of the `koopman(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_koopman>"
:end-before: "#: </parse_koopman>"
```
