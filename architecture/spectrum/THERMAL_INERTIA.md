# Thermal Inertia: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See THERMAL_INERTIA.md](../../math/spectrum/THERMAL_INERTIA.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Multi-timescale physical thermal inertia is exposed via `inertia()` and `thermal()`:

```python
# Multi-timescale thermal inertia (exponentially smoothed B-splines)
formula = "Y ~ inertia(temperature, type='thermal', alphas='0.05|0.01|0.002', k=4)"
```

## Parameter Specification

### `InertiaEffect` (`inertia()` / `thermal()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Input feature column name. |
| `type` | `str` | `'thermal'` | Inertia regime (`'thermal'`, `'rotational'`, `'behavioral'`). |
| `alphas` | `Sequence[float]` or `str` | `(0.05, 0.01, 0.002)` | Exponential smoothing decay constants $\alpha_m \in (0, 1]$. |
| `k` | `int` | `8` | Spline knots per decay scale. |
| `deg` | `int` | `3` | B-spline polynomial degree. |
| `p` | `int` | `2` | Difference penalty order. |

## Complexity & Performance

## State, Complexity and Limits

- **Class and file**: `InertiaEffect` (alias `ThermalInertiaEffect`), `src/tam/model/spectrum/_inertia.py`. Label: (BETA).
- **State**: `{"y": (B, M)}`, the last smoothed value of every group and decay constant.
- **Layout**: `(B, T)` in, `(B, T, M (k + degree))` out.
- **Time and memory**: $\mathcal{O}(B\,T\,M)$ for the recurrences (loop below 64 steps, FFT convolution above) plus the spline evaluation; no scipy filter, no flattening of groups.
- **Penalty coordinates**: `M`, labelled `alpha=<value>`, all starting at $\lambda$; GCV and REML tune them separately.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction (the effect is temporal); the spline continues linearly outside the normalized range of the smoothed series.
- **Limits**: the bounds of the input are frozen on the training period, so a temperature outside it reaches the linear continuation of the splines.
- **Smoothing criterion**: the effect has several penalty coordinates (see above). Prefer `auto_fit(criterion="reml")` to the default coordinate GCV. Signature: `auto_fit(self, data_train, alpha_p_bounds=(-30.0, 6.0), number_of_steps=10, alpha_p_list=None, gamma=1.4, verbose=False, criterion="gcv")`. This is a recommendation, not a measured result for this effect: the repository's REML tests check the analytic gradient, the penalty basis of the multi-coordinate effects and that REML is more stable than GCV on autocorrelated noise (`test_reml_criterion`); no benchmark compares the two criteria on this effect.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `InertiaEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <inertia_feature_map>"
:end-before: "#: </inertia_feature_map>"
```

Penalty of `InertiaEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <inertia_penalty_matrix>"
:end-before: "#: </inertia_penalty_matrix>"
```

Parsing of the `inertia(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_inertia>"
:end-before: "#: </parse_inertia>"
```
