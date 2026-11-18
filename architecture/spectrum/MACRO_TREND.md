# Macro-Trend (`hp_trend`): PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See MACRO_TREND.md](../../math/spectrum/MACRO_TREND.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

```python
# Natural cubic Hodrick-Prescott macro-trend with linear extrapolation
formula = "Y ~ hp_trend(trend, k=20, ap=-6)"
```

## Parameter Specification

### `MacroTrendEffect` (`hp_trend()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `k` | `int` | `20` | Number of natural cubic B-spline knots. |
| `lambda_p` | `float` | `1600.0` | HP second-difference smoothing weight. |
| `extrapolate` | `str` | `'linear'` | Extrapolation mode. |

The class lives in `src/tam/model/spectrum/_inertia.py` with the other load effects.

## State, Complexity and Limits

- **Class and file**: `MacroTrendEffect`, `src/tam/model/spectrum/_inertia.py`; `_load_forecasting.py` re-exports it under the old module name. Label: (BETA).
- **State**: none. **Layout**: any shape, one row of features per input value. **Time and memory**: $\mathcal{O}(B\,T\,k)$.
- **Penalty**: $\lambda\,D_2^\top D_2$ with the default $\lambda = 1600$ of the Hodrick–Prescott filter; `ap` in a formula replaces it.
- **Extrapolation**: only `'linear'` is implemented for `hp_trend`: `build_feature_map` always continues the basis by its tangent beyond the training range, whatever value `extrapolate` has; the argument is kept for API symmetry.
- **Horizon**: the linear continuation carries the slope of the trend at the end of the training period over the whole horizon, so over a long static horizon the extrapolated level can drift far from the actual one. On the national load benchmark (a static fit on the years before the test year) the test RMSE of the model with this term was about four times that of the spline reference (+320 %), while its adaptive variant, which refits and extrapolates one step at a time, was -1.38 % on validation and +0.39 % on test. Use the effect in `AdaptiveTAM`, or refit often, rather than as a static term over a long horizon.

## Code

Feature map of `MacroTrendEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <hp_trend_feature_map>"
:end-before: "#: </hp_trend_feature_map>"
```

Penalty of `MacroTrendEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <hp_trend_penalty_matrix>"
:end-before: "#: </hp_trend_penalty_matrix>"
```

Parsing of the `hp_trend(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_hp_trend>"
:end-before: "#: </parse_hp_trend>"
```
