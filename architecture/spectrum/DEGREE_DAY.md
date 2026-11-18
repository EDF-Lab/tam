# Smooth Degree-Day Hinges: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See DEGREE_DAY.md](../../math/spectrum/DEGREE_DAY.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Smooth softplus heating and cooling degree-day hinges are exposed via `hdd_cdd()` (aliases `dd()` and `smooth_hinge()`):

```python
# C-infinity softplus heating and cooling degree-day hinges
formula = "Y ~ hdd_cdd(temperature, sharpness=5.0, quad=True)"
```

## Parameter Specification

### `DegreeDayEffect` (`hdd_cdd()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `heat_thresholds` | `Sequence[float]` or `str` | `(-0.2, 0.0)` | Heating reference thresholds. |
| `cool_thresholds` | `Sequence[float]` or `str` | `(0.3, 0.5)` | Cooling reference thresholds. |
| `sharpness` | `float` | `10.0` | Softplus sharpness $\beta > 0$. |
| `quad` | `bool` | `True` | Append quadratic saturation terms. |

## Complexity & Performance

- Pointwise in the temperature: $\mathcal{O}(B\,T\,(n_{\text{heat}} + n_{\text{cool}}))$ time and memory.

## State, Complexity and Limits

- **Class and file**: `DegreeDayEffect` (alias `SmoothHingeEffect`), `src/tam/model/spectrum/_inertia.py`. Label: (BETA).
- **State**: none. **Layout**: any shape, one row of features per input value.
- **Thresholds**: kept in ascending order inside each family (the penalty couples neighbours); the columns follow that order.
- **Penalty**: squared hinges weigh 4, a first-difference coupling acts inside the heating family and inside the cooling family; one threshold per side gives a diagonal matrix. One penalty coordinate.
- **Extrapolation**: all modes of the router are available.

## Code

Feature map of `DegreeDayEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <dd_feature_map>"
:end-before: "#: </dd_feature_map>"
```

Penalty of `DegreeDayEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <dd_penalty_matrix>"
:end-before: "#: </dd_penalty_matrix>"
```

Parsing of the `dd(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_hdd_cdd>"
:end-before: "#: </parse_hdd_cdd>"
```
