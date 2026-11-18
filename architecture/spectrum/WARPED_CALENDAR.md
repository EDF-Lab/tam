# Warped Calendar Harmonics: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See WARPED_CALENDAR.md](../../math/spectrum/WARPED_CALENDAR.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Phase-warped calendar harmonics by regime indicator are exposed via `wcal()`:

```python
# Diffeomorphic phase-warped calendar harmonics by regime indicator
formula = "Y ~ wcal(tod, by='is_holiday', period=48.0, n_harmonics=8)"
```

## Parameter Specification

### `WarpedCalendarEffect` (`wcal()`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `by` | `str` | `None` | Regime indicator column name. |
| `period` | `float` | `48.0` | Cycle period length. |
| `n_harmonics` | `int` | `6` | Number of Fourier harmonics $K$. |
| `gamma_warp`, `gamma_amp` | `float` | `1.0` | Penalty multipliers for phase-warping and amplitude blocks. |

## Complexity & Performance

- **Full-Rank Conditioning**: `WarpedCalendarEffect` modulates phase derivatives with $(\tau - \pi)/\pi$ to preserve full $6K$ column rank.

## Grouped models

When the model is grouped by the column that `wcal` reads (for example `group_col='tod'` with `wcal(tod, ...)`), that column is constant inside each group. The base harmonic columns are then constant, and the remaining columns vary only through `by`, so they add nothing that a day-type term on the same `by` column does not already span. Use `wcal` in a model that is not grouped by its input column. Measured on the national load benchmark: no group gains rank under `group_col='tod'`; ungrouped, `wcal` lowers the static test RMSE by 12.8 % over the same reference terms.

## State, Complexity and Limits

- **Class and file**: `WarpedCalendarEffect`, `src/tam/model/spectrum/_inertia.py`. Label: (BETA).
- **State**: none. **Layout**: `(B, T)` without regime, `(B, T, 2)` (phase, regime) with `by=` (or `others=`, or `regime_feature=`: the three spellings read the same column).
- **Time and memory**: $\mathcal{O}(B\,T\,K)$, pointwise.
- **Penalty coordinates**: `base`, `warp`, `amp` with a regime, `base` alone without.
- **Extrapolation**: `'periodic'` (default) wraps the phase around the period; the other modes of the router are available.
- **Limits**: see *Grouped models* above. The effect needs a column that varies inside a group, so it is meant for models that are not grouped by its own input column.
- **Collinearity guardrail**: when `group_col` is the effect's own phase column, the phase is constant inside a group and the columns are spanned by the per-group intercepts: the effect adds nothing (the rank argument of the note in the class docstring). In a model not grouped by that column, the derivative columns are multiplied by the phase envelope $g(\tau) = (\tau - \pi)/\pi \in [-1, 1)$ (with $\tau$ the angular phase), which keeps the base, warp and amplitude blocks of full column rank together: $6K$ columns with a regime, rank $6K$ (checked by `test_warped_calendar_effect`).
- **Smoothing criterion**: the effect has three penalty coordinates. Prefer `auto_fit(criterion="reml")` to the default coordinate GCV. Signature: `auto_fit(self, data_train, alpha_p_bounds=(-30.0, 6.0), number_of_steps=10, alpha_p_list=None, gamma=1.4, verbose=False, criterion="gcv")`. This is a recommendation, not a measured result for this effect: the repository's REML tests check the analytic gradient, the penalty basis of the multi-coordinate effects and that REML is more stable than GCV on autocorrelated noise (`test_reml_criterion`); no benchmark compares the two criteria on this effect.

## Code

Feature map of `WarpedCalendarEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <wcal_feature_map>"
:end-before: "#: </wcal_feature_map>"
```

Penalty of `WarpedCalendarEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_inertia.py
:language: python
:start-after: "#: <wcal_penalty_matrix>"
:end-before: "#: </wcal_penalty_matrix>"
```

Parsing of the `wcal(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_wcal>"
:end-before: "#: </parse_wcal>"
```
