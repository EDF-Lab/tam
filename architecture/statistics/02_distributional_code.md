# Distributional (Location-Scale): Thin Delegation

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [Distributional](../../math/statistics/02_distributional.md)

> Implementation only, for the two-stage schedule and quantile formulas see the [math theory](../../math/statistics/02_distributional.md).

There is **no `DistributionalTAM` class**: a dict formula makes `StaticTAM` a thin frontend that delegates the whole schedule to `statistics/estimation/_distributional.py`. Sub-models are built with `type(model)(...)`, so the module never imports `StaticTAM` (no circular import).

## The schedule (location, then scale on the residuals)

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: fit
```

## Weighted refit of the location and debiased scale

`dist_kwargs` takes three more options, defaults of which reproduce the fit above bit for bit: `refit_rounds` (0), `refit_tol` (1e-4) and `scale_correction` (`"none"`, `"leverage"`, `"edf"`).

- **Weights.** A plain `StaticTAM` has a private attribute `_weight_col_`; when it names a column of the training frame, `BaseTAM.fit` stacks that column exactly like the target (`_stacked_column`: same balancing, groups and order), checks it (finite, non-negative, not all zero) and passes it to the penalized least-squares atom (`_solve_prepared(..., weights)`). `predict` ignores it.
- **Loop.** After the first location and scale fits, each round computes the unshrunk conditional variance of the scale model, turns its inverse into weights (normalised to unit mean so that the `n P` penalty keeps its scale, then kept within a factor 1e3 of the median: `_inverse_variance_weights`), refits the same location sub-model with them (its penalties and knots are kept: the smoothing chosen by `select` on the first pass is not re-chosen), and refits the scale on the new residuals. It stops when the relative change of the fitted location is below `refit_tol`; if the cap `refit_rounds` is reached first, a `UserWarning` gives the last change and the last round is kept.
- **Leverage.** `StaticTAM.leverage(data)` returns `h_ii` for every row (one batched solve per group, with the same penalty and ridge floor as the fit and the weights of the fit). `scale_correction="leverage"` divides the squared residuals by `1 - min(h, 0.9)`, `"edf"` multiplies them per group by `n / (n - sum h)` (`_corrected_squared_residual`); `_global_scale_variance_` is the mean of the corrected values. It works with `refit_rounds=0`. A warning is given once when more than 1% of the leverages hit the clamp.
- **Record.** `refit_history_` (one dict per round: `round`, `change`, `log_likelihood`, `clipped_weights`, `clamped_leverages`), `n_refits_`, `converged_` and `log_likelihood_`, the log-likelihood of the training rows at the final location, scale and tail law.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: _corrected_squared_residual
```

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: _inverse_variance_weights
```

## The tail law and the location/scale predictions

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: select_tail_family
```

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: mu_sigma
```

For a log target the quantile and the median are `exp` of a model-scale value. A value beyond the float64 range (a location or scale
prediction far outside the training range) is returned as `inf` with an explicit `UserWarning` naming the quantity and the first rows,
never numpy's silent overflow warning:

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: _to_response_scale
```

The log target is the default (`dist_kwargs={"log_target": True}`). `fit` first checks the training target: a value `<= 0` has no
logarithm, and clipping it to a tiny constant would turn it into a huge negative outlier, so the fit raises a `ValueError` naming the
count and the option `log_target=False` instead. The mixture mode (`_mixture.py`) runs the same check. Only the training target is
checked: the CDF of a new observation `<= 0` under a log-scale law is ~0, a valid answer.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: require_positive_target
```

**Target transform.** `dist_kwargs={"target_transform": ...}` takes `"none"`, `"log"` (the default, same as `log_target=True`), `"log1p"`, `"asinh"`, `"logit"` or `"boxcox"`; giving it together with a contradicting `log_target` raises a `ValueError`, and a mixture accepts `"none"` and `"log"` only. The maps themselves (`estimate_params`, `forward`, `inverse`, `inverse_mean`) live in `statistics/estimation/_target_transform.py`, one code path for the distributional and the plain model. `resolve_target_transform` reads the options once, in the constructor:

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: resolve_target_transform
```

`fit` estimates the parameters of the transform on the training target and stores them in `_transform_params_`: for `"asinh"` the median and a robust scale (normalised MAD, else the standard deviation); for `"logit"` the bounds, from `dist_kwargs={"bounds": (low, high)}` or the observed range widened by 5% on each side, and a training value outside given bounds raises a `ValueError`.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: fit_target_transform
```

The forward map is applied to the target everywhere it is read (fit, CDF, CRPS, anomaly score), and `from_model_scale` is the single back-transform used by `predict_median`, `predict_quantile` and `anomaly_score`, and by the studentized conformal intervals through the model method `_from_model_scale`:

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_distributional.py
:language: python
:pyobject: from_model_scale
```

## The `additive.py` bridge (one-line delegates)

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:pyobject: StaticTAM.predict_quantiles
```

## Target transform of a plain model

`StaticTAM(formula, target_transform="log", transform_kwargs={...})` applies the same maps without a scale law. `fit`, `auto_fit` and `grid_search_fit` fit the model on a copy of the frame whose target is $g(y)$ (`_transformed_frame`; the parameters of $g$ are estimated on the training frame only), then store the model-scale residual variance and a 20 001-point quantile function of the residuals (`_finish_transform_fit`). `predict` returns $g^{-1}(\hat t)$ (or the second-order mean with `transform_kwargs={"mean_correction": True}`), and `predict_quantiles` returns $g^{-1}(\hat t+\hat F_e^{-1}(\tau))$ as columns `q<tau>`. A GLM loss refuses a transform (its link is one already), as do a dict formula and a mixture, which take `dist_kwargs`. Lag columns that hold past values of the target are built by the caller and are not transformed. `decompose_prediction` stays on the model scale, `calibrate_conformal` works on the response scale.
