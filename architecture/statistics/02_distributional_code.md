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

## The `additive.py` bridge (one-line delegates)

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:pyobject: StaticTAM.predict_quantiles
```
