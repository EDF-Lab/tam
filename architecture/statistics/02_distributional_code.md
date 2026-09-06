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

## The `additive.py` bridge (one-line delegates)

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:pyobject: StaticTAM.predict_quantiles
```
