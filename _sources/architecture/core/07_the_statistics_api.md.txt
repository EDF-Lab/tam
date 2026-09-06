# The Statistics API: One Atom, Many Schedules

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [One Atom, Many Statistics](../../math/core/07_the_statistics_api.md)

> This page shows only the *implementation* wiring. For the equations (working response, weights, links, quantiles) see the [math theory](../../math/core/07_the_statistics_api.md).

**Core interface:** `_base.py::BaseTAM._solve_pwls_step`  ·  **Schedule layer:** `statistics/estimation/`

## The Atom and its bit-identical default

Per-observation weights enter through a numerically-stable `sqrt(W)` row-scaling of both `Phi` and `Y`; `sample_weights=None` never multiplies, so the default `loss="l2"` path is bit-identical to the original least-squares solver.

```{literalinclude} ../../../../src/tam/model/_math.py
:language: python
:start-after: "#: <weighted_cov>"
:end-before: "#: </weighted_cov>"
```

Every schedule reaches the solver through this single method, the atom (`weights=None` is the ordinary penalized solve; the schedules call it repeatedly with updated `(z, W)`).

```{literalinclude} ../../../../src/tam/model/_base.py
:language: python
:start-after: "#: <pwls_atom>"
:end-before: "#: </pwls_atom>"
```

## The router

`StaticTAM.fit` routes by its inputs: a string formula to standard IRLS, a `{param: formula}` dict to the location-scale schedule, and `mixture_components=K` to the EM schedule. No wrapper classes.

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:pyobject: StaticTAM.fit
```

## The strategy contract and factory

A `ReweightingStrategy` is the statistical analogue of a `BaseEffect`: a small, swappable object that reshapes `(z, W)`. `build_strategy` maps a user-facing `loss` name to a concrete strategy.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_base_strategy.py
:language: python
:start-after: "#: <strategy_protocol>"
:end-before: "#: </strategy_protocol>"
```

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_factory.py
:language: python
:start-after: "#: <build_strategy>"
:end-before: "#: </build_strategy>"
```
