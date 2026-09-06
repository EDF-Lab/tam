# Reweighted Estimation: Strategy Pattern & IRLS Driver

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [Reweighted estimation](../../math/statistics/01_reweighted_estimation.md)

> Implementation only, for the working-response/weight formulas see the [math theory](../../math/statistics/01_reweighted_estimation.md).

The `estimation/` package mirrors `spectrum/`: a base contract, one file per loss family, and a registry.

## The GLM families

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_glm.py
:language: python
:start-after: "#: <glm_family>"
:end-before: "#: </glm_family>"
```

## Robust M-estimators and expectiles

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_robust.py
:language: python
:pyobject: HuberLoss
```

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_expectile.py
:language: python
:pyobject: ExpectileLoss
```

## The IRLS schedule

`reweighted_penalized_fit` normalises the weights to unit mean (preserving the `nS` penalty scale), solves one atom, and, for GLMs, step-halves on the penalized objective. For the Gaussian identity it converges in a single step.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_reweighting.py
:language: python
:start-after: "#: <reweighting_loop>"
:end-before: "#: </reweighting_loop>"
```
