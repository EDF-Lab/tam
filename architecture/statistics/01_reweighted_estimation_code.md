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

## Negative binomial and Tweedie

`negative_binomial` and `tweedie` are `GLMFamily` subclasses in `_count_families.py`, registered in `build_strategy` (`StaticTAM(..., loss="negative_binomial", loss_kwargs={"dispersion": 0.5})` fixes the dispersion, `loss_kwargs={"power": 1.3}` sets the Tweedie power). The negative binomial variance reads the current `dispersion`, so the same IRLS loop serves any value of it:

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_count_families.py
:language: python
:start-after: "#: <negative_binomial>"
:end-before: "#: </negative_binomial>"
```

After the first fit, `BaseTAM.fit` asks a family that has `estimate_dispersion` for the dispersion at the fitted means and, if it moved by more than 0.1%, refits the means with the new variance (at most ten rounds). The Tweedie dispersion does not enter the weights, so its estimate returns without a refit. `validate_target` rejects a negative target before the fit. The estimate is exposed as `StaticTAM.dispersion_` (None for the other losses).

The Tweedie family computes the compound Poisson-gamma distribution function with the Poisson jump probabilities up to a tail of 1e-10, and its quantiles by bisection on that function (cost proportional to rows times jumps times 45 steps; a quantile below the mass at zero is exactly 0):

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_count_families.py
:language: python
:start-after: "#: <tweedie>"
:end-before: "#: </tweedie>"
```

`StaticTAM.predict_quantiles(data, taus)` on a plain model whose loss has a `quantiles` method (`poisson`, `negative_binomial`, `tweedie`) returns the `q<tau>` columns of `predict_count_quantiles`; on any other plain loss it raises as before.

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
