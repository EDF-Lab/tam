# Mixtures & Copulas: EM over the Atom, and the Standalone Joiner

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [Mixtures & copulas](../../math/statistics/03_mixture_copula.md)

> Implementation only, for the EM equations and the copula construction see the [math theory](../../math/statistics/03_mixture_copula.md).

## The EM loop (M-step is the weighted atom)

The inner loop drives any model exposing `_solve_pwls_step`, so a mixture is just `StaticTAM` on a different schedule, no standalone class. It lives in `statistics/estimation/_mixture.py`.

```{literalinclude} ../../../../src/tam/model/statistics/estimation/_mixture.py
:language: python
:pyobject: _single_em_run
```

## The Gaussian copula (a standalone joiner)

The copula is the one distributional object *not* folded into `StaticTAM`: it binds several fitted margins, duck-typed on each margin's `.cdf()` / `.fit()`, and never imports `StaticTAM`.

```{literalinclude} ../../../../src/tam/model/statistics/distributions/copula.py
:language: python
:pyobject: GaussianCopulaTAM
```
