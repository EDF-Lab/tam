# Distributional (Location-Scale) Models

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Distributional (architecture)](../../architecture/statistics/02_distributional_code.md)

A `{ "mu": ..., "sigma": ... }` formula fits a 2-parameter location-scale distribution (a GAMLSS {cite:p}`rigby2005generalized`) as a schedule that **composes two sub-atoms**, no new solver.

## The two-stage schedule

1. **Location.** Fit $\mu(x)$ with an L2 atom on the (log-)target $t=\log y$.
2. **Scale.** Read the squared residuals $r^2=(t-\hat\mu)^2$ and fit $\sigma(x)$ as a **Gamma-GLM** atom on $r^2$ (or an L2 atom on $\log r^2$ with a $\log\chi^2_1$ bias correction).

## The tail law and quantiles

The standardized residual law $F$ (Normal or Student-t) is selected from the residual excess kurtosis. Conditional quantiles are then, by construction non-crossing:

$$ Q_\tau(x) = \exp\!\Big(\hat\mu(x) + \hat\sigma(x)\,F^{-1}(\tau)\Big). $$

An optional convex `scale_shrinkage` $\lambda\in[0,1]$ blends the conditional variance toward the global one. The probability integral transform $F\big((t-\hat\mu)/\hat\sigma\big)$ gives the CDF, a two-sided anomaly score, and the CRPS (closed form for the Normal, else a Gauss-Legendre quantile integral).
