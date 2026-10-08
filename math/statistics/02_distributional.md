# Distributional (Location-Scale) Models

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Distributional (architecture)](../../architecture/statistics/02_distributional_code.md)

A `{ "mu": ..., "sigma": ... }` formula fits a 2-parameter location-scale distribution (a GAMLSS {cite:p}`rigby2005generalized`) as a schedule that **composes two sub-atoms**, no new solver.

## The two-stage schedule

1. **Location.** Fit $\mu(x)$ with an L2 atom on the (log-)target $t=\log y$. The log scale requires $y>0$: $\log y$ is undefined at and
   below zero, so a target taking such values is modelled on its own scale, $t=y$.
2. **Scale.** Read the squared residuals $r^2=(t-\hat\mu)^2$ and fit $\sigma(x)$ as a **Gamma-GLM** atom on $r^2$ (or an L2 atom on $\log r^2$ with a $\log\chi^2_1$ bias correction).

## The tail law and quantiles

The standardized residual law $F$ (Normal or Student-t) is selected from the residual excess kurtosis. Conditional quantiles are then, by construction non-crossing:

$$ Q_\tau(x) = \exp\!\Big(\hat\mu(x) + \hat\sigma(x)\,F^{-1}(\tau)\Big). $$

## Monotone target transforms

The location-scale law can sit on a transformed target $t=g(y)$ instead of the logarithm, for any strictly increasing $g$. Quantiles of $y$ follow exactly, since a monotone map preserves order:

$$ Q_\tau^{(y)}(x) = g^{-1}\Big(\hat\mu(x) + \hat\sigma(x)\,F^{-1}(\tau)\Big). $$

Two maps cover the targets that the logarithm cannot take.

- **Real-valued, skewed, heavy-tailed targets with negative values** (electricity prices, spikes and returns): the area hyperbolic sine $g(y)=\operatorname{arsinh}\big((y-c)/s\big)$, with the centre $c$ the median and the scale $s$ a robust spread of the training target. It behaves like the identity near $c$ and like a logarithm in both tails, and it is the standard variance-stabilizing choice for electricity prices {cite:p}`uniejewski2018variance`. With a Normal law on $t$, $y$ follows the Johnson $S_U$ family {cite:p}`johnson1949systems`: skewed, with heavier tails than a Normal.
- **Targets bounded in an interval $(a,b)$** (shares, humidity, capacity-capped power): the logit $g(y)=\log\big(u/(1-u)\big)$ with $u=(y-a)/(b-a)$. Every quantile lies strictly inside $(a,b)$, which a Normal law on the response scale cannot guarantee. A value exactly on a bound has an infinite logit and is moved inside by a tiny fraction of the interval.

An optional convex `scale_shrinkage` $\lambda\in[0,1]$ blends the conditional variance toward the global one. The probability integral transform $F\big((t-\hat\mu)/\hat\sigma\big)$ gives the CDF, a two-sided anomaly score, and the CRPS (closed form for the Normal, else a Gauss-Legendre quantile integral).
