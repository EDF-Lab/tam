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

## Efficient location and unbiased scale

The two-stage schedule is exact for the scale given the location, but two things are left on the table.

**Efficiency of the location.** Given the scale, the minimum-variance estimator of $\theta$ in $\mu=\Phi\theta$ is the weighted penalized least-squares solution with weights $w_i=1/\sigma_i^2$,

$$ \hat\theta=\big(\Phi^\top W\Phi+nP\big)^{-1}\Phi^\top W t, $$

while the first stage uses $W=I$: it is unbiased, but its variance exceeds that of the weighted one when the noise is heteroscedastic. For a symmetric law the information matrix is block-diagonal in $(\mu,\sigma)$ (the parameters are orthogonal), so alternating the two steps, location with the weights of the last scale and scale on the residuals of the last location, converges to the joint maximum of the likelihood, usually in one to three rounds, and loses nothing against a joint Newton step. The gain depends on how local the location is: the variance of a flexible smoother at a point is governed by the noise at that point only, so the weights change it little; a parsimonious location (a line, a few knots) over a noise whose spread varies several times gains a third of its error or more.

**Bias of the scale.** The scale is fitted to squared residuals, which the fit has shrunk: with the hat matrix $H$ of the location, $\mathbb E[r_i^2]=\sigma_i^2(1-2h_{ii})+\sum_jH_{ij}^2\sigma_j^2\approx\sigma_i^2(1-h_{ii})$ when $\sigma$ varies slowly compared to the width of the smoother, since $\sum_jH_{ij}^2=h_{ii}$. The variance is therefore underestimated by the fraction $h_{ii}$, which is large when the location is flexible and the sample small, and the intervals are too narrow. Dividing by the leverage gives an unbiased target,

$$ \tilde r_i^2=\frac{r_i^2}{1-h_{ii}},\qquad h_{ii}=w_i\,\phi_i^\top\big(\Phi^\top W\Phi+nP\big)^{-1}\phi_i, $$

the per-row form of the degrees-of-freedom correction of restricted maximum likelihood {cite:p}`patterson1971recovery`, whose aggregate version replaces $n$ by $n-\operatorname{edf}$ with $\operatorname{edf}=\sum_ih_{ii}$ the trace of the hat matrix {cite:p}`wood2017generalized`. The leverage is clamped below $0.9$ because a row that the location nearly interpolates would make $\tilde r_i^2$ explode.

## The tail law and quantiles

The standardized residual law $F$ (Normal or Student-t) is selected from the residual excess kurtosis. Conditional quantiles are then, by construction non-crossing:

$$ Q_\tau(x) = \exp\!\Big(\hat\mu(x) + \hat\sigma(x)\,F^{-1}(\tau)\Big). $$

## Monotone target transforms

The location-scale law can sit on a transformed target $t=g(y)$ instead of the logarithm, for any strictly increasing $g$. Quantiles of $y$ follow exactly, since a monotone map preserves order:

$$ Q_\tau^{(y)}(x) = g^{-1}\Big(\hat\mu(x) + \hat\sigma(x)\,F^{-1}(\tau)\Big). $$

Two maps cover the targets that the logarithm cannot take. The shifted logarithm $\log(y+c)$, $\log(1+y)$ and the Box-Cox map $g(y)=\big((y+c)^{\lambda}-1\big)/\lambda$ ($\log(y+c)$ at $\lambda=0$, $\lambda$ by maximum likelihood unless given) {cite:p}`box1964analysis` are available for positive targets whose spread grows less than proportionally to the level.

- **Real-valued, skewed, heavy-tailed targets with negative values** (electricity prices, spikes and returns): the area hyperbolic sine $g(y)=\operatorname{arsinh}\big((y-c)/s\big)$, with the centre $c$ the median and the scale $s$ a robust spread of the training target. It behaves like the identity near $c$ and like a logarithm in both tails, and it is the standard variance-stabilizing choice for electricity prices {cite:p}`uniejewski2018variance`. With a Normal law on $t$, $y$ follows the Johnson $S_U$ family {cite:p}`johnson1949systems`: skewed, with heavier tails than a Normal.
- **Targets bounded in an interval $(a,b)$** (shares, humidity, capacity-capped power): the logit $g(y)=\log\big(u/(1-u)\big)$ with $u=(y-a)/(b-a)$. Every quantile lies strictly inside $(a,b)$, which a Normal law on the response scale cannot guarantee. A value exactly on a bound has an infinite logit and is moved inside by a tiny fraction of the interval.

### The same maps on a plain model

A model without a scale law can sit on $t=g(y)$ too. The fit is the ordinary penalized fit of $t$, and the quantiles are those of the empirical law of its residuals $e_i=t_i-\hat t_i$, shifted to the fitted value and mapped back:

$$ \hat Q_\tau^{(y)}(x) = g^{-1}\big(\hat t(x) + \hat F_e^{-1}(\tau)\big), $$

which is exact for the quantiles of $y$ whatever the monotone $g$, with a residual spread that does not depend on $x$ (a location-scale model with a scale law relaxes this). The point forecast $g^{-1}(\hat t)$ is a median-type value. A mean forecast needs the second-order correction of Jensen's inequality, with $v$ the variance of the residuals:

$$ \mathbb E[y\mid x]\approx g^{-1}(\hat t)+\tfrac12\,(g^{-1})''(\hat t)\,v, $$

which is $\exp(\hat t+v/2)$ for the logarithm, where it is the mean of the log-normal law.

An optional convex `scale_shrinkage` $\lambda\in[0,1]$ blends the conditional variance toward the global one. The probability integral transform $F\big((t-\hat\mu)/\hat\sigma\big)$ gives the CDF, a two-sided anomaly score, and the CRPS (closed form for the Normal, else a Gauss-Legendre quantile integral).
