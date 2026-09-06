# Reweighted Estimation: GLMs, Expectiles & Robust M-Estimators

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Reweighted estimation (architecture)](../../architecture/statistics/01_reweighted_estimation_code.md)

Every non-Gaussian scalar target is fitted by **Iteratively Reweighted Penalized Least Squares (IRLS)** {cite:p}`nelder1972generalized`: repeat the [P-WLS atom](../core/07_the_statistics_api.md) with a working response $z$ and weights $W$ that are refreshed from the current linear predictor $\eta=\Phi\theta$ until convergence.

## Exponential-family GLMs (P-IRLS)

For a link $g$ ($\eta=g(\mu)$), variance $V(\mu)$ and mean $\mu=g^{-1}(\eta)$, the Penalized Iteratively Reweighted Least Squares (P-IRLS) updates {cite:p}`nelder1972generalized,green1984iteratively,wood2017generalized` are:

$$ z = \eta + (y-\mu)\,g'(\mu), \qquad w = \frac{1}{g'(\mu)^2\,V(\mu)}. $$

Supported families: Gaussian (identity), Poisson & Gamma (log link), Binomial (logit). Step-halving on the **penalized** deviance guarantees monotone descent {cite:p}`wood2017generalized`.

## Robust M-estimators

$z=y$; the loss enters through a bounded-influence weight {cite:p}`huber1964robust` against a MAD robust scale $s$:

$$ \text{Huber: } w=\min\!\left(1,\ \tfrac{\delta}{|u|/s}\right), \qquad \text{Student-t: } w=\frac{\nu+1}{\nu + (u/s)^2}. $$

## Expectiles (asymmetric least squares)

The $\tau$-expectile (asymmetric least squares {cite:p}`newey1987asymmetric`) weights residuals asymmetrically ($w=\tau$ above the fit, else $1-\tau$); the Jones/Yao-Tong bijection {cite:p}`jones1994expectiles,yao1996asymmetric` maps an expectile level back to a genuine quantile.

$$ w = \tau\,\mathbb 1\{y\ge\eta\} + (1-\tau)\,\mathbb 1\{y<\eta\}. $$
