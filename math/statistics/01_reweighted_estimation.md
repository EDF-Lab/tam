# Reweighted Estimation: GLMs, Expectiles & Robust M-Estimators

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Reweighted estimation (architecture)](../../architecture/statistics/01_reweighted_estimation_code.md)

Every non-Gaussian scalar target is fitted by **Iteratively Reweighted Penalized Least Squares (IRLS)** {cite:p}`nelder1972generalized`: repeat the [P-WLS atom](../core/07_the_statistics_api.md) with a working response $z$ and weights $W$ that are refreshed from the current linear predictor $\eta=\Phi\theta$ until convergence.

## Exponential-family GLMs (P-IRLS)

For a link $g$ ($\eta=g(\mu)$), variance $V(\mu)$ and mean $\mu=g^{-1}(\eta)$, the Penalized Iteratively Reweighted Least Squares (P-IRLS) updates {cite:p}`nelder1972generalized,green1984iteratively,wood2017generalized` are:

$$ z = \eta + (y-\mu)\,g'(\mu), \qquad w = \frac{1}{g'(\mu)^2\,V(\mu)}. $$

Supported families: Gaussian (identity), Poisson, Gamma, negative binomial and Tweedie (log link), Binomial (logit). Step-halving on the **penalized** deviance guarantees monotone descent {cite:p}`wood2017generalized`.

## Overdispersed counts and zero-inflated positive targets

A count with variance above its mean, and a positive target with a share of exact zeros, need a variance that grows faster than the Poisson one, and a law that tells the probability of zero.

**Negative binomial** {cite:p}`lawless1987negative`: $V(\mu)=\mu+\alpha\mu^2$ with dispersion $\alpha=1/\theta$, the Poisson law with a Gamma-distributed rate. The mean is fitted by the reweighting above for a given $\alpha$; $\alpha$ is then the maximiser of the log-likelihood
$$ \sum_i \Big[\log\Gamma(y_i+\theta)-\log\Gamma(\theta)+\theta\log\frac{\theta}{\theta+\mu_i}+y_i\log\frac{\mu_i}{\theta+\mu_i}\Big] $$
at the fitted means, and the two steps alternate until $\alpha$ stops moving. The quantile of level $\tau$ is the smallest integer $k$ with $F(k)\ge\tau$, so it is an integer and never negative.

**Tweedie** with power $1<p<2$ {cite:p}`jorgensen1987exponential`: $V(\mu)=\phi\,\mu^{p}$, the law of a Poisson number $N$ of Gamma-distributed jumps,
$$ N\sim\text{Poisson}(\lambda),\quad \lambda=\frac{\mu^{2-p}}{\phi\,(2-p)}, \qquad \text{jump}\sim\text{Gamma}\Big(\tfrac{2-p}{p-1},\ \phi\,(p-1)\,\mu^{p-1}\Big), $$
which puts the mass $P(y=0)=e^{-\lambda}$ on zero and a continuous positive law elsewhere. The weights are $\mu^{2-p}$ and do not depend on $\phi$, so the mean is fitted once and $\phi$ is read from the Pearson residuals, $\hat\phi=\operatorname{mean}\,(y-\mu)^2/\mu^{p}$. The distribution function is $F(y)=e^{-\lambda}+\sum_{n\ge1}P(N=n)\,G_{na}(y/b)$ with $G$ the Gamma distribution function; the quantile of level $\tau$ is $0$ when $\tau\le e^{-\lambda}$ and otherwise the root of $F(y)=\tau$.

## Robust M-estimators

$z=y$; the loss enters through a bounded-influence weight {cite:p}`huber1964robust` against a MAD robust scale $s$:

$$ \text{Huber: } w=\min\!\left(1,\ \tfrac{\delta}{|u|/s}\right), \qquad \text{Student-t: } w=\frac{\nu+1}{\nu + (u/s)^2}. $$

## Expectiles (asymmetric least squares)

The $\tau$-expectile (asymmetric least squares {cite:p}`newey1987asymmetric`) weights residuals asymmetrically ($w=\tau$ above the fit, else $1-\tau$); the Jones/Yao-Tong bijection {cite:p}`jones1994expectiles,yao1996asymmetric` maps an expectile level back to a genuine quantile.

$$ w = \tau\,\mathbb 1\{y\ge\eta\} + (1-\tau)\,\mathbb 1\{y<\eta\}. $$
