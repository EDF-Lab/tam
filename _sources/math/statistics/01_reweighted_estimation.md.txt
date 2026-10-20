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

## Choosing the smoothing parameters of a GLM

Generalized cross-validation for a Gaussian model scores $\lVert y-\hat y\rVert^2$ and the trace of the hat matrix. For a GLM the same score is taken on the **working system** of the last reweighting pass: with the working response $z$ and the Fisher weights $W$ at the fitted $\eta$, the penalized fit is a weighted least-squares problem in $(\sqrt W\,\Phi,\ \sqrt W\,z)$, whose hat matrix is $A_\lambda=\sqrt W\,\Phi\,(\Phi^\top W\Phi+nS_\lambda)^{-1}\Phi^\top\sqrt W$. The criterion is

$$ \mathrm{GCV}(\lambda)=\frac{n\,\lVert\sqrt W(z-\Phi\hat\theta_\lambda)\rVert^2}{\bigl(n-\gamma\,\operatorname{tr}A_\lambda\bigr)^2}, $$

which is the Pearson-type criterion of the performance iteration {cite:p}`wood2017generalized`. Because $z$ and $W$ depend on $\lambda$ through $\eta$, one pass chooses $\lambda$ for the current working system, the model is refitted at that $\lambda$, and the two steps are repeated until $\log_{10}\lambda$ moves by less than a fraction of a decade. Scoring the raw response with an identity link instead would ignore the variance function $V(\mu)$ and choose a $\lambda$ for a model that is not the one fitted.

## Exposure offsets

Counts observed over a variable effort $E_i$ (a filtered volume, a surveyed area, a population at risk) have a mean proportional to the effort, $\mathbb E[Y_i]=E_i\,e^{\eta_i}=e^{\log E_i+\eta_i}$. The term $o_i=\log E_i$ is a **known** part of the linear predictor, $\eta_i^{\text{total}}=o_i+\Phi_i\theta$, with coefficient fixed at 1 and no parameter. It cannot be replaced by a feature $\log E$ with a free coefficient: that coefficient is shrunk by the penalty and, when $\log E$ is correlated with the other features, it absorbs part of their signal and moves the predictions where the effort differs.

The fit changes in one place only. The working response of a pass is shifted by the offset, $z_i-o_i=(\eta_i-o_i)+(y_i-\mu_i)\,g'(\mu_i)$, and the same penalized weighted least-squares system is solved for $\theta$ with $\mu_i=g^{-1}(o_i+\Phi_i\theta)$; for the Gaussian identity case this is the fit of $y-o$. The offset takes the raw column, with no rescaling, because a rescaled column would no longer have coefficient 1.

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
