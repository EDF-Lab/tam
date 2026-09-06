# Mixtures & Gaussian Copulas

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Mixtures & copulas (architecture)](../../architecture/statistics/03_mixture_copula_code.md)

## Finite Gaussian mixture (EM)

A $K$-component mixture of TAM regressions is fitted by Expectation-Maximization {cite:p}`dempster1977maximum` whose **M-step is exactly the weighted atom** with the posterior responsibilities $r_{ik}$ as per-observation weights:

$$
\text{E: } r_{ik} = \frac{\pi_k\,\mathcal N(t_i;\,\Phi_i\theta_k,\,\sigma_k)}{\sum_j \pi_j\,\mathcal N(t_i;\,\Phi_i\theta_j,\,\sigma_j)}, \qquad
\text{M: } \theta_k = \text{atom}(x,\,t;\,W=r_k),\ \ \sigma_k,\pi_k \text{ from } r_k.
$$

No standalone class: a mixture is `StaticTAM` on a different schedule (single-group).

## Gaussian copula

Sklar's theorem {cite:p}`sklar1959fonctions` separates several correlated responses into independent **margins** (each a distributional `StaticTAM`) plus a **copula** for the dependence. Map each margin through its probability integral transform to a normal score $z_j=\Phi^{-1}(F_j(y_j))$, estimate the score correlation $R$, and obtain a joint anomaly score from the Mahalanobis distance:

$$ \text{joint p-value} = \chi^2_d.\mathrm{sf}\big(z^\top R^{-1} z\big). $$
