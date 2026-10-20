# AutoTAM Probabilistic Selection and Mathematical Validity

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)

* **Related code architecture:** [See the Code Architecture](../../architecture/meta/11_autotam_probabilistic_code.md)

* **Related AutoTAM topic:** [Evolutionary Orchestrator](08_auto_orchestrator.md)


> Mirror-Architecture math document (pairs with the page of the same number in the architecture mirror).
> This document establishes the validity of the AutoTAM evolutionary selection machinery under a probabilistic loss: variance and parsimony pruning, the Island-UCB reward, and conformal calibration. The structural-search loss is extended from the standard $\ell_2$ norm to the **expectile / pinball** geometry, which gives quantiles and conformal intervals. It answers four theoretical objections that this extension raises.

---

## 0. Setup and Notation

A TAM candidate is a coefficient vector $\theta$ over a fixed block basis $\Phi = [\Phi_1 \mid \dots \mid \Phi_L]$. For **any** fitting loss, the predictor on its native scale is:

$$f(x) \;=\; \sum_{l=1}^{L} h_l(x), \qquad h_l(x) = \Phi_l(x)\,\theta_l, \tag{1}$$

with each $h_l$ mean-centred under the GAM identifiability constraint, allowing the intercept to absorb the means. The **only** component the choice of loss changes is *which* $\theta$ the optimisation selects:

* **$\ell_2$ (legacy):** $\hat\theta = \arg\min \lVert Y - \Phi\theta\rVert_2^2 + n\,\theta^\top P\theta$, computed via one exact solve.


* **Expectile $\tau$:** $\hat\theta_\tau = \arg\min \sum_i \rho_\tau(y_i - \Phi_i\theta) + n\,\theta^\top P\theta$, where $\rho_\tau(r) = \vert{}\tau - \mathbb 1\{r<0\}\vert{}\,r^2$. It is solved by iteratively reweighted least squares (IRLS) over the same penalised weighted least-squares step.



Both formulations remain linear in the basis, so equation (1) holds verbatim for either.

---

## 1. Variance Pruning under Non-$\ell_2$ Geometry

**Objection.** Variance pruning of the knowledge graph drops term $l$ when $V_l := \operatorname{Var}(h_l)/\operatorname{Var}(f) < \gamma$; it might break under the expectile loss. The following propositions answer it:

* **Proposition 1 (Well-posedness is loss-invariant):** $V_l$ is a functional of the *fitted additive surface* (1), rather than the estimator that produced $\theta$. For any $\hat\theta$ (OLS or IRLS), the contributions $\{h_l\}$ exist, sum exactly to $f$, and possess finite empirical variances {cite:p}`newey1987asymmetric, jones1994expectiles`. Thus, $V_l$ is reproducible under expectile loss and is **not arbitrary**. The expectile estimator itself is a well-posed asymmetric-least-squares M-estimator with a consistent, finite asymptotic covariance {cite:p}`newey1987asymmetric`, computed with the same penalised weighted least-squares step.


* **Proposition 2 (Non-orthogonality is pre-existing):** An exact ANOVA partition ($\sum_l V_l = 1$) requires $\operatorname{Cov}(h_i,h_j)=0\ \forall i\neq j$. This strict orthogonality holds under **neither** $\ell_2$-with-correlated-bases **nor** expectile loss. The search already applies a redundancy guard that prunes $l$ when $\operatorname{corr}(h_l,h_j) > \rho^\star$, so the switch to expectile introduces **no new violation**.



**Conclusion.** While $V_l$ remains a valid structural measure, a more rigorous refinement aligns the importance with the loss geometry:

* **Proposition 3 (Loss-aligned importance):** Because the pinball loss is a *strictly proper scoring rule* for the quantile functional {cite:p}`gneiting2007strictly`, the decision-relevant importance for a model selected by $R_\tau(f)$ is the **ablation risk-increase** on the development block:



$$I_l \;=\; \frac{R_\tau(f_{-l}) - R_\tau(f)}{R_\tau(f)}, \qquad f_{-l} := f - h_l.$$



Pruning by $I_l < \gamma$ is *monotonically aligned* with the selection objective. The legacy $V_l$ is merely a surrogate for $I_l$.


* **Decision:** the probabilistic path prunes by the ablation importance $I_l$ (which generalises to $\Delta\text{CRPS}$), while the $\ell_2$ path keeps $V_l$ unchanged, so that its results are unchanged.



---

## 2. Incongruent UCB Reward Scaling

**Objection.** The Island-UCB reward of the $\ell_2$ path normalises an $\ell_2$ error by an $\ell_2$ scale $\sigma_Y$; applying $\mathrm{pinball}/\sigma_Y$ would improperly mix geometries.

**Answer.** Because pinball is a proper scoring rule {cite:p}`gneiting2007strictly`, the natural scale-free fitness is a **skill score** against the in-geometry baseline $f_0$ (the *unconditional empirical $\tau$-quantile*):

$$r_k \;=\; \frac{1}{\,R_\tau(f)\,/\,R_\tau(f_0) + \varepsilon\,}, \qquad R_\tau(f_0) = \min_{c}\tfrac1n\!\sum_i \rho^{\mathrm{pin}}_\tau(y_i - c).$$

This baseline $R_\tau(f_0)$ is an $L_1$-type quantity, ensuring the numerator and denominator share units and curvature. This creates a scale-free, bounded metric comparable across islands and $\tau$ levels. The $\sigma_Y$ normaliser is **not** used on the probabilistic path.

---

## 3. Conformal Calibration and Single-Branch Genome

**(a) Split-conformal on point residuals.** Applying the symmetric $\vert{}y-\hat\mu\vert{}$ score of split-conformal calibration around a single non-central $\tau=0.9$ estimate is mathematically unsound: point-residual calibration is valid for point champions only.

**(b) Single-branch genome for CQR.** The assumption that a single-branch genome prevents sound Conformalized Quantile Regression (CQR) is incorrect. CQR (Romano, Patterson, Candès 2019) requires two *quantile functions*, but does **not** strictly require a two-branch $(\mu,\sigma)$ genome. A single-branch genome *fit at two distinct levels* supplies them:

* **Proposition 4 (CQR coverage with single-branch fits):** Let $\hat q_{\mathrm{lo}}, \hat q_{\mathrm{hi}}$ be two single-branch TAM fits using expectile levels mapped from $\alpha/2,\,1-\alpha/2$. On an exchangeable calibration block, defining $E_i = \max\{\hat q_{\mathrm{lo}}(x_i) - y_i,\; y_i - \hat q_{\mathrm{hi}}(x_i)\}$ and setting $Q$ as the $\lceil (n{+}1)(1{-}\alpha)\rceil$-th smallest $E_i$, the interval $C(x) = [\hat q_{\mathrm{lo}}(x) - Q,\ \hat q_{\mathrm{hi}}(x) + Q]$ guarantees $\mathbb P(y \in C(x)) \ge 1-\alpha$ {cite:p}`romano2019conformalized`.


* Expectiles are mathematically admissible base estimators, and conformal calibration naturally repairs any level miscalibration. Furthermore, if the generated curves cross, pointwise monotone rearrangement preserves coverage without increasing estimation error {cite:p}`chernozhukov2010quantile`.



**Answer.** Intervals are **CQR fed by two single-branch quantile fits**, calibrated with the conformal quantile of the calibration block. Point-residual conformal calibration is reserved for genuine point champions.

---

## 4. Summary

| System Component | Design |
| --- | --- |
| **Parsimony Pruning** | **Ablation** importance $I_l=\Delta R_\tau/R_\tau$ on the probabilistic path; $V_l$ is kept for the $\ell_2$ path.

 |
| **UCB Reward** | A **skill score** evaluated against the unconditional-quantile baseline $R_\tau(f_0)$.

 |
| **Intervals** | **CQR** built on two single-branch $\tau$-fits; point-conformal solely for point champions.

 |
| **Genome Structure** | **The single-branch structure is kept**; CQR requires two distinct *fits*, not a specialized two-branch genome.

 |
