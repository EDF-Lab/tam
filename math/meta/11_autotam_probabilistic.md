# AutoTAM Probabilistic Selection and Mathematical Validity

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)

* **Related code architecture:** [See the Code Architecture](../../architecture/meta/11_autotam_probabilistic_code.md)

* **Related AutoTAM topic:** [Evolutionary Orchestrator](08_auto_orchestrator.md)


> Mirror-Architecture math document outlining the planned implementation (pairs with `architecture/meta/11_autotam_probabilistic_code.md`).
> This document establishes the validity of the planned AutoTAM evolutionary selection machinery, variance/parsimony pruning, Island-UCB reward, and conformal calibration for the upcoming update. The structural-search loss will be expanded from the standard $\ell_2$ norm to include the **expectile / pinball** geometry to enable quantiles and conformal intervals. It also addresses the four theoretical objections raised in review to establish a clear roadmap for the code.

---

## 0. Setup and Notation

A TAM candidate is a coefficient vector $\theta$ over a fixed block basis $\Phi = [\Phi_1 \mid \dots \mid \Phi_L]$. For **any** fitting loss, the predictor on its native scale is:

$$f(x) \;=\; \sum_{l=1}^{L} h_l(x), \qquad h_l(x) = \Phi_l(x)\,\theta_l, \tag{1}$$

with each $h_l$ mean-centred under the GAM identifiability constraint, allowing the intercept to absorb the means. The **only** component the planned loss update changes is *which* $\theta$ the optimisation selects:

* **$\ell_2$ (legacy):** $\hat\theta = \arg\min \lVert Y - \Phi\theta\rVert_2^2 + n\,\theta^\top P\theta$, computed via one exact solve.


* **Expectile $\tau$ (Planned modification):** $\hat\theta_\tau = \arg\min \sum_i \rho_\tau(y_i - \Phi_i\theta) + n\,\theta^\top P\theta$, where $\rho_\tau(r) = \vert{}\tau - \mathbb 1\{r<0\}\vert{}\,r^2$. This will be solved by IRLS over the same P-WLS atom (`BaseTAM._solve_pwls_step`).



Both formulations will remain linear-in-basis, meaning equation (1) holds verbatim and fundamentally drives the planned design.

---

## 1. Planned Resolution for Objection 1: Variance Pruning under Non-$\ell_2$ Geometry

During the review, it was noted that `KnowledgeGraph.update_and_prune` (which drops term $l$ when $V_l := \operatorname{Var}(h_l)/\operatorname{Var}(f) < \gamma$) might break under the expectile loss. The implementation plan addresses this via the following propositions:

* **Proposition 1 (Well-posedness is loss-invariant):** $V_l$ is a functional of the *fitted additive surface* (1), rather than the estimator that produced $\theta$. For any $\hat\theta$ (OLS or IRLS), the contributions $\{h_l\}$ exist, sum exactly to $f$, and possess finite empirical variances {cite:p}`newey1987asymmetric, jones1994expectiles`. Thus, $V_l$ is reproducible under expectile loss and is **not arbitrary**. The expectile estimator itself will be a well-posed asymmetric-least-squares M-estimator with a consistent, finite asymptotic covariance {cite:p}`newey1987asymmetric`, computed via the existing P-WLS atom.


* **Proposition 2 (Non-orthogonality is pre-existing):** An exact ANOVA partition ($\sum_l V_l = 1$) requires $\operatorname{Cov}(h_i,h_j)=0\ \forall i\neq j$. This strict orthogonality holds under **neither** $\ell_2$-with-correlated-bases **nor** expectile loss. The engine already applies a redundancy guard to prune $l$ when $\operatorname{corr}(h_l,h_j) > \rho^\star$ (`is_redundant`), meaning the switch to expectile introduces **no new violation**.



**Conclusion & Action Plan:** While $V_l$ remains a valid structural measure, we will adopt a more rigorous refinement to explicitly align with the loss geometry:

* **Proposition 3 (Loss-aligned importance):** Because the pinball loss is a *strictly proper scoring rule* for the quantile functional {cite:p}`gneiting2007strictly`, the decision-relevant importance for a model selected by $R_\tau(f)$ will be implemented as the **ablation risk-increase** on the development block:



$$I_l \;=\; \frac{R_\tau(f_{-l}) - R_\tau(f)}{R_\tau(f)}, \qquad f_{-l} := f - h_l.$$



Pruning by $I_l < \gamma$ is *monotonically aligned* with the selection objective. The legacy $V_l$ is merely a surrogate for $I_l$.


* **Modification decision:** The probabilistic path will be updated to prune strictly by the ablation importance $I_l$ (which will generalize to $\Delta\text{CRPS}$ in further modification), while the legacy $\ell_2$ path will keep $V_l$ unchanged to ensure prior runs remain bit-identical.



---

## 2. Planned Resolution for Objection 2: Incongruent UCB Reward Scaling

The review correctly noted that the legacy Island-UCB reward normalizes an $\ell_2$ error using an $\ell_2$ scale $\sigma_Y$, and applying $\mathrm{pinball}/\sigma_Y$ would improperly mix geometries.

**Action Plan:** Because pinball is a proper scoring rule {cite:p}`gneiting2007strictly`, the natural scale-free fitness will be implemented as a **skill score** against the in-geometry baseline $f_0$ (the *unconditional empirical $\tau$-quantile*):

$$r_k \;=\; \frac{1}{\,R_\tau(f)\,/\,R_\tau(f_0) + \varepsilon\,}, \qquad R_\tau(f_0) = \min_{c}\tfrac1n\!\sum_i \rho^{\mathrm{pin}}_\tau(y_i - c).$$

This baseline $R_\tau(f_0)$ is an $L_1$-type quantity, ensuring the numerator and denominator share units and curvature. This creates a scale-free, bounded metric comparable across islands and $\tau$ levels. The legacy $\sigma_Y$ normalizer will **not** be used on the new probabilistic path.

---

## 3. Planned Resolution for Objection 3: Conformal Calibration and Single-Branch Genome

**(a) Split-conformal on point residuals:** We agree that applying `SafetyTAM`'s symmetric $\vert{}y-\hat\mu\vert{}$ score around a single non-central $\tau=0.9$ estimate is mathematically unsound. The shorthand draft suggesting to "wrap the champion in `SafetyTAM`" will be corrected in the final code.

**(b) Single-branch genome for CQR:** The assumption that a single-branch genome prevents sound Conformalized Quantile Regression (CQR) is incorrect. CQR (Romano, Patterson, Candès 2019) requires two *quantile functions*, but does **not** strictly require a two-branch $(\mu,\sigma)$ genome. We will use a single-branch genome *fit at two distinct levels* to supply them:

* **Proposition 4 (CQR coverage with single-branch fits):** Let $\hat q_{\mathrm{lo}}, \hat q_{\mathrm{hi}}$ be two single-branch TAM fits using expectile levels mapped from $\alpha/2,\,1-\alpha/2$. On an exchangeable calibration block, defining $E_i = \max\{\hat q_{\mathrm{lo}}(x_i) - y_i,\; y_i - \hat q_{\mathrm{hi}}(x_i)\}$ and setting $Q$ as the $\lceil (n{+}1)(1{-}\alpha)\rceil$-th smallest $E_i$, the interval $C(x) = [\hat q_{\mathrm{lo}}(x) - Q,\ \hat q_{\mathrm{hi}}(x) + Q]$ will guarantee $\mathbb P(y \in C(x)) \ge 1-\alpha$ {cite:p}`romano2019conformalized`.


* Expectiles are mathematically admissible base estimators, and conformal calibration naturally repairs any level miscalibration. Furthermore, if the generated curves cross, pointwise monotone rearrangement preserves coverage without increasing estimation error {cite:p}`chernozhukov2010quantile`.



**Action Plan:** The engine will implement **CQR fed by two single-branch quantile fits**, calibrated via `SafetyTAM`'s conformal-quantile primitive. The legacy `SafetyTAM`-on-point-residuals method will be strictly reserved for genuine point champions (`objective="point"`, `needs_conformal=True`).

---

## 4. Mirror Architecture Proof Requirement

In accordance with `CONTRIBUTING` §2, this `math/` document and its theoretical discussion fulfill the requirement for new features. Its `architecture/` companion (`11_autotam_probabilistic_code.md`) will be drafted subsequently to cover the software engineering (IRLS schedule reuse, ablation computation, and conformal calibration). No probabilistic code will be committed to the framework before this pair is fully reviewed.

---

## 5. Summary of the Planned Implementation

| System Component | Planned Rigorous Modification Design |
| --- | --- |
| **Parsimony Pruning** | Implement **ablation** importance $I_l=\Delta R_\tau/R_\tau$ on the probabilistic path; keep $V_l$ for the $\ell_2$ path.

 |
| **UCB Reward** | Implement a **skill score** evaluated against the unconditional-quantile baseline $R_\tau(f_0)$.

 |
| **Intervals** | Execute **CQR** built on two single-branch $\tau$-fits; limit point-conformal solely to point champions.

 |
| **Genome Structure** | **Retain the single-branch structure**; CQR requires two distinct *fits*, not a specialized two-branch genome.

 |
| **Documentation Process** | Submit this `math/` document and its `architecture/` counterpart strictly following the Mirror Architecture rules.
