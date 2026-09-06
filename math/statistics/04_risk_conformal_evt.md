# Risk: Conformal, ACI, EVT & Epistemic Uncertainty

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Synthesis:** [One Atom, Many Statistics](../core/07_the_statistics_api.md)
  * **Related implementation:** [Risk (architecture)](../../architecture/statistics/04_risk_conformal_evt_code.md)

This chapter establishes the mathematical foundation for the framework's uncertainty quantification. The risk layer bounds predictions *after* the fit. It operates on raw arrays / duck-typed model outputs, so any meta-model's predictions flow in identically. 

One source of truth: the static engine `SafetyTAM` holds the finite-sample quantile and p-value; the layers below build on it.

---

## Split Conformal Prediction & CQR

Unlike traditional Bayesian confidence intervals (which depend heavily on prior distributional assumptions) or bootstrap methods, Conformal Prediction offers a finite-sample marginal coverage guarantee without assuming anything about the underlying error distribution, provided the data points are *exchangeable* {cite:p}`angelopoulos2023conformal,vovk2005algorithmic`.

The framework implements the **Split Conformal** method {cite:p}`lei2018distribution` to ensure computational efficiency. The mathematical procedure is defined as follows:

1. **Data Splitting:** The historical data is divided into a disjoint Training set and a Calibration set of size $n$.
2. **Non-Conformity Scores:** We define a non-conformity measure. For standard regression, this is the absolute residual $s_i = \left| Y_i - \Phi_i \hat{\theta} \right|$. `ConformalDistributionalTAM` uses a Conformalized Quantile Regression (CQR) score {cite:p}`romano2019conformalized` defined as $s_i = \max(\hat{Q}_{\alpha/2}-Y_i,\ Y_i-\hat{Q}_{1-\alpha/2})$ and supports **Mondrian** (per-stratum) calibration.
3. **Finite-Sample Quantile:** To guarantee exact marginal coverage of $1 - \alpha$, we compute the empirical quantile $\hat{q}$ of the calibration scores at a corrected level:
   $$ \hat q = \mathrm{Quantile}\!\left(\{s_i\},\ (1-\alpha)\big(1+\tfrac1n\big)\right) $$
4. **Prediction Interval:** For any new observation at time $t$, the prediction interval is symmetrically defined around the baseline by $\hat{q}$.

---

## Adaptive Conformal Inference (ACI)

The fundamental vulnerability of the Static mode in industrial time series is that the *exchangeability assumption is frequently violated*. Continuous Concept Drift causes the underlying data distribution to shift {cite:p}`principato2025blackwell`. 

To maintain valid coverage in non-stationary environments, TAM implements the **Adaptive Conformal Inference (ACI)** algorithm {cite:p}`gibbs2021adaptive`. Instead of relying on a fixed historical quantile, ACI dynamically adjusts the risk parameter $\alpha_t$ at every time step using an online feedback loop:

$$ \alpha_{t+1} = \alpha_t + \gamma\,(\alpha_{\text{target}} - \mathrm{err}_t), \qquad \mathrm{err}_t=\mathbb 1\{Y_t\notin \hat C_t\} $$

If the model makes an error ($\text{err}_t = 1$), the effective $\alpha_{t+1}$ decreases, forcing the model to widen the interval for the next step. If safely covered ($\text{err}_t = 0$), $\alpha_{t+1}$ increases, tightening the interval. Tracking cumulative miscoverage errors guarantees asymptotic long-term validity even under arbitrary, adversarial distribution shifts {cite:p}`principato2025blackwell`.

---

## Extreme Value Theory (EVT)

A parametric location-scale fit constrains the whole shape and may misfit the extreme tail. By the Pickands-Balkema-de Haan theorem {cite:p}`pickands1975statistical,balkema1974residual`, exceedances of a high threshold $u$ follow a Generalized Pareto Distribution (GPD). `fit_gpd_tail` models these upper tails, giving asymptotically-justified tail probabilities and an EVT surprisal score $-\log P(M>m)$ for severe anomalies.

---

## Epistemic (Parameter) Uncertainty

The penalized fit is a MAP estimate under a Gaussian prior {cite:p}`wahba1983bayesian`. The posterior covariance is $V_\beta=\sigma^2(\Phi^\top\Phi+nS)^{-1}$, and the epistemic standard error $\sqrt{\phi V_\beta \phi^\top}$ widens where the training design is data-poor. 

*Note: This is computed via an efficient Cholesky solve, never requiring a dense matrix inverse.*

---

## Hierarchical Limitations & Future Upgrades

The current `SafetyTAM` module operates marginally—meaning it calculates safety bounds for each node independently. However, Conformal Prediction intervals do not obey linear summation. Summing the marginal intervals of the children (via Minkowski sum) does not mathematically yield the correct joint interval for the aggregated parent {cite:p}`principato2024conformal`. Applying ACI independently level-by-level inevitably breaks the physical realities of the hierarchy.

**Roadmap:** Future iterations of the framework must implement interval reconciliation via polytope projection to strictly enforce hierarchical aggregation constraints across the entire uncertainty band {cite:p}`principato2024conformal`.