# AutoTAM Feature Profiling, Basis and Capacity Diagnosis

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/meta/12_autotam_feature_profiling_code.md)
* **Related AutoTAM topic:** [Evolutionary Orchestrator](08_auto_orchestrator.md)

> Mirror-Architecture math doc (pairs with `architecture/meta/12_autotam_feature_profiling_code.md`).
> Establishes why basis capacity, not generalisation, sets the accuracy ceiling of an AutoTAM champion, and derives the estimators used to measure that capacity before the evolutionary search.

---

## 1. Motivation: Capacity is the Ceiling

Let $\hat f$ be a frozen static champion model and $\hat f_{\mathrm{ada}}$ be the exact same champion corrected online by an error-correction model acting on its decomposed components. In the presence of non-stationary time-series data, empirical and mathematical regularities demonstrate that:

$$\mathrm{RMSE}_{\text{test}}(\hat f_{\mathrm{ada}}) \;\approx\; \mathrm{RMSE}_{\text{train}}(\hat f), \qquad \mathrm{RMSE}_{\text{test}}(\hat f) \;\approx\; \text{const}.$$

The frozen static out-of-sample error is dominated by the regime shift and is nearly model-independent. Once an online corrector mathematically removes that structural break, what remains is the base model's underlying ability to *represent* the signal, its training error. Accuracy is therefore bounded below by **representational capacity**, rather than by generalisation or the aggregation layer. The primary optimisation target must therefore be the reduction of the base model's training error.

Furthermore, the structural complexity penalty is not the limiting constraint in this regime. With an annealed penalty factor $1+\alpha k/N$, when the sample size $N$ is massive relative to the degrees of freedom $k$ ($N \gg k$), the ratio $k/N$ approaches zero. Structural capacity is therefore essentially "free" from an overfitting perspective, leaving no overfitting margin to defend. The mathematical challenge is purely to diagnose the required capacity directly from the signal.

---

## 2. Diagnosing on Partial Residuals

An additive model attributes to each feature only the specific signal left over by the other features. Diagnosing a feature strictly on its raw marginal $y$ vs $x_j$ mathematically mis-types correlated covariates, causing each to inherit the other's curvature.

Let $\hat\beta$ solve one ridge problem covering all candidate features. The **partial residual** for feature $j$ is calculated as:

$$r_j \;=\; y - \sum_{i \neq j} \hat\beta_i x_i \;=\; \big(y - X\hat\beta\big) + \hat\beta_j x_j,$$

This represents the full residual with feature $j$'s own linear contribution added back in. This is the standard backfitting diagnostic {cite:p}`hastie2017generalized`: it is the exact mathematical quantity a GAM smoother for feature $j$ would be fitted against, making it the correct object on which to judge $j$'s required topological shape.

**Panel Structure:** Because the solver is block-diagonal over the grouping key, group level shifts are absorbed by group-specific intercepts and must be excluded from the diagnosis. We apply the within-group (fixed-effects) transform $\tilde z = z - \bar z_{g(i)}$ to the response and to every feature prior to the ridge solve. As a result, the diagnosis evaluates only within-group variation, perfectly matching what the actual mathematical solver fits.

---

## 3. Stage 1: Is a Linear Term Enough?

We partition the support of $x_j$ into $B$ quantile bins with occupancies $n_b$ and bin means $\bar r_b$ of the partial residual. We define the **correlation ratio** as:

$$\eta^2 \;=\; \frac{\sum_b n_b\,(\bar r_b - \bar r)^2}{\sum_i (r_{j,i} - \bar r)^2},$$

This represents the fraction of variance explained by conditioning on $x_j$ *in any form*. We also define $r^2 = \operatorname{corr}(x_j, r_j)^2$, which is the fraction a straight line explains. Because a straight line is merely one particular measurable function of $x_j$, the following inequality strictly holds:

$$0 \;\le\; r^2 \;\le\; \eta^2 \;\le\; 1,$$

The excess $\eta^2 - r^2$ is exactly the variance explainable by $x_j$ that a linear term cannot reach. Declaring the feature linear when $\eta^2 - r^2 < \tau_{\mathrm{lin}}$ serves as a non-parametric lack-of-fit criterion that requires no iterative model fit. This relies on the classical $\eta$ of Pearson and the standard one-way lack-of-fit decomposition.

---

## 4. Stage 2: Which Basis, and at What Capacity?

**Harmonics.** Let $\{\bar r_b\}$ be the binned profile, mathematically centred, with discrete Fourier coefficients $\hat c_\nu$. By Parseval's theorem, the profile's energy decomposes perfectly over frequency, allowing us to take the smallest $m$ whose cumulative share clears a defined target $\rho$:

$$m \;=\; \min\Big\{ M : \frac{\sum_{\nu=1}^{M}\vert{}\hat c_\nu\vert{}^2}{\sum_{\nu\ge1}\vert{}\hat c_\nu\vert{}^2} \;\ge\; \rho \Big\}.$$

This defines energy compaction: $m$ represents the exact number of harmonics required to reproduce the response shape to a stated mathematical fidelity. Capacity is therefore **measured directly from the signal** rather than blindly searched.

**Knots.** A cubic penalized spline typically resolves roughly one oscillation per two interior knots. Therefore, a spline matching an $m$-harmonic Fourier fit structurally requires:

$$k \;\approx\; 2m .$$

**Localisation.** For a profile with range $R$, the normalised total variation $\mathrm{TV}(\bar r)/R = \sum_b \vert{}\bar r_{b+1} - \bar r_b\vert{} / R$ equals $1$ for a strictly monotone curve and grows proportionally with the number of reversals. If this exceeds a set threshold, the response is mathematically better modeled by a localised (wavelet) basis than a global smooth one. The profile is smoothed prior to taking the statistic, as bin-level sampling noise would otherwise accumulate into $\mathrm{TV}$ and erroneously report a smooth periodic curve as spiky.

---

## 5. Indices and Lags are Excluded by Construction

For a strictly monotone counter (a trend index) or an autoregressive lag, the map $b \mapsto \bar r_b$ is effectively a re-indexing of the residual *time series*. Its apparent roughness mathematically reflects serial correlation rather than true curvature in a response, meaning any structural capacity fitted there models pure noise. Such features are strictly held at a linear term with a near-zero penalty. The most recent autoregressive lag (the smallest lag order) may additionally take a PID term from any Island (`pid(w=W)`): this applies three linear coefficients on the lag, its rolling mean over $W$ steps, and its first difference, ensuring the term remains structurally linear in the lag and the lock is maintained.

Monotonicity must be tested **strictly**. A cyclic coordinate such as time-of-year is *almost* monotone (over $N$ samples spanning $Y$ years it contains only $Y-1$ decreasing steps, yielding a negligible fraction $O(Y/N)$). Consequently, adding any mathematical tolerance to the monotone test would incorrectly swallow precisely the seasonal features the diagnosis is designed to protect.

---

## 6. Prior, Not Constraint

The linearity verdict acts as a formal hypothesis test utilizing a controlled statistic and is applied as a hard lock. The basis-family verdict is a weaker inference and is therefore mathematically applied as a **decaying prior**. The diagnosed mathematical family is sampled with probability:

$$p(n) \;=\; \frac{p_0}{1 + n/n_0},$$

where $n$ counts the number of observations the Bayesian knowledge graph has accumulated for that specific feature. Early in the process, the search is steered heavily by the diagnosis. As empirical evidence arrives, the prior washes out and the bandit's own gathered statistics govern the search, ensuring that a mis-diagnosis is structurally recoverable rather than permanently locked in.

---

## 7. Tensor Products: Ranking Cost vs. True Width

A tensor product composed of sub-bases with dimensions $d_1,\dots,d_p$ mathematically spans $\prod_i d_i$ Kronecker columns. That product dictates the **true width** and must remain authoritative wherever coefficient blocks are physically sliced from $\theta$.

However, charging this true width to the parsimony penalty prices an interaction out of the search before it can ever demonstrate predictive value. For *ranking purposes only*, we substitute the following surrogate:

$$k_{te} \;=\; \min\Big( \prod_i d_i,\;\; \sum_i d_i + \sqrt{\textstyle\prod_i d_i} \Big),$$

This surrogate grows with the interaction size but far more slowly than the full product. The outer minimum is strictly required: when one sub-basis is a two-level categorical ($d=1$), the surrogate would otherwise exceed the very product it is mathematically designed to discount. This substitution touches absolutely no linear algebra; the design matrix, the penalty matrix, and the coefficient layout remain entirely unchanged.

**Status of the surrogate:** $k_{te}$ functions as an empirical ranking heuristic chosen explicitly to keep interactions from being priced out of the search before evaluation. It is not a complexity measure derived from effective degrees of freedom or from the norm of the tensor-product RKHS. It is nonetheless mathematically well behaved for $p \ge 2$ sub-bases with $d_i \ge 1$:

1. $\max_i d_i \le k_{te} \le \prod_i d_i$. The upper bound is guaranteed by the outer minimum. For the lower bound, $\prod_i d_i \ge \max_i d_i$ because every factor is at least $1$, and $\sum_i d_i + \sqrt{\prod_i d_i} \ge \sum_i d_i \ge \max_i d_i$. Thus, an interaction is never ranked as cheaper than its largest marginal component, nor as more expensive than its true width.
2. $k_{te}$ is non-decreasing in every $d_i$, as it is the minimum of two non-decreasing functions: adding a richer sub-basis will never lower the calculated cost.
3. If every $d_i \ge m$ and $m \to \infty$, then $k_{te} / \prod_i d_i \to 0$. With $D = \max_i d_i$, the ratio $\sum_i d_i \big/ \prod_i d_i \le p\,D \big/ (D\,m^{p-1}) = p / m^{p-1}$ and $\sqrt{\prod_i d_i} \big/ \prod_i d_i = (\prod_i d_i)^{-1/2}$. This mathematically enforces the discount the substitution exists to provide.

---

## 8. Theoretical Computational Cost

The diagnostic process requires one exact ridge solve on an $[N \times p]$ matrix plus, per feature, a binning pass and a Fast Fourier Transform (FFT) over $B$ quantile bins. This yields a theoretical complexity of:

$$O(Np^2 + pN + pB\log B)$$

Because $B \ll N$, this complexity is strictly dominated by the single linear solve. Because no iterative model fits enter this loop, the diagnosis mathematically evaluates partial residuals without adding any exposure to the intermittent VRAM exhaustion that typically affects heavy grouped tensor operations.
