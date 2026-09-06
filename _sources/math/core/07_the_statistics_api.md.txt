# One Atom, Many Statistics: The Statistics API

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related implementation:** [The Statistics API (architecture)](../../architecture/core/07_the_statistics_api.md)

Where the [Spectrum library](../spectrum/LINEAR.md) is a *heterogeneous* collection of bases, each effect a different $\Phi$/$P$, the Statistics layer is the mirror image: a **single** numerical primitive reused for *every* statistical target. This is the **"One Atom, Many Statistics"** philosophy.

## The Atom

Every fit reduces to one penalized weighted least-squares solve:

$$ \left(\Phi^\top W \Phi + n\,S\right)\theta = \Phi^\top W z. $$

A Gaussian mean fit calls it **once**; the default `loss="l2"` (with $W=I$) is bit-identical to the ordinary least-squares solver. Every non-Gaussian target changes **only** the working response $z$ and the per-observation weights $W = \mathrm{diag}(w_i)$, the high-performance linear algebra never changes, only the *schedule* that feeds it.

## The Three Schedules (and the Risk layer above)

1. **IRLS**: GLMs (Poisson/Gamma/Binomial), robust M-estimators and expectiles: $z=\eta+(y-\mu)g'(\mu)$, $w=1/(g'(\mu)^2 V(\mu))$. See [Reweighted estimation](../statistics/01_reweighted_estimation.md).
2. **Location-scale (distributional)**: a `{param: formula}` dict composes two sub-atoms (an L2 location on $\log y$, a Gamma scale on the squared residuals). See [Distributional](../statistics/02_distributional.md).
3. **EM mixture**: the M-step is the weighted atom with responsibilities as weights. See [Mixtures & copulas](../statistics/03_mixture_copula.md).

The post-fit **risk** bounds, conformal, ACI, EVT and epistemic uncertainty, operate on the predictions of any of these. See [Risk: conformal / EVT](../statistics/04_risk_conformal_evt.md).
