# Exp-Centered Units (ExU)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/spectrum/EXU.md)

## Exp-Centered Units (`ExU`) for Jagged Regimes

Standard smooth activations ($\operatorname{Tanh}$, $\operatorname{SiLU}$) require divergent weight norms to approximate sharp step functions (e.g. tariff switches, breaker trips). Following Agarwal et al. ({cite:p}`agarwal2021neural`), Exp-Centered Units parameterize pre-activations in log-weight space:

$$\phi_{\text{ExU}}(x; w, b) = \operatorname{clip}_{[0, 1]}\!\Big( \operatorname{ReLU}\!\big( e^w (x - b) \big) \Big)$$

with $w \sim \mathcal{N}(3.5, 0.5^2)$ producing sharp local ramps of width $e^{-w} \approx 0.03$, and shifts $b \sim \mathcal{U}[-1, 1]$ distributed across the covariate range. Linear combinations $\Phi_{\text{ExU}}\theta$ fit piecewise-constant step functions without Gibbs oscillations.

---

## Default activations are unchanged

The unit is only used when `activation='exu'` is asked: the activations `relu`, `tanh` and `cos` of the neural effect are the ones of the previous version, bit for bit (a parity test checks it). The effect is pointwise in the covariate: no memory, no state.

## What is new here

The exp-centered unit is the one of the cited paper {cite:p}`agarwal2021neural`. What this adds:

- the **ExU activation as a fixed random feature of the neural effect**, with log-weights and shifts drawn once, so that sharp steps are represented by a linear combination in a closed-form penalized term instead of a trained network.

## References

- Agarwal, Rishabh et al. (2021). *Neural additive models: Interpretable machine learning with neural nets*. Advances in neural information processing systems. `agarwal2021neural`
- Wood, Simon N (2017). *Generalized additive models: an introduction with R*. chapman and hall/CRC. `wood2017generalized`
