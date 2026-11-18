# Caputo Fractional Physics Penalties

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/spectrum/FRACTIONAL_PHYSICS.md)

## Caputo Fractional Derivative Penalties

### Continuous Caputo operator
For fractional order $\alpha \in (0, 2)$, the continuous Caputo fractional derivative of a function $f(t)$ is defined by ({cite:p}`caputo1967linear`, {cite:p}`podlubny1999fractional`):

$${}^C D_t^\alpha f(t) = \frac{1}{\Gamma(n - \alpha)} \int_0^t \frac{f^{(n)}(\tau)}{(t - \tau)^{\alpha - n + 1}}\, d\tau, \qquad n = \lceil \alpha \rceil$$

A crucial property of the Caputo definition is that the fractional derivative of a constant is zero (${}^C D_t^\alpha (1) = 0$), preserving compatibility with TAM's global intercept.

### Discrete representation
1. **Fourier Domain**: The symbol of fractional differentiation is $(i\omega)^\alpha$. The corresponding spectral penalty scales as:

   $$P_{\text{frac}}^{(\alpha)} = \operatorname{diag}\big( |\omega|^{2\alpha} \big) \cdot \lambda_p$$

   which continuously bridges between Ridge ($\alpha \to 0$), Dirichlet total-variation ($\alpha = 1$), and cubic spline curvature ($\alpha = 2$).
2. **Spline Domain**: Evaluated via Grünwald–Letnikov binomial convolution weights $w_k^{(\alpha)} = (1 - \frac{\alpha+1}{k}) w_{k-1}^{(\alpha)}$ with lower-triangular Toeplitz structure and Caputo boundary normalization.

---

## Integer orders are unchanged

A derivative order that is an integer is read by the integer-order code that existed before: the penalty matrices of `phys()` with integer orders are bit-identical to those of the previous version (a parity test compares them). Only orders that are not integers select the Caputo operator.

## Causality and state

The penalty acts on the coefficients of the basis, not on the data: the effect is pointwise in the covariate, with no memory and no state.

## What is new here

The Caputo derivative and the Grünwald–Letnikov discretization are those of the cited references {cite:p}`caputo1967linear, podlubny1999fractional`. What this adds to the physics-informed penalty of `phys()`:

- **fractional derivative orders in the linear differential operator of the penalty**, so that the smoothness constraint can sit between a ridge and a curvature penalty;
- the discretization on the **spline and Fourier bases of the framework**, with the constant function in the null space.

## References

- Caputo, Michele (1967). *Linear models of dissipation whose Q is almost frequency independent-II*. Geophysical Journal International. `caputo1967linear`
- Podlubny, Igor (1999). *Fractional Differential Equations: An Introduction to Fractional Derivatives, Fractional Differential Equations, to Methods of Their Solution and Some of Their Applications*. Academic Press. `podlubny1999fractional`
- Doumeche, Nathan et al. (2025). *Physics-informed kernel learning*. Journal of Machine Learning Research. `doumeche2025physics`
