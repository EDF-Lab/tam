# Hierarchical Joint Optimization [BETA]

> 🚧 **Beta – Research Module**

⚠️ **Status: Active development**

- Core functionality is implemented but still evolving  
- Some features may be incomplete or subject to change  
- API is not yet stable  
- Intended for research use only (not production-ready)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/03_hierarchical_code.md)

## Sinkhorn Entropic Optimal Transport Temporal Coherence

To reconcile temporal hierarchies across steps without excessive quadratic smoothing, the coherence penalty can be formulated via a doubly-stochastic Sinkhorn transport plan $P^* \in \mathbb{R}^{K \times K}$ over the temporal step cost matrix $C_{ij} = (i - j)^2$:

$$P^* = \operatorname{Sinkhorn}\!\left(\exp\left(-\frac{C}{\varepsilon}\right)\right), \qquad L_{\text{sinkhorn}} = I - P^*$$

producing a positive semi-definite Laplacian $L_{\text{sinkhorn}}^\top L_{\text{sinkhorn}}$ that couples neighboring horizons softly in proportion to entropic regularization $\varepsilon > 0$.
