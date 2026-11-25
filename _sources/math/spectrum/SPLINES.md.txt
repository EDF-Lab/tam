# The Spline Effect (P-Splines)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## Foundational Theory: Continuous vs. Discrete Splines

In the original formulation of Generalized Additive Models (GAMs) {cite:p}`hastie2017generalized`, non-linear relationships are estimated using flexible smoothing functions. The mathematically optimal solution to this problem is the continuous Smoothing Spline, which operates within a Reproducing Kernel Hilbert Space (RKHS) by penalizing the continuous integral of squared second derivatives {cite:p}`wahba1990spline`. 

However, exact continuous Smoothing Splines theoretically require placing a knot at every single unique data point. This results in a dense Dual $\mathcal{O}(N_{total}^3)$ Gram matrix inversion that completely paralyzes standard hardware on Gigadata. 

To resolve this computational bottleneck, the TAM framework abandons the continuous integral approach and instead utilizes the discrete **P-spline (Penalized B-spline)** formulation pioneered by {cite:p}`eilers1996flexible, wood2017generalized`. P-splines entirely decouple the knot dimension $K$ from the sample size $N_{total}$ ($K \ll N_{total}$), explicitly truncating the continuous problem into a highly efficient, finite-dimensional Primal space $\mathcal{O}(N_{total} K^2)$.

## Formula Definition: B-Spline Basis ($\Phi$)

To construct the finite-dimensional feature map, the continuous domain $\mathcal{X}$ is partitioned by a sequence of $K$ fixed knots $\tau$. The input data is then projected into the Primal space using a basis of strictly local B-splines. 

In the `_spline.py` implementation, this is evaluated entirely on the GPU without slow python loops by natively executing the classical **Cox-de Boor recursion formula** {cite:p}`de1978practical`. A B-spline of degree zero (order $m=1$) is defined as a simple step function over the knot interval:

$$B_{i,1}(x) = \begin{cases} 1 & \text{if } \tau_i \le x < \tau_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

Higher-order B-splines (e.g., cubic splines where $m=4$) are constructed recursively via strictly positive, linearly interpolated divided differences:

$$B_{i,m}(x) = \frac{x - \tau_i}{\tau_{i+m-1} - \tau_i} B_{i,m-1}(x) + \frac{\tau_{i+m} - x}{\tau_{i+m} - \tau_{i+1}} B_{i+1,m-1}(x)$$

The final mapping matrix extracted for the Primal matrix $\Phi$ is the horizontally concatenated matrix of these recursive polynomial evaluations:

$$\phi_{spline}(x) = [B_{1,m}(x), \dots, B_{K,m}(x)]^\top$$

## The Discrete Difference Penalty ($P$)

Because the B-spline basis is strictly local, evaluating the continuous derivative integrals required by Wahba's formulation {cite:p}`wahba1990spline` is computationally wasteful. Eilers and Marx proved that applying a discrete difference operator directly to adjacent B-spline coefficients serves as a perfect algebraic proxy for the continuous derivative penalty {cite:p}`eilers1996flexible`.

In `_spline.py`, TAM executes this by generating an identity matrix and recursively applying `torch.diff` according to the requested `penalty_order` ($d$). If $d=2$, the penalty enforces a proxy for the second derivative (penalizing abrupt changes in slope). 

Let $\Delta_d$ represent the discrete difference operator matrix of order $d$. The strictly convex, block-diagonal structural penalty matrix $P_{spline}$ is explicitly constructed as:

$$P_{spline} = \lambda (\Delta_d^\top \Delta_d)$$

This formulation rigorously conserves the fundamental moments of the data (mean and variance) without the boundary bias typical of standard neural networks {cite:p}`eilers1996flexible`, and is a cornerstone of modern GAM theory {cite:p}`wood2017generalized`.

## Locally Adaptive Smoothing

A single $\lambda$ gives the same stiffness to the whole curve: a function that is flat over most of its range and sharply curved in a narrow window (a demand trough at the end of the year, an impact) is either oversmoothed in the window or wiggly elsewhere. The adaptive P-spline {cite:p}`ruppert2000spatially` lets the smoothing parameter vary along the curve by writing the penalty with a weight on every difference:

$$P_{spline}(\lambda_1,\dots,\lambda_m)=\sum_{r=1}^{m}\lambda_r\,\Delta_d^\top\operatorname{diag}(W_r)\,\Delta_d,\qquad W_r\ge 0,\quad \sum_{r=1}^{m}W_r=\mathbf 1,$$

where $W_r\in\mathbb R^{K-d}$ holds the values of the $r$-th of $m$ smooth B-spline basis functions (over the position of the difference in the coefficient sequence). Each part has its own smoothing parameter, chosen by the same generalized cross-validation as the others. Because the weights add up to one, equal parameters give back $\lambda\,\Delta_d^\top\Delta_d$ exactly, and $m=1$ is the ordinary penalty. The penalty stays a positive semi-definite quadratic form in the coefficients, so the fit is still one linear system.

## Boundary Extrapolation & Operational Safety

A known pathology of unconstrained polynomial basis expansions is Runge's phenomenon, catastrophic oscillatory explosions when evaluating data points outside the training domain (Out-of-Distribution). 

Original GAM frameworks mitigated this using Natural Cubic Splines, which enforce linearity beyond the outermost boundary knots {cite:p}`hastie2017generalized`. The `SplineEffect` natively supports this operational safety protocol via its `extrapolate` parameter. When `extrapolate='linear'` is triggered, `_spline.py` utilizes PyTorch finite differences at the boundary knots to compute the exact Jacobian (gradient) of the spline surface, explicitly projecting a stable, first-order Taylor extension infinitely outward.

### The edge slope of a clamped spline

Take a clamped B-spline of degree $k\ge 2$ with uniform interior spacing $h$ and coefficients $\theta_1,\dots,\theta_K$. By Marsden's identity the spline reproduces a straight line $a+bx$ when $\theta_j=a+b\,\xi_j$, where $\xi_j$ is the Greville abscissa (the mean of $k$ consecutive knots) of the $j$-th function {cite:p}`de1978practical`. With clamped end knots the last abscissae are

$$\xi_K=x_{\max},\qquad \xi_{K-1}=x_{\max}-\frac{h}{k},\qquad \xi_{K-2}=x_{\max}-\frac{2h}{3}\ \ (k=3),$$

so the coefficients of a linear trend are spaced by $b\,h/k$ at the edge. The derivative at the edge is

$$f'(x_{\max}^-)=\frac{k}{h}\,\bigl(\theta_K-\theta_{K-1}\bigr).$$

The coefficient differences next to the edge are thus a fraction $1/k$ of what the interior spacing would give, while a second-order difference penalty $\Delta_2^\top\Delta_2$ {cite:p}`eilers1996flexible` treats all differences alike and, with a free end, leaves the last one larger than a straight line would. The edge derivative $f'(x_{\max}^-)$ then exceeds the secant slope of the curve over the last knot span, by a factor of order $k/2$ on the fits we measured, and an extrapolation that continues this derivative outward overshoots on a trending input.

For a fit whose smoothing parameters come from the REML criterion (`auto_fit(criterion="reml")`) the outward slope used by the `linear` and `saturation` extrapolations is therefore multiplied by

$$s_k=\frac{2}{k}\quad(k\ge 2,\ d\ge 2),\qquad s=1\ \text{otherwise},$$

with $d$ the penalty order. Inside the range nothing changes, and a model with fixed penalty weights keeps $s=1$. A margin of a tensor product keeps $s=1$: the Kronecker penalty already bounds the edge magnitude of each margin.

## The Engineering Novelty

What distinguishes the `SplineEffect` in TAM is the translation of classical statistical mathematics into a purely matrix-free, GPU-native operation. By explicitly embedding the Cox-de Boor recursion and the precise discrete $\Delta_d^\top \Delta_d$ penalty into PyTorch tensors, the framework allows exact P-splines to be assembled block-diagonally alongside deep neural features (NNGP) and physics-informed constraints, solving the entire unified spectrum simultaneously via Sparse Conjugate Gradients.