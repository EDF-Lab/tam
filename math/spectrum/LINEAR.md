# The Linear Effect (Ridge Regression)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## Formula Definition

The linear effect establishes the global baseline of the Generalized Additive Model (GAM) by projecting the input space $\mathcal{X} \subseteq \mathbb{R}^d$ identically onto itself, subject to an optional scaling factor $s$ (implemented as `scaled` in the framework). For a given input subspace, the explicit Primal mapping is defined as:

$$\phi_{lin}(x) = s \cdot x$$

Unlike spectral effects that project the input into a high-dimensional functional space, the Linear Effect operates directly in the scaled input space.

## Optimal Penalization

To bound the hypothesis space, we apply Tikhonov (Ridge) regularization {cite:p}`tikhonov1943stability, hoerl1970ridge`. The structural penalty sub-matrix is defined as a scaled identity matrix, $P_{lin} = \lambda I$ (where $\lambda$ represents the penalty weight, classically denoted as $\lambda$). Consequently, the penalty evaluated within the global unified risk translates exactly to the squared $L_2$ norm of the linear coefficients:

$$||P_{lin}^{1/2} \theta_{lin}||_2^2 = \lambda ||\theta_{lin}||_2^2$$

## RKHS Eligibility

This direct projection corresponds precisely to the homogeneous Dot Product Kernel (or Linear Kernel), defined as $k(x, x') = \langle x, x' \rangle = x^\top x'$ {cite:p}`williams2006gaussian`. 

By equipping the standard Euclidean space with the strictly positive definite penalty norm induced by $\lambda I$, the coefficients are rigorously bounded. This guarantees that the linear mapping satisfies Mercer's conditions, formally operating as a valid, finite-dimensional Reproducing Kernel Hilbert Space (RKHS). The framework mathematically ensures the strict invertibility of the global covariance matrix $\left( \Phi^\top \Lambda^\top \Lambda \Phi + n P \right)$. This stabilizes the global matrix resolution, entirely preventing numerical singularities without artificially suppressing the linear signal.

## The Offset (Intercept) Integration

A critical distinction of the TAM framework lies in its handling of the global bias (the Intercept). 

Standard practice in Ordinary Least Squares (OLS) and traditional Ridge regression leaves the intercept unpenalized. This ensures that the solution remains invariant to translations of the target variable $Y$, and guarantees that the intercept simply evaluates to the mean of the response when the predictors are centered {cite:p}`hastie2009elements`. However, the TAM algorithm abandons this separation to maintain maximum hardware efficiency and matrix uniformity. The bias is treated as a regularized weight natively integrated into the global linear system {cite:p}`doumeche2025forecasting`. 

To achieve this, the Offset is defined mathematically as a constant "phantom" linear effect, where the feature map evaluates to a vector of ones:

$$\phi_{offset}(x) = 1$$

Theoretically, applying a penalty $\lambda_{offset}$ to this vector equates to placing a zero-mean Gaussian prior $\mathcal{N}(0, \lambda_{offset}^{-1})$ strictly on the mean of the process {cite:p}`williams2006gaussian`. By adjusting this specific $\lambda$ toward zero (e.g., $10^{-3}$), the prior variance approaches infinity. This seamlessly recovers the unpenalized, translation-invariant behavior of classical models (acting as an improper prior) while keeping the global algebraic resolution perfectly intact and OOM-safe.