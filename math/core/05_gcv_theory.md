# Hyperparameter Optimization: Generalized Cross-Validation (GCV)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/05_gcv_implementation.md)

This chapter details the mathematical calibration of the continuous regularization parameters ($\lambda$) within the TAM framework. It explains how the model autonomously balances structural complexity against empirical fit, enabling real-time Auto-ML without relying on computationally prohibitive out-of-sample validation sets.

---

## The Bottleneck of K-Fold Validation

Tuning hyperparameters (such as the penalty strength $\lambda$) traditionally requires K-Fold Cross Validation. For large-scale time series models involving Gigadata, retraining the system $K$ times per parameter combination across a massive grid is prohibitively slow and computationally intractable.

To solve this, the framework implements the **Generalized Cross-Validation (GCV)** theory {cite:p}`golub1979generalized`. GCV mathematically approximates the Leave-One-Out (LOO) validation error analytically in a single pass, completely bypassing the need for iterative data splitting.

The fundamental goal of hyperparameter optimization is to satisfy the principle of Structural Risk Minimization (SRM). As defined by {cite:p}`vapnik2013nature`, minimizing empirical training error alone is insufficient; one must also minimize a bound on the structural complexity of the model to guarantee out-of-sample generalization. The GCV algorithm mathematically solves this SRM tradeoff by dynamically finding the optimal continuous penalty $\lambda$ without requiring an explicit hold-out validation set.

---

## Connecting GCV to the Primal Matrices

To understand the GCV objective, we must first map it to the core matrices of the framework's Primal solver. Recall the exact equation for the estimated coefficients:

$$\hat{\theta} = \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1} \Phi^T \Lambda^T \Lambda Y$$

The predicted values (the fitted model) are derived by projecting these coefficients back through the design matrix $\Phi$:

$$\hat{Y} = \Phi \hat{\theta} = \left[ \Phi \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1} \Phi^T \Lambda^T \Lambda \right] Y$$

**1. The Smoothing Matrix ($H$):**
The bracketed term above maps the true targets $Y$ to the predicted targets $\hat{Y}$. In statistical literature, this is known as the "Hat Matrix" or the Smoothing Matrix $H$ {cite:p}`wood2004stable`:

$$H = \Phi \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1} \Phi^T \Lambda^T \Lambda$$

**2. The Residual Sum of Squares ($RSS$):**
The $RSS$ is simply the weighted squared empirical error of the fitted model:

$$RSS = || \Lambda (Y - \hat{Y}) ||^2 = || \Lambda (I - H) Y ||^2$$

---

## The Mathematical Demonstration: From LOO to GCV


How does evaluating the smoothing matrix $H$ replace data splitting?

In classic Leave-One-Out Cross-Validation (LOOCV), the model is trained $n$ times, leaving out one observation $i$ each time to predict its value. Thanks to the Sherman-Morrison-Woodbury matrix identity, it is a proven theorem that LOOCV can be calculated exactly without any retraining, using only the diagonal elements of the smoothing matrix ($H_{ii}$) {cite:p}`golub1979generalized`:

$$LOOCV = \frac{1}{T} \sum_{i=1}^T \left( \frac{y_i - \hat{y}_i}{1 - H_{ii}} \right)^2$$

However, Golub, Heath, and Wahba {cite:p}`golub1979generalized` proved that LOOCV is not rotation-invariant and can become highly unstable if the design matrix is unbalanced (e.g., if a specific $H_{ii}$ approaches 1).

To fix this, **Generalized Cross-Validation** replaces the individual diagonal elements $H_{ii}$ with their global average. The sum of the diagonal elements of a matrix is its trace, so the average is $\frac{1}{T} Tr(H)$. Substituting this average into the LOOCV formula yields the GCV objective {cite:p}`golub1979generalized`:

$$GCV(\lambda) = \frac{\frac{1}{T} \sum_{i=1}^T (y_i - \hat{y}_i)^2}{\left(1 - \frac{1}{T} Tr(H)\right)^2} = \frac{\frac{1}{T} RSS}{\left(1 - \frac{1}{T} Tr(H)\right)^2}$$

### Autocorrelation Inflation ($\gamma$)
The framework evaluates a strictly modified version of this equation:

$$GCV(\lambda) = \frac{\frac{1}{T} RSS}{\left(1 - \frac{\gamma}{T} Tr(H)\right)^2}$$

Where $\gamma$ is an inflation multiplier (typically $\gamma = 1.4$). Standard GCV assumes independent and identically distributed (i.i.d.) noise. In chaotic time-series environments, errors are frequently autocorrelated, causing uncorrected GCV to severely overfit the noise. The $\gamma$ parameter artificially inflates the penalty of the trace, forcing the solver to favor smoother, more robust functional mappings as theoretically justified by Kim and Gu {cite:p}`kim2004smoothing`.

---

## The Cyclic Trace Trick

A direct application of the GCV formula requires computing the trace of the smoothing matrix $H$. In the Dual space, $H$ is an $T \times T$ matrix. Computing its trace requires $\mathcal{O}(T^3)$ operations, which would cause an immediate bottleneck.

To completely decouple the hyperparameter optimization time from the number of rows in the dataset, the framework exploits the cyclic property of the trace operator ($Tr(AB) = Tr(BA)$).

By cyclically permuting the matrices, the framework evaluates the trace purely in the Primal feature space ($D \times D$):

$$Tr(H) = Tr\left(\Phi^T \Lambda^T \Lambda \Phi \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1}\right)$$

This allows the framework to compute the exact degrees of freedom instantaneously, even on datasets with millions of observations, simply by multiplying the weighted feature covariance matrix by the inverted regularized system.

---

## Multiple Smoothing Parameters (MSP) via Coordinate Descent

Because TAM concatenates highly heterogeneous mathematical bases (e.g., mixing Splines with Fourier series and Random Features), applying a single global penalty $\lambda$ is statistically invalid. The framework must optimize a different regularization parameter for *each* architectural block simultaneously {cite:p}`wood2004stable`.



While classical statistics recommends using a Newton-Raphson method based on exact analytical derivatives to solve Multiple Smoothing Parameters, calculating these higher-order derivative tensors requires massive memory allocations that frequently crash GPU VRAM.

**The Engineering Compromise:**
To safely scale, TAM deploys a Discrete Coordinate Descent algorithm {cite:p}`wright2015coordinate`, seeded once from the penalties the formula declares. Rather than computing the dense Hessian of the GCV landscape, the solver iteratively cycles through the parameter axes (one block-diagonal penalty at a time), calculating the GCV score using the cyclic trace trick until global convergence is achieved. This proves far more physically robust for navigating non-differentiable or highly complex feature spaces.

### The Penalty Coordinates

Each coordinate $\alpha_j = \log_{10} \lambda_j$ of the descent, the `ap` a formula writes, bounded by `alpha_p_bounds`, is a penalty exponent read on the same scale as the weight the model keeps. Every basis is homogeneous in its own weight, $P_j(\lambda) = \lambda \, P_j(1)$, so the global penalty evaluated at a point $\alpha$ of the search space is the block-diagonal sum

$$P(\alpha) = \bigoplus_{j=1}^{J} P_j\left(10^{\alpha_j}\right)$$

assembled by the same basis code the P-WLS atom calls when the model is fitted. The coordinates the search converges to are therefore exactly the $\lambda_j$ the model stores, and refitting the model from those stored weights reproduces the GCV estimator. The same holds for the numerical ridge floor: the system GCV scores is $\Phi^\top\Phi + T P + 10^{-6} T I$, exactly the one the solver inverts (linear-system chapter), so the effective system is $(\Phi^\top\Phi / T) + P + 10^{-6} I$. Coordinates below $\alpha_j \approx -6$ are dominated by that floor: the default `alpha_p_bounds` $(-30, 6)$ keeps them reachable, but the GCV surface is flat there.

The alternative, rescaling the block the formula already carries, $10^{\alpha_j} P_j(\lambda_j^{\text{formula}})$, would optimize the product $\lambda_j^{\text{formula}} 10^{\alpha_j}$ while reporting only the second factor. The formula's own weight would then be applied twice during the search and once at the refit, so the two estimators would differ by a factor $\lambda_j^{\text{formula}}$ per block. It would also translate the reachable interval to $\lambda_j^{\text{formula}} \cdot [10^{\alpha_{\min}}, 10^{\alpha_{\max}}]$ instead of the intended $[10^{\alpha_{\min}}, 10^{\alpha_{\max}}]$: with the framework's default $\alpha^{\text{formula}} = -9$, the nine highest decades, the only ones that visibly remove degrees of freedom, would drop out of the search entirely.

One composite block escapes this reading. A tensor product scales the margin penalties it is built from, $P_{\otimes}(\lambda) = \lambda \sum_i I \otimes \cdots \otimes P_i(\lambda_i) \otimes \cdots \otimes I$, and the margin weights $\lambda_i$ stay at the values the formula gave them. The block is homogeneous in $\lambda$, so the descent still converges to a weight the refit reproduces exactly, but that weight is a multiplier on the surface rather than an absolute penalty: its magnitude carries the $\lambda_i$ with it.

---

## The Restricted Marginal Likelihood Alternative (REML)

GCV scores an asymptotic approximation of the leave-one-out error, and its curve in the smoothing parameters can have several minima when the errors are correlated or clustered: a wiggly fit that follows the noise then looks as good as a smooth one, and the parameters come out too small. Treating the coefficients as a Gaussian random effect with precision $T_{\text{eff}}P(\boldsymbol\rho)$, $P(\boldsymbol\rho)=\sum_p\lambda_pP_p$ with $\lambda_p=e^{\rho_p}$, and integrating them out by the Laplace approximation gives the restricted marginal likelihood {cite:p}`wood2011fast`

$$ \ell_R(\boldsymbol\rho)=-\tfrac12\log\big|\Phi^\top W\Phi+T_{\text{eff}}P(\boldsymbol\rho)\big|+\tfrac12\log|T_{\text{eff}}P(\boldsymbol\rho)|_+-\frac{T_{\text{eff}}-M_0}{2}\log\hat\phi,\qquad \hat\phi=\frac{D_p}{T_{\text{eff}}-M_0}, $$

with $\rho_p=\log\lambda_p$, $M_0$ the dimension of the null space of the penalty (the part of the fit that is not penalised), $|\cdot|_+$ the determinant on the penalised subspace and $D_p=\lVert\sqrt W(z-\Phi\hat\theta)\rVert^2+T_{\text{eff}}\hat\theta^\top P(\boldsymbol\rho)\hat\theta$ the penalised residual sum of squares (the working response $z$ is $y$ in the Gaussian case). When the scale is known (Poisson, binomial) the last term is $-D_p/(2\phi)$. Its gradient is analytic: $\partial_{\rho_p}\log|A|=\operatorname{tr}\big(A^{-1}\partial_{\rho_p}T_{\text{eff}}P(\boldsymbol\rho)\big)$ for $A=\Phi^\top W\Phi+T_{\text{eff}}P(\boldsymbol\rho)$, the same with the pseudo-inverse for $\log|P|_+$, and $\partial_{\rho_p}D_p=\hat\theta^\top\big(\partial_{\rho_p}T_{\text{eff}}P(\boldsymbol\rho)\big)\hat\theta$ because $\hat\theta$ minimises $D_p$. The sampling distribution of the REML estimate of a smoothing parameter is narrower than that of the GCV estimate and it is much less prone to undersmoothing {cite:p}`reiss2009smoothing`. For a non-Gaussian family the criterion is evaluated on the working system of the last reweighting pass, as for GCV.

## Matrix-Free GCV & Analytical Derivatives

When the combined feature dimension $D$ exceeds dense inversion budgets, the framework evaluates the effective degrees of freedom $\operatorname{Tr}(H)$ using **Hutchinson's stochastic trace estimator** with Rademacher probe vectors $z_r \in \{-1, +1\}^D$:

$$\operatorname{Tr}\!\left(A^{-1} G\right) \approx \frac{1}{R} \sum_{r=1}^R z_r^\top \left(A^{-1} G z_r\right)$$

where $A^{-1} (G z_r)$ is solved matrix-free via Preconditioned Conjugate Gradient, avoiding $\mathcal{O}(D^2)$ dense storage. Furthermore, for differentiable smoothing selection, **Wood's analytical GCV gradient** computes $\partial_{\rho_p} \operatorname{GCV}(\boldsymbol{\rho})$ directly from $\partial_{\rho_p} \hat{\theta} = -A^{-1} (\partial_{\rho_p} T_{\text{eff}} P(\boldsymbol{\rho})) \hat{\theta}$ and $\partial_{\rho_p} \operatorname{Tr}(H) = -\operatorname{Tr}(A^{-1} G A^{-1} \partial_{\rho_p} T_{\text{eff}} P(\boldsymbol{\rho}))$.
