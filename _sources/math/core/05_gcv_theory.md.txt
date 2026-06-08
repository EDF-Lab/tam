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

**1. The Smoothing Matrix ($S$):**
The bracketed term above maps the true targets $Y$ to the predicted targets $\hat{Y}$. In statistical literature, this is known as the "Hat Matrix" or the Smoothing Matrix $S$ {cite:p}`wood2004stable`:

$$S = \Phi \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1} \Phi^T \Lambda^T \Lambda$$

**2. The Residual Sum of Squares ($RSS$):**
The $RSS$ is simply the weighted squared empirical error of the fitted model:

$$RSS = || \Lambda (Y - \hat{Y}) ||^2 = || \Lambda (I - S) Y ||^2$$

---

## The Mathematical Demonstration: From LOO to GCV


How does evaluating the smoothing matrix $S$ replace data splitting? 

In classic Leave-One-Out Cross-Validation (LOOCV), the model is trained $n$ times, leaving out one observation $i$ each time to predict its value. Thanks to the Sherman-Morrison-Woodbury matrix identity, it is a proven theorem that LOOCV can be calculated exactly without any retraining, using only the diagonal elements of the smoothing matrix ($S_{ii}$) {cite:p}`golub1979generalized`:

$$LOOCV = \frac{1}{T} \sum_{i=1}^T \left( \frac{y_i - \hat{y}_i}{1 - S_{ii}} \right)^2$$

However, Golub, Heath, and Wahba {cite:p}`golub1979generalized` proved that LOOCV is not rotation-invariant and can become highly unstable if the design matrix is unbalanced (e.g., if a specific $S_{ii}$ approaches 1). 

To fix this, **Generalized Cross-Validation** replaces the individual diagonal elements $S_{ii}$ with their global average. The sum of the diagonal elements of a matrix is its trace, so the average is $\frac{1}{T} Tr(S)$. Substituting this average into the LOOCV formula yields the GCV objective {cite:p}`golub1979generalized`:

$$GCV(\lambda) = \frac{\frac{1}{T} \sum_{i=1}^T (y_i - \hat{y}_i)^2}{\left(1 - \frac{1}{T} Tr(S)\right)^2} = \frac{\frac{1}{T} RSS}{\left(1 - \frac{1}{T} Tr(S)\right)^2}$$

### Autocorrelation Inflation ($\gamma$)
The framework evaluates a strictly modified version of this equation:

$$GCV(\lambda) = \frac{\frac{1}{T} RSS}{\left(1 - \frac{\gamma}{T} Tr(S)\right)^2}$$

Where $\gamma$ is an inflation multiplier (typically $\gamma = 1.4$). Standard GCV assumes independent and identically distributed (i.i.d.) noise. In chaotic time-series environments, errors are frequently autocorrelated, causing uncorrected GCV to severely overfit the noise. The $\gamma$ parameter artificially inflates the penalty of the trace, forcing the solver to favor smoother, more robust functional mappings as theoretically justified by Kim and Gu {cite:p}`kim2004smoothing`.

---

## The Cyclic Trace Trick

A direct application of the GCV formula requires computing the trace of the smoothing matrix $S$. In the Dual space, $S$ is an $T \times T$ matrix. Computing its trace requires $\mathcal{O}(T^3)$ operations, which would cause an immediate bottleneck. 

To completely decouple the hyperparameter optimization time from the number of rows in the dataset, the framework exploits the cyclic property of the trace operator ($Tr(AB) = Tr(BA)$).

By cyclically permuting the matrices, the framework evaluates the trace purely in the Primal feature space ($D \times D$):

$$Tr(S) = Tr\left(\Phi^T \Lambda^T \Lambda \Phi \left( \Phi^T \Lambda^T \Lambda \Phi + T P \right)^{-1}\right)$$

This allows the framework to compute the exact degrees of freedom instantaneously, even on datasets with millions of observations, simply by multiplying the weighted feature covariance matrix by the inverted regularized system.

---

## Multiple Smoothing Parameters (MSP) via Coordinate Descent

Because TAM concatenates highly heterogeneous mathematical bases (e.g., mixing Splines with Fourier series and Random Features), applying a single global penalty $\lambda$ is statistically invalid. The framework must optimize a different regularization parameter for *each* architectural block simultaneously {cite:p}`wood2004stable`.



While classical statistics recommends using a Newton-Raphson method based on exact analytical derivatives to solve Multiple Smoothing Parameters, calculating these higher-order derivative tensors requires massive memory allocations that frequently crash GPU VRAM.

**The Engineering Compromise:**
To safely scale, TAM deploys a Multi-Start Discrete Coordinate Descent algorithm {cite:p}`wright2015coordinate`. Rather than computing the dense Hessian of the GCV landscape, the solver iteratively cycles through the parameter axes (one block-diagonal penalty at a time), calculating the GCV score using the cyclic trace trick until global convergence is achieved. This proves far more physically robust for navigating non-differentiable or highly complex feature spaces.

---

## Future Roadmap: Matrix-Free GCV

Currently, the exact algebraic trace evaluation requires the dense inversion of the $D \times D$ system matrix ($\Phi^T \Lambda^T \Lambda \Phi + T P$). In PyTorch hardware engineering, this imposes a hard physical ceiling. If the combined feature dimension exceeds $D = 7500$, the dense inversion will trigger an Out-Of-Memory (OOM) error. For extreme topologies exceeding this limit, the framework is mathematically forced to abandon exact GCV and fall back to a pure Matrix-Free grid search using the Conjugate Gradient solver.



To unify these pipelines in the future, the framework aims to evaluate the GCV score using **Hutchinson's stochastic trace estimator**. By drawing independent random vectors with Rademacher entries, Hutchinson's method approximates the trace of the inverse matrix using only matrix-vector products. This would entirely eliminate the need for the dense exact matrix inversion, allowing the continuous GCV algebraic solver to scale infinitely alongside the sparse Matrix-Free Conjugate Gradient solver.

