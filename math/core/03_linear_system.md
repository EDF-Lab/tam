
# Linear Algebra: Direct vs. Iterative Solvers

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/03_math_dispatcher.md)
  * **Related topic:** [Computational Complexity](04_complexity.md)
  
This chapter explains the mathematical resolution of the continuous optimization problem formulated in the Primal space. Once the data is projected into the finite-dimensional Reproducing Kernel Hilbert Space (RKHS), finding the optimal function reduces to solving a large-scale linear system.

---

## The Regularized Normal Equations

As established, the primal objective is to minimize the empirical risk penalized by a structural prior. Setting the gradient of this objective function to zero yields the regularized normal equations:

$$(\Phi_g^\top\Lambda_g^\top\Lambda_g\Phi_g + T \cdot P)\hat{\theta}_g = \Phi_g^\top\Lambda_g^\top\Lambda_g Y_g$$

Where:
* $\Phi_g^\top\Lambda_g^\top\Lambda_g\Phi_g$ is the weighted Gram matrix (acting as the Hessian of the empirical risk) for group $g$, where $\Lambda_g$ is the diagonal matrix of sample weight square roots.
* $\Phi_g^\top\Lambda_g^\top\Lambda_g Y_g$ is the weighted feature-target cross-correlation vector for group $g$.
* $P$ is the block-diagonal structural penalty matrix. In the context of Regularization Networks, this matrix acts as the explicit regularization operator $P$, mapped to a dot product space to mathematically enforce the smoothness and flatness of the estimated function {cite:p}`smola2004tutorial`.
* $T$ is the number of time steps per group, acting as the exact scaling factor for the Tikhonov penalty term {cite:p}`tikhonov1943stability, hastie2009elements, vapnik2013nature`.

> **Architecture Context:** To see how these equations are converted into batched tensor multiplications over the temporal or group dimensions without triggering memory faults, refer to the PyTorch implementation in [PyTorch Math Dispatcher](../../architecture/core/03_math_dispatcher.md).

---

## The Direct Resolution and Memory Workspaces

When the total number of features $D$ (the combined dimension of all additive effects) is bounded, the system is solved analytically via direct matrix inversion algorithms (such as LU or Cholesky decomposition). This yields the exact, globally optimal coefficients $\hat{\theta}$ in a single algebraic step.

While mapping from the Dual space to the Primal space drastically reduces the computational complexity of the matrix multiplication to $\mathcal{O}(G \times T \times D^2)$, materializing the dense Gram matrix $\Phi_g^\top\Lambda_g^\top\Lambda_g\Phi_g$ creates a strict per-group memory footprint of $\mathcal{O}(D^2)$.

When building highly expressive models utilizing Deep Learning interactions (Neural Explicit Primal Tensorization) or vast 2D Tensor Products, $D$ can easily exceed 7,500. Attempting a direct inversion at this scale will immediately exhaust the VRAM of modern hardware accelerators (GPUs), resulting in catastrophic Out-Of-Memory (OOM) crashes.

---

## The Matrix-Free Iterative Alternative

To bypass the physical memory limit of the direct solver, the framework implements a dynamic dispatching system that routes massive topologies to a **Sparse Conjugate Gradient (CG)** solver.



As theoretically justified by Wu et al. {cite:p}`wu2016revisiting`, attempting to materialize dense matrices for high-dimensional randomized features is highly inefficient. Instead, the Conjugate Gradient method allows the framework to solve the linear system without ever constructing the $D \times D$ covariance matrix in memory.

The algorithm relies entirely on evaluating matrix-vector products. The exact closure $Av$ dynamically computes the product on the fly:

$$A_g v = \Phi_g^\top \Lambda_g^\top \Lambda_g (\Phi_g v) + T \cdot P v$$

By computing the inner product $(\Phi_g v)$ first, the framework collapses the dimensionality before multiplying by $\Phi_g^\top$. This strictly bounds the active memory allocation to $\mathcal{O}(T \times D)$ per group, allowing the system to resolve models with millions of features purely within the fast caches of the GPU.

---

## The Convergence Crisis and the Preconditioner Roadmap

While the Matrix-Free approach is mathematically elegant, utilizing a standard Conjugate Gradient on highly heterogeneous dictionaries is theoretically dangerous. As established by LeCun et al. regarding the conditioning of Hessian matrices {cite:p}`lecun1998efficient`, and confirmed in classical numerical analysis {cite:p}`golub1996matrix`, the convergence speed of the CG algorithm is strictly bound by the condition number of the matrix, $\kappa(A)$. If the matrix is ill-conditioned-an exceptionally common scenario when mixing continuous splines with discrete categorical bases-the convergence rate collapses.

**The Adaptive Jitter:**
To counter this vulnerability and guarantee strict determinism (Float64), the framework currently injects an adaptive Jitter directly into the evaluation of the $Av$ product:

$$\delta I = 10^{-6} \times T \times I$$

This acts as a microscopic, baseline Ridge regularization {cite:p}`hoerl1970ridge`. While this "numerical anvil" forcefully guarantees the positive-definiteness of the matrix and stabilizes the Krylov subspace, it fundamentally deviates from the purity of classic preconditioning methods.

**Update Roadmap (PCG):**
The mathematical implementation must evolve towards a Preconditioned Conjugate Gradient (PCG). The addition of a diagonal preconditioner (Jacobi)-which involves estimating and dividing by the diagonal of the implicit matrix-is an operation with a negligible memory cost of $\mathcal{O}(D)$. This theoretical upgrade will allow the removal of the massive Jitter while drastically dividing the number of iterations required to solve extreme systems ($D > 7500$), aligning the framework perfectly with the mathematical state-of-the-art for iterative solvers.