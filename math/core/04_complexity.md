# Computational Complexity ($\mathcal{O}(D^3+T D^2)$ vs $\mathcal{O}(T^3)$)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/04_hardware_memory.md)
  * **Related topic:** [Linear Systems & Solvers](03_linear_system.md)
  
This chapter outlines the exact computational bounds of the framework, demonstrating how TAM formally decouples model complexity from the sheer volume of empirical data.

---

## The Dual Bottleneck ($\mathcal{O}(T^3)$)

Traditional Kernel Methods, popularized by Vapnik's Support Vector Machines (SVM) {cite:p}`cortes1995support, vapnik2013nature`, rely on the Dual formulation and the 'Kernel Trick' to handle high-dimensional mappings implicitly. While mathematically elegant for finding maximum-margin hyperplanes, this requires explicitly computing and inverting an $T \times T$ Gram matrix. This leads to an intractable $\mathcal{O}(T^3)$ computational bottleneck, which the TAM framework completely bypasses via its Primal exact resolution.

This creates a crippling theoretical bottleneck. The memory complexity scales at $\mathcal{O}(T^2)$ and the computational time complexity scales at $\mathcal{O}(T^3)$. For modern industrial applications involving millions of high-frequency time series observations (Gigadata), calculating this full dense space is a physical impossibility.

---

## The Primal Shift to $\mathcal{O}(T D^2 + D^3)$

By strictly projecting all functional bases explicitly into a finite Reproducing Kernel Hilbert Space (RKHS), the framework executes a "Primal Shift". The core mathematical resolution relies on solving the regularized normal equations:

$$\hat{\theta}_g = \left( \Phi_g^\top \Lambda_g^\top \Lambda_g \Phi_g + T \cdot P \right)^{-1} \Phi_g^\top \Lambda_g^\top \Lambda_g Y_g$$

Because the framework operates strictly in this Primal space, the dimensionality of the system is dictated exclusively by the number of synthesized features $D$ (the column width of $\Phi$), completely bypassing the $\mathcal{O}(T^3)$ sample limit. As theoretically justified by Kim and Gu {cite:p}`kim2004smoothing`, projecting the optimization problem onto a much smaller approximating space ($D \ll T$) allows for massive scalability without sacrificing asymptotic statistical efficiency.

The computational cost of forming the covariance matrix is $\mathcal{O}(T D^2)$, and the direct inversion via Cholesky or LU decomposition costs $\mathcal{O}(D^3)$.

---

## Matrix-Free Bypassing of the $D$ Limit

While the Primal Shift solves the $T$ bottleneck, building highly expressive models with Neural Tangent Kernels or deep Tensor Products can cause the feature dimension $D$ to become massive. Materializing the $\Phi^\top \Phi$ matrix requires $\mathcal{O}(D^2)$ contiguous memory, which can easily exceed GPU VRAM limits.

To break this final barrier, the framework utilizes the Matrix-Free Sparse Conjugate Gradient solver, as validated by Wu et al. for solving high-dimensional randomized features efficiently {cite:p}`wu2016revisiting`. 

By evaluating the system dynamically through a mathematical closure $A_g v = \Phi_g^\top \Lambda_g^\top \Lambda_g (\Phi_g v) + T \cdot P v$, the matrix multiplications collapse sequentially. Since the inner product $(\Phi_g v)$ is computed first, the algorithm never explicitly forms the dense $D \times D$ covariance matrix. This shifts the per-iteration time complexity down to purely $\mathcal{O}(T D)$ and bounds the memory allocation linearly, allowing the exact resolution of effectively limitless dimensions strictly within the GPU caches.

---

## Hardware Group-Chunking and Exact Determinism

When the system cannot fit entirely in memory even at $\mathcal{O}(T D)$, the data must be chunked. However, statistical stability demands exact resolution without the noise of iterative stochastic approximations. To guarantee perfect mathematical determinism and stability when estimating Multiple Smoothing Parameters face-to-face with Gigadata {cite:p}`wood2004stable`, the framework entirely forbids temporal slicing.

Instead, the data is partitioned horizontally via **Group-Chunking**. 

By treating the problem as a 3D tensor ($\mathcal{X} \in \mathbb{R}^{G \times T \times D}$), the system dispatches mathematical blocks by independent groups (e.g., individual smart meters or distinct spatial regions). This confines the maximum spatial footprint to $\mathcal{O}(G_{chunk} \times T \times D)$ while ensuring the entire history of any specific dynamic is parsed in a single atomic mathematical pass.

> **Architecture Context:** To see how the theoretical Group-Chunking bounds are calculated to strictly prevent Out-Of-Memory (OOM) errors, see [Hardware Memory & Anti-OOM Systems](../../architecture/core/04_hardware_memory.md).

---

## Future Algorithmic Upgrades

 While the coupling of Group-Chunking (for hardware acceleration) and Adaptive Jitter (for matrix stability) currently offers an optimal compromise to solve massive RKHS systems, a future iteration of the framework's mathematical core aims to implement Block-QR or rank-1 Cholesky update/downdate algorithms. By mathematically rotating the existing decomposed matrix as new sliding-window data arrives, the online $\mathcal{O}(W D^2)$ time complexity will collapse to strictly $\mathcal{O}(D^2)$ per step. This would allow the framework to process data in chunks while entirely avoiding the explicit calculation of the $\Phi^\top \Lambda^\top \Lambda \Phi$ cross-product, offering the flawless numerical stability demanded by {cite:p}`wood2004stable` on pathologically ill-conditioned datasets without needing artificial jitter.

