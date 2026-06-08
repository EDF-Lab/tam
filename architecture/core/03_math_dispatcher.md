# PyTorch Math Dispatcher & Matrix-Free Solvers

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/03_linear_system.md)
  
This chapter explores how the linear algebra theory described in [Linear Algebra: Direct vs. Iterative Solvers](../../math/core/03_linear_system.md) is translated into high-performance, batched PyTorch tensor operations, seamlessly bridging statistical abstractions with low-level hardware constraints.

## Covariance Accumulation

The `_math.py` module handles the foundational matrix multiplications. The `_compute_weighted_covariances` function is responsible for building the Left-Hand Side (`cov_X`) and Right-Hand Side (`cov_XY`) of the regularized normal equations {cite:p}`doumeche2025forecasting`.

By utilizing `torch.mT` (batched matrix transpose) and the `@` operator, PyTorch computes these covariance matrices across the temporal or group batch dimension simultaneously. The script dynamically handles target weighting via a loss-weighting matrix `loss_L_star_L` prior to the dot product, ensuring that multidimensional outputs are appropriately scaled.

```{literalinclude} ../../../../src/tam/model/_math.py
:language: python
:start-after: "#: <weighted_cov>"
:end-before: "#: </weighted_cov>"
```

## Direct Solvers and the Jitter Application

When the topological complexity (the feature dimension $D$) is small enough to fit safely in VRAM, the orchestrator routes the problem to the `solve_linear_system` function, which utilizes exact direct inversion via `torch.linalg.solve` {cite:p}`golub1996matrix`.

To guarantee that the matrix remains strictly positive-definite across diverse precision levels (Float32 vs Float64) and highly correlated feature spaces, the framework injects an adaptive Jitter. 

This numerical anvil adds a microscopic trace ($\delta I = 10^{-6} \times T \times I$) to the diagonal, acting as a baseline Ridge penalty {cite:p}`hoerl1970ridge`. This immediately stabilizes the condition number of the matrix before passing it to the underlying linear algebra backend (e.g., LAPACK for CPU, or cuSOLVER/MAGMA for GPU).

```{literalinclude} ../../../../src/tam/model/_math.py
:language: python
:pyobject: solve_linear_system
```

## The Matrix-Free Conjugate Gradient

If the user designs a massive architecture (e.g., crossing a Random Forest with 10,000 leaves against a Fourier series), the resulting dense covariance matrix $\Phi^T \Phi$ would trigger a catastrophic Out-Of-Memory (OOM) error.

To bypass this physical barrier, the framework utilizes `solve_sparse_cg`, a Matrix-Free Conjugate Gradient solver. 



Instead of computing and allocating the massive $D \times D$ system, the solver operates entirely within a Krylov subspace {cite:p}`golub1996matrix`. It requires only a closure function `compute_Av(v)` that evaluates the matrix-vector product $(A \cdot v)$. This allows the framework to iteratively discover the optimal coefficients $\hat{\theta}$ without ever materializing the dense matrix in memory.

```{literalinclude} ../../../../src/tam/model/_math.py
:language: python
:pyobject: solve_sparse_cg
```

## The Dispatcher Routing Logic

The decision to route between the exact chunked Direct Solver and the iterative Matrix-Free Conjugate Gradient is dynamically evaluated by the `smart_solve` function in `_dispatcher.py`.



Before attempting to allocate the global covariance matrix, the orchestrator cross-references the topological complexity against the physical hardware limits using `can_fit_dense_matrix`. 

1. **Safe VRAM (Direct Solver):** If direct inversion is deemed safe, it triggers `_run_chunked_direct_solver`. This function executes predictive Group Chunking by allocating up to 80% of available memory to calculate a `safe_group_batch`.
2. **Exhaustion Risk (CG Fallback):** If the feature dimension $D$ exceeds the safety threshold, `smart_solve` automatically routes to `_run_sparse_cg_solver` to prevent VRAM exhaustion. This function defines the `compute_Av(v)` closure on-the-fly, safely computing the matrix-vector products recursively across manageable data chunks.

If an unexpected OOM exception is still encountered during extreme scaling, both solvers are wrapped in a robust `try/except` loop. This loop communicates with the hardware manager to iteratively reduce the `safe_group_batch` size, clearing the cache until the computation proceeds stably.

```{literalinclude} ../../../../src/tam/model/_dispatcher.py
:language: python
:start-after: "#: <smart_solve_router>"
:end-before: "#: </smart_solve_router>"
```
