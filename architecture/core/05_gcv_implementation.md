
# Auto-ML via Generalized Cross-Validation (GCV)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/05_gcv_theory.md)

Hyperparameter tuning via standard Grid Search is computationally expensive. While earlier versions relied heavily on "Multi-Start" Coordinate Descent for all parameters, the framework fully automated the continuous penalty optimization via `StaticTAM.auto_fit()`.

This engine leverages the `smart_solve_gcv` dispatcher, implementing Golub's trace trick {cite:p}`golub1979generalized` to find the optimal regularization parameter ($\lambda$) for each effect simultaneously, without requiring K-fold retraining.

---

## The Orchestrator and API Entry Point

The `auto_fit` method acts as the user-facing bridge. It aligns the data, establishes the loss bounds, and initializes the Multiple Smoothing Parameter (MSP) estimation algorithm.

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:start-after: "#: <auto_fit>"
:end-before: "#: </auto_fit>"
:caption: src/tam/model/additive.py (MSP-GCV Auto-Fitting Loop)
```

---

## VRAM Protection and Chunked Covariances



The `smart_solve_gcv` function operates as the exact algebraic engine. However, computing the trace of the inverse matrix requires the explicit instantiation of the global covariance matrix $\Phi^T \Phi$. 

To strictly prevent Out-Of-Memory (OOM) faults during the iterative search, the engine queries the hardware oracle to enforce an advanced **3-Tiered Memory Waterfall** specifically designed for Coordinate Descent:

1. **The Caching Fast-Path (< 30% Limit):** If caching the unpenalized covariance matrices (`cov_X` and `cov_XY`) for *all* spatial groups requires less than $30\%$ of available VRAM, the orchestrator caches the entire system strictly on the GPU. This completely bypasses the massive $\mathcal{O}(G \times T \times D^2)$ tensor product during the search phase.
2. **The Iterative Accumulation (< 40% Limit):** If a global GPU cache is impossible, but a single data group requires less than $40\%$ of VRAM, the engine falls back to looping over the groups on the GPU natively, accumulating covariances iteratively and aggressively freeing memory per step.
3. **Extreme CPU Offloading (> 40% Fallback):** If evaluating a single group exceeds the $40\%$ VRAM safety threshold (frequently triggered by massive Neural or Tensor interactions), the orchestrator intercepts the impending GPU crash and gracefully forces `eval_device = torch.device('cpu')`, offloading the entire cyclic trace resolution to the Host system's RAM.

Because the system allows different regularization weights for different topological bases (e.g., one $\lambda$ for Splines, another for Fourier), the global penalty matrix $P$ must be updated dynamically during the optimization loop.

The dispatcher pre-calculates the index boundaries (`start`, `end`) for every individual effect in the formula and maps them to a `blocks` array. During each evaluation of the GCV objective function, it injects the actively tested $\lambda$ values strictly into their corresponding diagonal blocks, ensuring zero cross-contamination between distinct structural penalties.

---

## Discrete Coordinate Descent Optimization



To optimize the multiple parameters efficiently across the block-diagonal structure, `smart_solve_gcv` deploys a memory-safe Discrete Coordinate Descent loop {cite:p}`wright2015coordinate`.

1.  **Initialization:** It extracts the starting $\log_{10}(\lambda)$ values from the `effects_list`.
2.  **Iterative Search:** For a maximum of 15 cycles, it iterates through each individual effect.
3.  **Local Perturbation:** It tests localized candidates (`original - step_size` and `original + step_size`) while holding all other effects' penalties constant.
4.  **Scoring:** The system recalculates the exact cyclic trace and GCV score for each perturbation. If a candidate improves the global GCV score, it becomes the new baseline.

The loop naturally terminates early if a full cycle completes without any parameter achieving a lower GCV score. Once optimal parameters are found, it triggers a final dense inversion using `solve_linear_system` to map the optimal coefficients exactly to the target distribution.

```{literalinclude} ../../../../src/tam/model/_dispatcher_gcv.py
:language: python
:pyobject: smart_solve_gcv
:caption: src/tam/model/_dispatcher_gcv.py (Memory-Safe MSP GCV Solver)
```
