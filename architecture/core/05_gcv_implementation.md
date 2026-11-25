
# Auto-ML via Generalized Cross-Validation (GCV)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/05_gcv_theory.md)

Hyperparameter tuning via standard Grid Search is computationally expensive. While earlier versions relied heavily on "Multi-Start" Coordinate Descent for all parameters, the framework fully automated the continuous penalty optimization via `StaticTAM.auto_fit()`.

This engine leverages the `smart_solve_gcv` dispatcher, implementing Golub's trace trick {cite:p}`golub1979generalized` to find the optimal regularization parameter ($\lambda$) for each effect simultaneously (for each margin of a `te()`: an effect exposes `penalty_coordinates()` and `set_penalty_coordinates()`, one coordinate by default and one per margin for a tensor product), without requiring K-fold retraining.

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

The dispatcher pre-calculates the index boundaries (`start`, `end`) for every individual effect in the formula and maps them to a `spans` array. During each evaluation of the GCV objective function, `sync_penalty_blocks` injects the actively tested $\lambda$ values strictly into their corresponding diagonal blocks, ensuring zero cross-contamination between distinct structural penalties.

---

## Discrete Coordinate Descent Optimization



To optimize the multiple parameters efficiently across the block-diagonal structure, `smart_solve_gcv` deploys a memory-safe Discrete Coordinate Descent loop {cite:p}`wright2015coordinate`.

1.  **Initialization:** It extracts the starting $\log_{10}(\lambda)$ values from the `effects_list`.
2.  **Iterative Search:** For a maximum of 15 cycles, it iterates through each individual effect.
3.  **Local Perturbation:** It tests localized candidates (`original - step_size` and `original + step_size`) while holding all other effects' penalties constant.
4.  **Scoring:** The system recalculates the exact cyclic trace and GCV score for each perturbation. If a candidate improves the global GCV score, it becomes the new baseline.

The loop naturally terminates early if a full cycle completes without any parameter achieving a lower GCV score. Once optimal parameters are found, it triggers a final dense inversion using `solve_linear_system` to map the optimal coefficients exactly to the target distribution.

**Penalty parity.** Step 3 does not rescale a block built once from the formula's own `lambda_p`. `sync_penalty_blocks` assigns $\lambda_p = 10^{\alpha}$ to the effect and asks the effect to rebuild its block, so the matrix being scored is the one `_build_penalty_matrix()` reassembles from the selected weights: on the default `loss="l2"` path, `auto_fit()` followed by `fit()` returns the same estimator. That is the contract AutoTAM depends on when `refit_on_full_train=True` puts a champion back on the full training window. At most two blocks are rebuilt per candidate, the coordinate under test, plus the previous one when its own search settled on a value other than the last one tried, and the trial weight is written on the effect only for the duration of that rebuild, so a search that raises leaves the caller's effects as the formula declared them.

**Non-Gaussian losses.** When the model has a reweighting strategy (`poisson`, `gamma`, `binomial`, `negative_binomial`, `tweedie`), `auto_fit()` calls `_auto_fit_reweighted`: it fits the model, takes the working response and Fisher weights at the fitted linear predictor, and passes them to `smart_solve_gcv(..., sample_weights=w)`, which scales $\Phi$ and $z$ by $\sqrt w$ before forming the covariances (`Y_sq` and the trace follow from them). The fit and the search alternate until the selected $\log_{10}\lambda$ moves by less than 0.25, with at most four passes, and a last reweighted fit at the selected parameters sets `coefficients_` and `dispersion_`. `grid_search_fit()` follows the same rule for its candidates: each is fitted by the reweighted loop and scored on the response scale (inverse link applied); the returned model keeps the loss and carries the winning score in `validation_rmse_`.

**REML.** `auto_fit(criterion='reml')` calls `smart_solve_reml` (`_reml.py`) instead. `PenaltyBasis` writes the penalty of the effects as `S0 + sum_c lambda_c Q_c`, one `Q_c` per smoothing coordinate, by evaluating each effect's penalty at weights 1 and 2 (every penalty of the library is linear in its weights, so this covers `m_pen`, the margins of a `te()`, `mode='per_level'` and the second weight of a thin-plate spline), and the subspace where the penalty acts (its null space, of dimension `null_dim`, is fixed by the structure and not by the current weights). `PenaltyBasis` keeps one block per effect and evaluates `log|S|_+`, its trace terms and, for the Hessian, the trace of the products of two penalty shares, block by block (a scalar block, a commuting Kronecker group and a dense group each have their own closed form; a one-dimensional diagonal penalty needs no factorisation; from `D = 32` the inverse diagonal of the gradient comes from a triangular solve instead of a full inverse). `reml_objective` returns minus the criterion summed over the groups and its analytic gradient in `log(lambda)`; with `compute_hess=True` it also fills `hess_out` with the analytic Hessian (`_reml_hessian`: the columns `Z_c = A^-1[:, I_c] T_c` of all coordinates side by side, reduced by block with `index_add_`, from the Cholesky factor and the inverse the gradient already needs) and with `efs_out` the numerator and denominator of the fixed-point step.

`smart_solve_reml` has one search. The start points (a six-point grid of common weights, the formula's own weights, the shifted trace-ratio seeds `-4, -2, 0, 2` times `ln 10` around the scale of each block, and the previous optimum when `init_rho=` is given) are scored together in one batched Cholesky in `float64` (on the device of the design when it is a GPU; above 256 MB the candidates are scored one by one), and the best one starts `_solve_reml_efs_newton`: at most three fixed-point steps (`_efs_step`, clamped to `[-5, 5]`, accepted when the criterion decreases) and at most 30 safeguarded Newton iterations on the coordinates that are neither at a bound with an outward gradient nor flat (`|H_cc| < 1e-6 max|diag H|` and `|g_c| < 1e-5`; recomputed at every iteration, so a coordinate re-enters when its gradient grows). The Newton step flips the eigenvalues of the Hessian of the active block to their absolute values (floored at `1e-6` of the largest), is capped at 5 in the max norm, and is shortened by at most 12 halvings on value-only evaluations, with the early exits of the line search described in the theory; one steepest-descent step capped at 2 (at most 8 halvings) is tried when none is accepted. The search stops on the exact Newton decrement (`< 0.05` REML unit, gradient below `1e-3`, Hessian convex down to `-1e-3` of its largest eigenvalue): the reasons are `newton_ok`, `newton_bounds` and `newton_step_limit`. The search runs inside the bounds of `alpha_p_bounds` (`alpha_p_list` narrows them; with a single value there is nothing to search and the fit is the plain `fit()`).

```{literalinclude} ../../../../src/tam/model/_reml.py
:language: python
:start-after: "#: <reml_newton_search>"
:end-before: "#: </reml_newton_search>"
:caption: src/tam/model/_reml.py (fixed-point warm start and safeguarded Newton search)
```

The design is built once and reused across the outer reweighting passes (`design_matrix=`, `basis=` and `init_rho=` carry it, the penalty basis and the previous optimum from one pass to the next); a design that is a `_StructuredDesign` (one-hot and `te(smooth, cat)` blocks) or a row-wise stateless design too large for memory gives the covariances without ever forming the dense `T x D` matrix (`reml_working_system`). The search is a dense Cholesky of the `D x D` system, with `D <= 7500`. `smart_solve_reml(..., stop_info=dict)` receives `reason`, `n_iter` and the counts and times of the search (fixed-point steps, Newton steps, Hessian evaluations, seconds in the Hessian, in the search and in the start screen); `auto_fit` keeps the reason and the iterations as `reml_stop_reason_` and `reml_iterations_` (one entry per reweighting pass for a GLM loss). For a GLM loss `_auto_fit_reweighted` runs at most four passes: a reweighted fit at the current weights, then `smart_solve_reml` on the working response and the unit-mean weights, with the previous optimum as a start point and the known scale `1 / mean(w)` for Poisson, binomial and negative binomial; the loop ends when `log10(lambda)` moves by less than 0.25, and a last reweighted fit sets `coefficients_` and `dispersion_`. The fitted model keeps `smoothing_criterion_` and `reml_score_`. For a GLM loss, the dispersion is refitted together with the smoothing weights in the outer loop, and the reweighted fit itself (`_reweighting.py`) accepts a step on the unit-mean penalised deviance, starts from the previous coefficients (`initial_theta`) and, only then, accelerates the iteration with a safeguarded Anderson window of 3 (a cold start keeps the plain iteration); see [the REML section of the theory](../../math/core/05_gcv_theory.md).

```{literalinclude} ../../../../src/tam/model/_reml.py
:language: python
:start-after: "#: <reml_hessian>"
:end-before: "#: </reml_hessian>"
:caption: src/tam/model/_reml.py (analytic Hessian of the criterion)
```

```{literalinclude} ../../../../src/tam/model/_reml.py
:language: python
:pyobject: reml_objective
:caption: src/tam/model/_reml.py (REML criterion and gradient)
```

```{literalinclude} ../../../../src/tam/model/_dispatcher_gcv.py
:language: python
:pyobject: smart_solve_gcv
:caption: src/tam/model/_dispatcher_gcv.py (Memory-Safe MSP GCV Solver)
```
