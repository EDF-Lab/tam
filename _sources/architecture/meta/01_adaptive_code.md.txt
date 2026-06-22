
# Online Error Correction (AdaptiveTAM)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/01_adaptive_online.md)

Standard Generalized Additive Models (GAMs) assume the underlying data generating process is globally stationary. However, real-world time series suffer from structural breaks, concept drift, and loss of exchangeability. While the underlying mathematics were explored in the theoretical formulations, `AdaptiveTAM` provides the production-grade hardware implementation.

It utilizes a Sliding Window tensorization approach to update the covariance matrices continuously, allowing the coefficients to smoothly adapt to new regimes without retraining the global model from scratch. This chapter details how this theory is translated into massively parallel PyTorch operations, strictly avoiding restrictive Python `for` loops.

-----

## Design Pattern: Instance Composition

The software architecture directly reflects the two-stage mathematical formulation of Online WeaKL. The `AdaptiveTAM` class does not inherit from `StaticTAM`; instead, it utilizes a **Composition** pattern. 

**Architectural Choice:** By using composition, the framework prevents namespace collision and guarantees mathematical isolation between the physics-based macroscopic model and the reactive microscopic model. It *owns* and orchestrates two distinct instances:
* `self.base_model_`: The expert instance, permanently frozen after its initial training.
* `self.adaptive_model_`: A blank, template instance that will be cloned and dynamically retrained on each sliding window to correct the base model's residuals.

```{literalinclude} ../../../../src/tam/model/adaptative.py
:language: python
:start-after: "#: <init_adaptive>"
:end-before: "#: </init_adaptive>"
:caption: src/tam/model/adaptative.py (Composition Pattern for the Two-Stage Model)
```

## Advanced PyTorch Indexing (4D Vectorization)

One of the major engineering challenges of online learning is extracting the training windows. Sequentially extracting 10,000 localized windows using a Python `for` loop would completely stall the CUDA pipeline due to excessive CPU-GPU kernel launch latency.

To solve this, the `_transform_data_adaptive` function (located in `_data.py`) exploits PyTorch's advanced indexing and broadcasting. 


**Architectural Choices & Frequency Agnosticism:**
The code deliberately transitions from a rigid "Calendar Day" perspective to a **Frequency-Agnostic Group Coordinate System**. By using mathematical abstract variables like `periods` and `steps_per_period`, it perfectly isolates panel data entities regardless of whether the native sampling rate is monthly, daily, or half-hourly.

1. **Anchor Calculation:** The code identifies all valid starting indices (`start_indices`) by walking backward through time, strictly ensuring no incomplete windows are processed.
2. **Causal Offset Generation (Preventing Target Leakage):** It generates two constant tensors: `predict_offsets` (the indices of the prediction window) and `train_offsets` (the relative indices of the learning window). Crucially, to prevent Target Leakage in multi-step forecasting, the `train_offsets` are dynamically shifted backward by `-(horizon_steps - 1)`. This physically truncates the "illegal future" from the training buffer before the tensor is ever evaluated on the GPU.
3. **Vectorized Extraction:** By adding the column matrix of start indices (`start_indices.view(-1, 1)`) to the offset vectors, PyTorch instantly broadcasts a complete 2D index matrix. The framework then extracts the entirety of the causally safe historical data in a single, contiguous memory call: `x_group[train_indices]`.

```{literalinclude} ../../../../src/tam/model/_data.py
:language: python
:start-after: "#: <transform_adaptive>"
:end-before: "#: </transform_adaptive>"
:caption: src/tam/model/_data.py (Vectorized Sliding Window Extraction Without For Loops)
```

## Dynamic Target and Preparation

The `prepare_simulation` method is executed before launching the online loop. Its role is to massively vectorize the dataset and shift the target space. 

**Architectural Choice:** It calculates the macroscopic predictions (`data_pred`) and subtracts them from the true target to isolate the residuals ($\epsilon_t = Y_t - \hat{Y}_{base}$) upfront. Executing this in a single block across the entire validation history prevents an immense computational bottleneck that would occur if the base model were evaluated repeatedly inside the sequential sliding window loop.

```{literalinclude} ../../../../src/tam/model/adaptative.py
:language: python
:start-after: "#: <prepare_sim>"
:end-before: "#: </prepare_sim>"
:caption: src/tam/model/adaptative.py (Batch Residual Calculation and Tensorization)
```

## Batch Execution and Numerical Safety (Clipping)

In the `simulation()` method, calculations are not executed step-by-step. The solver must simultaneously resolve the exact local empirical risk minimization problem for all sliding windows across all groups.

**Architectural Choice 1: Dimensional Flattening:**
The generated tensors naturally possess 4 dimensions `(n_groups, n_windows, num_samples, features)`. The standard Core mathematical solvers (`solve_linear_system`) are built exclusively for 3D batches `(batch, samples, features)`. Instead of rewriting the core linear algebra module, `simulation()` flattens the first two dimensions: `total_items = n_groups * n_windows`. This elegantly maps the massive localized window problem into a standard, independent batch structure, saturating the GPU cores perfectly.

**Architectural Choice 2: OOM Resilience:**
Because `total_items` can easily exceed tens of thousands of matrices, the method implements the dynamic `hw.handle_oom` fallback mechanism. If the GPU VRAM exhausts, it catches the `torch.OutOfMemoryError` and halves the `safe_batch_size` iteratively, ensuring the simulation completes gracefully even on constrained hardware.

**Architectural Choice 3: Algorithmic Safeguard (Target Clipping):**
Adaptive models can diverge wildly if they train on a window containing purely anomalous values (e.g., a sensor failure). To guarantee safety in production, the implementation applies strict bounded clipping. The final adapted prediction is physically bounded by the maximum and minimum historical values of the observed base residuals (`data_bm[self.target_col_].max()`), preventing the corrector from extrapolating unrealistic deviations during unexpected exogenous shocks.

```{literalinclude} ../../../../src/tam/model/adaptative.py
:language: python
:start-after: "#: <run_sim>"
:end-before: "#: </run_sim>"
:caption: src/tam/model/adaptative.py (Batch Linear Resolution, OOM Handling, and Safety Clipping)
```

## Coordinate Descent Search Algorithm

The `grid_search_fit()` method implements the "Multi-Start" hyperparameter solver described in the theory. It evaluates pairs (e.g., `training_window_periods`, default $\lambda$) sequentially, moving along a single axis at a time until local convergence is reached.


**Architectural Choice:** Because the global objective function evaluated across overlapping sliding windows creates highly non-convex search topologies, standard analytical Generalized Cross-Validation (GCV) breaks down. Coordinate Descent provides robust navigation through these complex spaces. To systematically avoid poor local minima, the algorithm leverages three distinct initialization strategies (Conservative, Median, Aggressive) before executing the cyclic axis search.

```{literalinclude} ../../../../src/tam/model/adaptative.py
:language: python
:start-after: "#: <grid_search_adaptive>"
:end-before: "#: </grid_search_adaptive>"
:caption: src/tam/model/adaptative.py (Coordinate Descent Algorithm for Hyperparameters)
```

## Separation of Concerns: Simulation vs. Inference

To guarantee speed and safety in operational production pipelines, `AdaptiveTAM` strictly separates the continuous historical simulation from out-of-sample inference.

**Architectural Choice (The $O(1)$ Inference Optimization):**
* **`fit(data)`:** Instead of running the massive sliding-window simulation over the entire dataset, `fit()` leverages the `_save_final_state()` method. This function slices *only* the final available training window from the 4D tensor, solves the exact Primal linear system for that single step, and freezes the optimal coefficients (`last_state_dict_`) along with historical safety clipping bounds.
* **`predict(data)`:** A purely deterministic, read-only method. It builds the design matrix $\Phi$ for the new data and multiplies it directly against the frozen coefficients. By completely bypassing sliding-window tensorization and system resolution during inference, it guarantees blazing-fast $O(1)$ execution time and absolute protection against target leakage.