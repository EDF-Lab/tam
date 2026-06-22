# The Additive API and Object-Oriented Architecture

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/01_primal_model.md)
  
This chapter details the overarching software engineering pipeline of the TAM framework. It explains how high-level user formulas are parsed, routed, and translated into robust linear algebra operations across the primary orchestration scripts (`utils.py`, `_base.py`, `_factory.py`, and `additive.py`), structurally grounded in the Primal resolution theory {cite:p}`doumeche2025forecasting`. 

To avoid redundancy, low-level data padding, hardware dispatching, and specific effect implementations are delegated to their respective documentation files.

## Global Configurations (`utils.py`)

Before any model is instantiated, the framework establishes a unified global configuration via `utils.py`. 

To maintain mathematical determinism in the exact Primal inversion and guarantee convergence {cite:p}`doumeche2025forecasting`, the framework dynamically forces PyTorch and NumPy to utilize 64-bit precision (`float64`) where the hardware supports it. This initialization acts as the absolute source of truth for the device (`TORCH_DEVICE`) and the precision target (`NUMPY_DTYPE`) across the entire architecture.

```{literalinclude} ../../../../src/tam/common/utils.py
:language: python
:start-after: "#: <config>"
:end-before: "#: </config>"
```

## The Foundation: `_base.py`

The `_base.py` script acts as the structural foundation for all models in the framework. It defines the abstract class `BaseTAM`, which orchestrates the standardized control flow.

To ensure a consistent API without duplicating boilerplate tensor operations across advanced Meta-Learners, `BaseTAM` utilizes the Object-Oriented Template Method Pattern.

  * It explicitly manages state definitions (`coefficients_`, `norm_params_`).
  * It standardizes the temporal alignment and handling of missing groups. *(For exact tensor padding logic, refer to the [Data Pipeline](02_data_pipeline.md)).*
  * It defines the continuous optimization problem logically, forcing child classes to implement the specific construction of the design matrix $\Phi$ and the penalty matrix $P$ required to stabilize the regularized normal equations {cite:p}`hoerl1970ridge`.

```{literalinclude} ../../../../src/tam/model/_base.py
:language: python
:start-after: "#: <class_def>"
:end-before: "#: </class_def>"
```

## The Factory Orchestration (`_factory.py` & `_base_effects.py`)

Instead of hardcoding basis functions into the solver, `StaticTAM` relies on an explicit Dependency Injection architecture. It expects components that conform strictly to the `BaseEffect` interface (`_base_effects.py`).

To bridge the user's R-style formula string to these concrete interface objects, the framework uses the `create_effects_from_parsed_terms` factory. This function is engineered specifically to support architectural Grid Searches. By passing a `token_values` dictionary, the factory dynamically substitutes string variables with concrete hyperparameters (e.g., swapping `'gk_la'` for `10`), allowing the solver to rebuild massive architectures on the fly without re-parsing the original regex structure.

*(For the specific mapping of every individual effect, refer to [The Spectral Dictionary](06_the_spectrum_api.md)).*

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <create_effects>"
:end-before: "#: <parse_linear>"
```

## The Core Solver (`additive.py`)

The `StaticTAM` class is the primary engine of the framework. It inherits from `BaseTAM` and acts as the grand orchestrator. Once the mathematical blocks are assembled via the Factory, `StaticTAM` delegates the actual matrix inversions to the underlying [Math Dispatcher](03_math_dispatcher.md).

### Initialization and Dependency Routing

When initialized, `StaticTAM` parses the formula. A critical engineering choice is the detection of Grid Search tokens. If the parsed parameters contain unresolved strings, it flags the model as a template (`is_grid_search_template_ = True`), intentionally halting the instantiation of the effects to defer to the Multi-Start Coordinate Descent engine.

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:start-after: "#: <init_additive>"
:end-before: "#: </init_additive>"
```

### Component Decomposition

Because the Primal space concatenates independent topological blocks, the framework can mathematically isolate the contribution of each effect. The `decompose_prediction` method vectors this operation, multiplying the partitioned design matrix by its corresponding isolated coefficients to return a structural breakdown of the forecast.

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:start-after: "#: <decompose_pred>"
:end-before: "#: </decompose_pred>"
```

## Hyperparameter Routing: Continuous vs. Discrete

To safely scale to Gigadata without exhausting computational time, `StaticTAM` divides hyperparameter tuning into two distinct structural methods.

### The Continuous Algebraic Solver (GCV)

For continuous structural penalties (the $\lambda$ regularization weights), iterative searching is mathematically obsolete. The `auto_fit` method routes the training data to the Generalized Cross-Validation (GCV) dispatcher. This computes the optimal Multiple Smoothing Parameters analytically via the cyclic trace trick {cite:p}`golub1979generalized`.

*(For the implementation details and block-diagonal routing logic of this solver, see the [GCV Implementation Guide](05_gcv_implementation.md)).*

### The Discrete Architectural Solver (Coordinate Descent)

While regularization is continuous, topological choices-such as the number of knots in a Spline or the maximum depth of a Tree-are strictly discrete. The `grid_search_fit` method employs a Multi-Start Coordinate Descent algorithm to resolve these non-differentiable tokens. It tests structural mutations by iteratively cycling through the parameter axes, executing rapid trial evaluations to find the optimal global architecture.

```{literalinclude} ../../../../src/tam/model/additive.py
:language: python
:start-after: "#: <grid_search_logic>"
:end-before: "#: </grid_search_logic>"
```

## Separation of Concerns: Simulation vs. Inference

To guarantee safety in operational production pipelines, `AdaptiveTAM` strictly separates historical learning from out-of-sample inference.

**Architectural Choice (The `fit` / `predict` Split):**
* **`fit(data)`:** Runs the full sliding-window `predict_online()` simulation. At the end of the simulation, it extracts the final historical residuals and trains a global, frozen `StaticTAM` model (`self.static_residual_model_`) to act as the permanent correction rule.
* **`predict(data)`:** A purely deterministic, read-only method. It applies the base model and the frozen static residual model to new data. By completely bypassing the sliding-window simulation during inference, it guarantees blazing-fast, $O(1)$ execution time and zero target leakage.