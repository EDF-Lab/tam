
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Source repository: https://github.com/EDF-Lab/tam

> The first official open-source release of the TAM framework under this structure is **[1.2.3]**. (Versions 1.1.1 - 1.2.2 correspond to internal development milestones and were not publicly released.)

> **[0.0.6]** corresponds to the legacy `weakl` package available on PyPI.

---

## [Unreleased]

### Added

- **Evolutionary Engine Upgrade:** Implemented a Mutant-UCB Multi-Armed Bandit for island selection and a Quality-Diversity (QD) Champion Archive using Topological Edit Distance (TED) inside `DragTAM`.
- **Online Sequential Inference:** Added `AutoTAM.predict_online()` to dynamically re-weight expert ensembles using `MLpol` during chronological backtesting (standard `predict()` retains frozen fit-time weights).
- **Deterministic O(N) Feature Profiler:** Introduced `FeatureProfiler` to diagnose non-linearity, basis families, and required signal capacity prior to the search, eliminating trial-and-error capacity grids.
- **Robust Graph Tracking:** Added exact term identity resolution (`term_members`, `term_signature`, `decomposition_names`) to properly track and prune tensor products in the `KnowledgeGraph`.
- **Documentation:** Expanded Mirror Architecture documentation for AutoTAM probabilistic selection (`math/meta/11`, `architecture/meta/11`) and feature profiling (`math/meta/12`, `architecture/meta/12`).
- **Mandatory Terms and Variables:** AutoTAM formulas may fix terms that every candidate keeps, written next to the pipeline macro (`load ~ s(temp, k=10) + AutoPipe(temp, humidity)`), and `AutoTAM(mandatory_variables=[...])` forces variables into every model. Mandatory terms may use dataset variables absent from `AutoPipe(...)`: only those exact terms are used for them, with their free parameters tuned. Mandatory terms are immutable during the evolutionary search and exempt from ablation.
- **Term Canonicalisation:** `canonicalize_term`, `term_subsumes` and `terms_are_equivalent` (`tam.model.autotam.parser`) treat `s(X, k=10, basis='cubic')` and `s(X, basis='cubic', k=10)` as the same term, so duplicates are dropped.

### Changed

- **Dynamic Apex Ensembles:** The Apex ensemble now aggregates the *entire* pool of dynamic experts (Kalman/Adaptive) rather than truncating to a Top-N, excluding only those whose validation error exceeds `apex_quality_ratio` (default 1.5x the best).
- **Universal Tensor Products:** Removed `CrossIsland`; all islands can now generate tensor products (`te`, max 2 per formula). Bounded continuous-continuous interactions to ~25 columns to prevent VRAM explosions.
- **Targeted Deep Interactions:** Deep Islands (Neural, RBF, Tree) now dynamically condition on 1 to 5 categorical partners (`others=`), leaving 25% of terms unpartnered to preserve marginal effects.
- **Expanded Search Grids:** Widened hyperparameter grids across all bases (e.g., Trees up to 25 estimators/50 leaves, Splines up to $k=50$). Excluded `cos` activations to prevent regime-shift instability. Added `LinearTreeEffect` (`lt`) and targeted `PID` terms.
- **Strongest-First Pruning:** `KnowledgeGraph` now resolves collinear terms by keeping the one with the highest explained variance first (default prune threshold: 0.5%).
- **Exploration Safeguards:** `DragTAM` explicitly reserves 25% of each generation for fresh spawns (`fresh_fraction`). Island UCB enforces 3 minimum pulls and 15% uniform exploration (`ucb_epsilon`) to prevent premature convergence.
- **Unbiased Island Selection:** Disabled the Knowledge Graph's basis-family prior by default to prevent search monocultures and maintain OPERA ensemble diversity.
- **OOD Target Clipping:** Expert predictions are strictly clipped to observed training bounds (plus a 15% margin) before aggregation to protect frozen-weight averages from extreme extrapolations.
- **Vectorized Windowing:** Refactored grouped rolling means/EWMAs in `FeatureEngineer` to use native positional Pandas operations, fixing index misalignment.
- **Report Generator Security & Metrics:** Added strict `run_id` validations and path-confining. PDP panels now report scale-free `Var(h_j) / Var(prediction)` driver scores. (Matplotlib is now optional).
- **Mandatory Terms Keyword:** `AutoTAM(mandatory_terms=...)` now raises `TypeError`: write mandatory terms in the formula instead.
- **Fail-Fast Validation:** `DataManager.prepare` rejects invalid mandatory terms before the search: syntax errors, duplicates under canonical equivalence, unknown features, more than `MAX_ACTIVE_EFFECTS_PER_FEATURE = 2` effects on a feature, more than `MAX_TENSOR_TERMS = 2` tensor products. Test data must contain the external mandatory features.
- **Explicit Splits:** fit/dev/val splits given with overlapping or non-unique indexes receive disjoint indexes, so every evaluation uses the rows of its own split.
- **Formula Splitting:** `split_args_respecting_parentheses` raises `ValueError` on a leading, trailing or doubled delimiter and on unbalanced parentheses (it used to accept them silently), and takes a `delimiter` argument.
- **Visible Expert Failures:** an expert whose cross-validation evaluation fails is logged (`logging` warning with the formula and the fold) instead of being scored infinite silently.

### Fixed

- **GCV Penalty Rescaling:** `smart_solve_gcv` rescaled blocks that already carried the formula's own $\lambda_p$, so the search scored the product of the two while `auto_fit()` stored only the tested weight: every later `fit()` rebuilt the penalty from that weight alone and returned a different model, and with the default `ap = -9` the search never reached the nine highest decades of its range. Each trial block is now rebuilt by the effect at $\lambda_p = 10^\alpha$, exactly as `fit()` assembles it. Tensor products keep their marginal weights inside the block, so their coordinate scales that surface rather than setting an absolute penalty.
- **LinearTree Regularization:** Added missing property setters to `LinearTreeEffect` so GCV weights correctly propagate to sub-blocks.
- **Sparsity-Adaptive Tree Fix:** Fixed `empirical_counts` dimension reduction in `TreeEffect` (now summing the batch axis instead of groups), repairing the `sparsity_alpha > 0` penalty generation.
- **Column Alignment:** Fixed label shifting in `decompose_prediction` to correctly map tensors and multi-bases to PyTorch columns. Previously, labels were misaligned by positional assignment from deduplicated feature lists, causing component contributions to be attributed to the wrong terms (note: the global fitted prediction was unaffected by this bug).
- **Knowledge Graph Pruning Fix:** Fixed term contribution lookups so redundant terms are correctly identified, rewarded, and pruned.
- **Simulation Memory Leak:** `EnsembleSelector` now aggressively releases AdaptiveTAM simulation tensors (`m_ref.simulation_data_ = None`) after scoring, resolving 60-73 GB RAM peaks during wide grid searches.
- **Time-Series Integrity:** Fixed Kalman/Adaptive prediction backtest alignment, patched a `groupby` crash caused by Daylight Saving Time (DST) duplicates, and prevented `__dummy_date__` overflows on massive inputs.
- **Grid-Search Experts:** the grid-search experts of `ExpertExpander` were never fitted (the `grid_search_fit(cv_folds=...)` call did not match the engine's `grid_search_fit(data_train, data_val, grid_search_config)`); they are now fitted on the first cross-validation fold.
- **Parenthesis Mismatch:** combining mandatory terms and mandatory variables no longer produces formulas with unbalanced parentheses.

---
## [1.3.0] - 2026-09-06

### Added
- **Statistics layer** (`tam.model.statistics`) : a modular "how" beside the structural spectrum "what", built on the single P-WLS atom (`BaseTAM._solve_pwls_step`), so the default `loss="l2"` path stays bit-identical to ordinary least squares.
  - **Reweighting & estimation** (`estimation/`): GLM families (Gamma, Poisson, Binomial), asymmetric expectiles, and robust M-estimators (Huber, Student-t) via `StaticTAM(loss=...)`, driven by an IRLS schedule over the atom.
  - **Distributional** location-scale fits via a dict formula/loss (`{"mu": ..., "sigma": ...}`), with automatic Normal/Student-t tail selection, `predict_quantiles`, `cdf`, `anomaly_score` and `crps`.
  - **Mixture** of TAM regressions (`mixture_components=K`) fitted by EM whose M-step is the responsibility-weighted atom.
  - **Gaussian copula** (`GaussianCopulaTAM`) binding several distributional margins.
  - **Conformal & ACI** (`statistics.risk`): distribution-free CQR intervals, conformal p-values, Mondrian (stratified) calibration (`ConformalDistributionalTAM`) and streaming Adaptive Conformal Inference, on the static `SafetyTAM` engine.
  - **EVT & epistemic uncertainty**: Generalized-Pareto tail scoring (`GeneralizedParetoTail`, `fit_gpd_tail`) and Bayesian posterior parameter uncertainty (`posterior_prediction`).

## [1.2.6] - 2026-07-17

### Added
- Added SPDX license identifiers (`LGPL-3.0-or-later`) and standardized file headers with proper authorship and copyright attribution across the entire codebase (2026-07-01).

### Changed
- **Major Data Upgrade:** Replaced the legacy `dataset_national.csv` with the new **FORCE** (French Open Research Catalogue of Energy) dataset. The library now defaults to this high-performance, harmonized dataset for baseline benchmarking and examples.

### Fixed
- **Critical Tensor Alignment Bug (`group_col`):** Fixed an order-sensitivity vulnerability in the tensor stacking and DataFrame reassembly layers. The framework now enforces strict chronological sorting (via `date_col`) before building 3D tensors, preventing penalty matrix corruption on non-sequential data. Additionally, it now safely maps chronological PyTorch predictions back to shuffled Pandas indices, guaranteeing row-to-row integrity.

## [1.2.5] - 2026-06-24

### Added
- **Topological Split Strategies (`TreeEffect`):** Formalized the `split_strategy` parameter. Users can now explicitly toggle between `'uniform'` (creating mathematically orthogonal, shift-invariant Cartesian grids) and `'quantile'` (applying the empirical Probability Integral Transform to create density-adaptive partitions that perfectly balance sample distributions across all leaves).

### Changed
- **Empirical Sparsity-Adaptive Penalty (Anisotropic Ridge):** Upgraded the structural penalty of the Random Forest (`t(...)`) and Linear Tree (`lt(...)`) modules. By setting `sp_alpha > 0`, the initialization pass now accurately records the empirical data density of each terminal leaf ($C_i$). 
- **Drift & Singularity Prevention:** Starved or empty edge-boundary leaves now receive geometrically massive penalties. This guarantees global matrix rank, eliminates the catastrophic test drift associated with hard Cartesian grids, and theoretically resolves the OLS singularities traditionally found in Model-Based Recursive Partitioning (MOB).

## [1.2.4] - 2026-06-22

### Added
- **API Standardization:** Introduced explicit `fit()` and `predict()` methods across all meta-models (`AdaptiveTAM`, `OperaTAM`, `KalmanTAM`). This establishes a unified, scikit-learn-like operational workflow (train on historical data, freeze state, predict out-of-sample) regardless of the underlying algorithm.
- **AdaptiveTAM:** Added `fit()` and `predict()` methods for production deployment. The `fit()` method efficiently extracts and solves the linear system strictly for the final available training window. The `predict()` method then applies this frozen state (`last_state_dict_`) to new data in $O(1)$ time with strict safety clipping, ensuring instant, deterministic inference without target leakage.
- **KalmanTAM:** Added `fit()` and `predict()` methods alongside end-of-training state extraction (`last_state_dict_` and `scale_dict_`). This allows users to project the finalized Kalman drift weights forward as a stable, static rule on new data, with the internal normalization math handled automatically.
- **OperaTAM:** Added `fit()` and `predict()` methods to transition from continuous dynamic simulation to frozen-weight inference. `fit()` runs the historical simulation, while `predict()` cleanly extracts and applies the final expert aggregation weights to new out-of-sample data.

### Fixed
- **StaticTAM:** Removed the `target_col` requirement from the required features check in `decompose_prediction`. This resolves a critical blocker for operational inference pipelines where the target variable is naturally unavailable.
- **KalmanTAM:** Patched `_prepare_kalman_features` to securely bypass target column extraction during out-of-sample inference, preventing crashes when the target variable is absent.

## [1.2.3] - 2026-06-08
> The DOI was generated via Zenodo on release : https://doi.org/10.5281/zenodo.20543272.

### ✨ Added (New Models & Core Features)

* **Universal Extrapolation Wrapper**: Introduced native Out-Of-Distribution (OOD) extrapolation for all base effects via the `extrapolate` parameter. It safely bounds the feature map to the $[-1, 1]^F$ hypercube and utilizes multidimensional directional derivatives (stepping strictly backward into the safe zone) for OOD inputs. Supported modes include `continue` (native topology), `constant` (plateau/clamping), `linear` (first-order Taylor expansion), and `saturation` (smooth asymptotic clamping).
* **Linear Tree (`lt(...)`)**: Added a native effect that generates piecewise linear models. It utilizes a dedicated `LinearTreeEffect` class to encapsulate a standard `TreeEffect` (acting as the local intercept/level) crossed with a `TensorProductEffect` (acting as the local linear slope). This provides a single, cohesive model for varying-coefficient trees, seamlessly handling multi-dimensional spatial data without requiring formula macro workarounds.
* **Flat N-ary Histograms (`TreeEffect`)**: Added the `max_leaves` parameter to bypass binary depth and force flat 1D N-ary splits. This includes an Anti-Starvation Protocol (evenly spaced bins) for single trees to guarantee full matrix rank and prevent over-complete matrix singularities in piecewise regressions.
* **Academic Reproductions**: Added official benchmark scripts reproducing foundational load forecasting architectures using the TAM framework:
    * `2011_pierrot_goude.py`: Benchmarks native grouping vs. PyGAM manual loops using local B-splines.
    * `2025_doumeche_et_al.py`: Benchmarks the transition from local splines to global Fourier bases with Sobolev regularization.
* **Theory, Cheatsheets and Documentation**: Added comprehensive TAM documentation:
    * [THEORY](THEORY.md) as the central mathematical reference and `cheatsheet.py` to showcase the entire TAM spectrum (Static bases, Neural Networks, Physics operators) wrapped in Meta-Learners.
    * **The StaticTAM**
        * **The Primal Model:** [Theory](math/core/01_primal_model.md) | [Architecture](architecture/core/01_additive_api.md)
        * **Tensorization & Data:** [Theory](math/core/02_tensorization.md) | [Architecture](architecture/core/02_data_pipeline.md)
        * **Linear Systems:** [Theory](math/core/03_linear_system.md) | [Architecture](architecture/core/03_math_dispatcher.md)
        * **Complexity & Hardware:** [Theory](math/core/04_complexity.md) | [Architecture](architecture/core/04_hardware_memory.md)
        * **GCV & Auto-ML:** [Theory](math/core/05_gcv_theory.md) | [Architecture](architecture/core/05_gcv_implementation.md)
        * **The base effects** : [Linear](math/spectrum/LINEAR.md) / [Fourier](math/spectrum/FOURIER.md) / [Spline](math/spectrum/SPLINES.md) / [Chebyshev](math/spectrum/CHEBYSHEV.md) / [Categorical](math/spectrum/CATEGORICAL.md) / [Interaction](math/spectrum/CROSS_TENSOR.md) / [RBF](math/spectrum/RBF.md) / [Neural](math/spectrum/NEURAL.md) / [Tree](math/spectrum/TREE.md) / [Linear Tree](math/spectrum/LINEAR_TREE.md) / [Wavelet](math/spectrum/WAVELETS.md) / [PID](math/spectrum/PID.md) / [Physics](math/spectrum/PHYSICS_PIKL.md)
    * **The AdaptiveTAM**  
        * **Adaptive Online Learning:** [Theory](math/meta/01_adaptive_online.md) | [Architecture](architecture/meta/01_adaptive_code.md)
    * **The OperaTAM**  
        * **Expert Aggregation (Opera):** [Theory](math/meta/05_opera_aggregation.md) | [Architecture](architecture/meta/05_opera_gpu.md)

### 🚀 Changed (Major Refactoring & Optimization)

* **OPERA Dual API Support (`OperaTAM`)**: Added a standard array-based initialization (`target_col="y"`, `expert_cols=["E1", "E2"]`) alongside the existing R-like formula API (`formula="y ~ l(E1) + l(E2)"`), allowing for simpler dynamic aggregation.
* **Architectural Shape Normalization**: Overhauled `build_feature_map` across `TreeEffect`, `NeuralEffect`, and `RBFEffect`. Added a dynamic dimensional router to natively resolve tensor broadcasting ambiguities across 1D (OOD wrappers), 2D (Kronecker `te(...)` interactions), and 3D+ (Primal Solver Factory) inputs.
* **Formula Parser Robustness**: Upgraded `parse_formula_to_terms` to explicitly track and uniquely index nested sub-arguments using positional indices (`i`, `j`). Resolves parameter collision and overwriting issues when parsing nested interaction terms containing identical effect types.
* **Memory Probe Safeguards (Dummy Pass)**: Improved robustness of the VRAM footprint estimation during the dummy pass across all base effects. This prevents premature initialization of randomized partition geometries, NEPT weights, or RBF centers during the framework's memory estimation phase.
* **Categorical Effect Automation**: The Categorical effect (`c(...)`) now automatically parses the dataset to count `n_cat` if the parameter is omitted by the user.
* **MLOps Dashboard (`plotting_dashboard`)**: 
    * Enhanced chronological forecast plots with a `forecast_smoothing` parameter (supports rolling averages and date-based resampling).
    * Implemented dynamic evaluation metrics for the Test Set Vulnerability heatmap (automatically scaling for RMSE, MAE, MAPE, etc.).
    * Unified color mapping across all subplots ensuring consistent model identification using Matplotlib's `tab10` colormap.

---

## [Internal] 1.2.2 - 2026-03-26 (Not publicly released)

### ✨ Added (New Models & Core Features)

* **Evolutionary Orchestrator (`AutoTAM`)**: A multi-fidelity AutoML engine for automated GAM discovery. It utilizes a Hub-and-Spoke evolutionary architecture, strict topological sanitization, and bi-level optimization (GPU MSP-GCV) to solve the combinatorial explosion of adaptive models, ultimately deploying orthogonal experts into a Dual OPERA arena.

* **OPERA (`OperaTAM`)**: A new expert aggregation meta-learner featuring a fast GPU implementation natively optimized via `@torch.jit.script`.
* **Kalman Filter (`KalmanTAM`)**: Dynamically tracks coefficient drift over time via a Fast Dynamic Extended Kalman Filter (EKF), highly optimized using the Woodbury matrix identity (reducing inversion complexity to $\mathcal{O}(T_{block}^3)$) and compiled with `TorchScript`.
* **DeepGAM (`NeuralTAM`)**: A new Deep-GAM hybrid model (Additive + Deep Learning) implementing Group-wise Orthogonal Backfitting.
* **Tree Effect (`TreeEffect`)**: Added the Tree / Random Forest effect (`t(...)`) designed for GPU, based on Oblivious Random Trees and Random Binning Features approximation.
* **Hardware Manager (`HardwareManager`)**: Centralized hardware abstraction layer to dynamically manage the capabilities of different compute backends.
* **`_dispatcher.py` (Mathematical Solver Dispatcher)**: Created an intelligent routing layer between statistical modeling abstractions and PyTorch linear algebra engines. It dynamically routes resolution to either a chunked direct solver or a Matrix-Free Sparse Conjugate Gradient (CG) solver based on topological complexity and available VRAM.
* **`_memory.py` (Hardware Memory Management and Estimation)**: Completely isolated low-level hardware interactions into a dedicated module. It estimates the byte footprint of massive matrices and calculates safe algorithmic chunk sizes.

### 🚀 Changed (Major Refactoring & Optimization)

* **Neural Effect Improvements (`NeuralEffect`)**: Added support for multiple hidden layers to project variables into higher dimensions.
* **Native GPU Acceleration**: Complete migration of intensive CPU to GPU calculation for Splines (`s(...)`), Wavelets (`w(...)`), and RBF (`rbf(...)`) effects, improving performance of design matrix construction.
* **Memory Management (Safeguards & Smart Chunking)**: 
    * Overhauled memory safety to prevent and correct CUDA Out of Memory bugs via a smart chunking system.
    * Implemented a memory safeguard for CPU / group-chunking by independent series.
    * Established strict dynamic RAM & VRAM safety margins to guarantee the stability of large matrix inversion operations.

---

## [Internal] 1.2.1 - 2025-12-18 (Not publicly released)

This version represents a complete architectural overhaul, introducing advanced functional bases (Spectrum), Conformal Prediction, and a full benchmark suite.

### ✨ Added (New Models & Core Features)

* **Auto-ML (GCV):** Added `StaticTAM.auto_fit()` using **Generalized Cross Validation (GCV)** for automatic global regularization parameter selection, eliminating the need for a validation set.
* **Safety Module (Conformal Prediction):** Added `SafetyTAM` implementing **Split Conformal** (static) and **Adaptive Conformal Inference (ACI)** (dynamic) to guarantee valid confidence intervals under distribution shift.
* **Hierarchical Reconciliation:** Added `HierarchicalTAM` to solve global constraints (e.g., National = Sum of Regions) via joint optimization on the primal system.
* **Model Introspection:** Added `StaticTAM.summary()` to display the model's structure, complexity, and regularization parameters.
* **Core Effects Library (`spectrum`):** Implemented a complete modular library of advanced functional bases:
    * `ChebyshevEffect` (`p(...)`): Global polynomials for stable trend approximation.
    * `WaveletEffect` (`w(...)`): Ricker wavelets for local anomaly and transient feature detection.
    * `NeuralEffect` (`n(...)`): Neural projection for high-dimensional non-linearity.
    * `RBFEffect` (`rbf(...)`): Support for both **Gaussian** and **Matérn** (physics-informed) kernels.
    * `TensorProductEffect` (`te(...)`): **Multivariate interactions** (Kronecker product) for surface modeling.
    * `UniversalPhysicsEffect` (`phys(...)`): **PIKL** (Physics-Informed Kernel Learning) for constraining models with differential operators (ODEs/PDEs).

### 🚀 Changed (Major Refactoring & Optimization)

* **Math Engine (Primal Solver):** Formally validated the exact **Primal Ridge Solver** utilizing block-diagonal covariance accumulation. Corrected performance tracking to accurately reflect the framework's time complexity of $\mathcal{O}(G \times T \times D^2 + G \times D^3)$, ensuring isolated mathematical resolution per group $G$.
* **Effect Architecture:** Refactored the core around `BaseEffect`, establishing the `List[BaseEffect]` as the standard configuration.
* **Modularization:** Monolithic `_effects.py` was entirely split into the `spectrum` package, improving modularity.
* **Normalization Domain:** Changed global feature normalization from the Fourier-centric $[-\pi, \pi]$ to the strictly orthogonal **$[-1, 1]$** domain in `_data.py`. Basis functions now apply internal scaling (e.g., Fourier rescales to $[-\pi, \pi]$).
* **Decomposition Robustness:** Implemented **collision detection** in `_math.py` to automatically prefix feature effects (e.g., `l_time`, `s_time`) when multiple bases share the same input variable.

### 🐛 Fixed (Critical)

* **Recursive Parsing:** Implemented an architectural fix in `parse_formula_to_terms` to correctly **identify and preserve string tokens** (like `ga_te` or `grid_k`) during the recursive parsing of `te(...)` terms.
* **Syntax Stability:** Converted all docstrings containing LaTeX math commands to **raw strings** (`r"""..."""`) to eliminate Python `SyntaxWarning`s.

---

## [Internal] 1.1.1 - 2025-11-21 (Not publicly released)

This version introduced the Formula API and the first object-oriented refactoring.

### ⚠️ Breaking Changes
- Removed legacy dictionary-based API (`m_orders`, `s_orders`, `alpha_list`)
- Introduced formula-based API as the primary interface

### ✨ Added

* **Formula-based API (`model/additive.py`):** Implemented a new, intuitive R-like formula API (e.g., `Load ~ s(temp, k=10) + l(day_type)`) as the new standard for model initialization.
* **Spline Effects (`model/_effects.py`):** Added `SplineEffect` (P-splines) as a new core effect type, available via `s(...)`.
* **Formula Parser (`common/utils.py`):** Added a `parse_formula_to_terms` function to support the new API.
* **`StaticTAM` & `AdaptiveTAM`:** Implemented the full object-oriented API (`.fit()`, `.predict()`) and the online error correction model.
* **Multi-Start Grid Search:** The `grid_search_fit` method now uses a **Multi-Start Coordinate Descent** strategy (Conservative, Median, Aggressive) to avoid local minima.
* **`diagnostics` Module:** Added a module for model analysis, including t-tests and feature importance visualization.

### ⚙️ Changed

* **Legacy API Removed:** Removed the old `m_orders`, `s_orders`, `alpha_list` dictionary-based configuration from `v0.0.6`.
* **Package Structure:** The codebase was refactored into a modular package structure (`common`, `model`).
* **Internal Math:** Math functions (`_math.py`) were cleaned of all effect-specific logic and made robust to 2D/3D tensor inputs.
* **Hardcoded Names Removed:** Removed dependencies on specific column names (`tod`, `timestamp`, `Load`).

---

## [0.0.6] - 2025-05-27

### Added
* Initial project setup based on the original `weakl` v0.0.6 package.

[1.3.0]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.0
[1.2.6]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.6
[1.2.5]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.5
[1.2.4]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.4
[1.2.3]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.3
[0.0.6]: https://pypi.org/project/weakl/0.0.6/