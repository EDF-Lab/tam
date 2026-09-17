# AutoTAM Feature Profiling: Engineering Implementation

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/12_autotam_feature_profiling.md)
* **Related AutoTAM architecture:** [Orchestrator Code](08_auto_orchestrator_code.md)

> Mirror-Architecture companion outlining the engineering implementation for `math/meta/12_autotam_feature_profiling.md`.
> This document details the "how": where the profiling stage runs, its internal data flow, and the strict invariants it must preserve. It does not re-derive the mathematical estimators.

---

## 1. Execution Context

The `FeatureProfiler` (`autotam/feature_profiler.py`) is a pre-search stage owned by the `EffectSelector`. It executes exactly once inside `build_search_space`, strictly before the evolutionary engine begins:

```text
DataProfiler -> FeatureEngineer -> EffectSelector[ FeatureProfiler ] -> DragTAM -> ExpertExpander -> EnsembleSelector

```

This stage is fully deterministic and stateless between runs. Its verdicts are written directly into the search space, meaning no other downstream modules need to invoke it.

---

## 2. Data Flow

1. **Initialization:** The `profile(df, target_col, features, group_col)` method selects numeric candidates possessing more than two distinct values, drops non-finite rows, and stacks `[y | X]` into a single unified array.
2. **Within-Group Transform:** The process subtracts per-group means using `np.add.at` on integer category codes. This acts as a vectorized scatter-add rather than a slow Python `groupby` loop, completing the transform in one vectorized pass yielding shape `[N, 1+P]`.
3. **Scaling:** Columns are scaled by their standard deviation; zero-variance columns are masked out and bypassed.
4. **Ridge Solve:** The pipeline executes one ridge solve on the `[P, P]` Gram matrix utilizing `np.linalg.solve`, with a fallback to `lstsq` for singular systems. The ridge strength is scaled by the mean diagonal to make it unit-free.
5. **Feature Diagnosis:** For each feature, the partial residual is computed (the full residual plus that feature's own fitted contribution). This is followed by a quantile-binned profile via `np.digitize` and `np.bincount` (strictly avoiding Python loops over rows), an FFT over the ~50 bin means, and the final computation of statistics.

*Return Format:* The stage returns a dictionary shaped `{feature: {linear, family, m, k, eta2, r2, gap, tv_ratio, reason}}`. If a feature cannot be diagnosed, it is **omitted**, allowing `EffectSelector` to safely fall back to prior behaviour without failing the pipeline.

---

## 3. Integration Invariants

* **Continuous Topologies Only:** `EffectSelector` consults a profiling verdict only when its `_analyze_topology` method returns `continuous`. Discrete and sparse features are naturally routed to categorical/tree bases where a basis-capacity verdict is meaningless.
* **Single Capacity Value:** Capacity grids stay fixed as `{"m": [value]}`; they must never become a list of multiple candidates. The profiler replaces a poor default but must not reintroduce a hyperparameter grid, as doing so would bloat the combinatorial search space and starve the structural search.
* **Legacy `max_k` Bypass:** The legacy capacity cap (`min(20, N//10)`) is bypassed for profiled capacity, because applying it to a measured `k=40` would silently discard the diagnosis. Profiled knots are bounded by sample size instead: `max(10, min(50, N//100))`, whose ceiling is the resolution of the reference `te(s(toy, k=50), c(.))` interaction.
* **Linear Locks:** Features with a linear verdict hard-lock `eligible_effects` to `['l']` (plus `pid` on the most recent autoregressive lag), which simultaneously frees a covariate-lock slot.
* **State Persistence:** The dictionary mapping `search_space[col]["profile"]` carries the verdict forward for the KnowledgeGraph prior.

---

## 4. KnowledgeGraph Prior Integration

The `set_basis_prior(feature, effect, strength)` function stores `feature -> (effect, strength)`. The `suggest_effect_for_feature` method consults it *before* the epsilon-greedy branch, with a probability of `strength / (1 + observations/10)` where `observations` sums the `usage_count` over that feature's edges. The prior only *biases sampling* and never removes an effect from `valid_effects`; it ships disabled (`strength = 0`), because seeding it collapsed the expert pool into a single basis family.

---

## 5. Tensor Complexity Guardrails

`context.estimate_complexity` acts strictly as a **ranking-only** estimator feeding the parsimony penalty, where the tensor branch applies its discounted surrogate math.

However, `TensorProductEffect.get_n_coeffs()` in `spectrum/_tensor.py` must remain **deliberately untouched**. It represents the authoritative width of the effect's coefficient block and exclusively governs slice arithmetic across the codebase (e.g., `coeff_idx += effect.get_n_coeffs()` in `bode.py` and `diagnostics.py`). Returning anything other than the true Kronecker product here would misalign all subsequent coefficient blocks, producing silently corrupted numbers rather than an exception. Any future complexity discounts belong exclusively in the scoring estimator.

---

## 6. Strict Engineering Constraints (CONTRIBUTING §3)

* **No Python Loops over Rows:** Binning, group demeaning, and residual construction must remain fully broadcasted operations.
* **OOM Safety:** The process is memory-limited to one `[P, P]` solve and one `[N, 1+P]` copy. Nothing scales with `N x N`, and no actual iterative model fits occur, ensuring the stage adds zero exposure to the intermittent Out-Of-Memory failures seen on heavy grouped fits.
* **Inline Comments:** Code comments must remain focused strictly on engineering (shapes, fallbacks, invariants). The mathematical estimators are justified exclusively in the paired `math/` document.

---

## 7. File Touch List

The stage spans the following files:

* `autotam/feature_profiler.py` (the stage itself)
* `autotam/effect_selector.py` (the profiler call, verdict application, basis-typed `ap`, and capacity bounding)
* `autotam/knowledge_graph.py` (`basis_prior`, `set_basis_prior`, `_prior_probability`, and the sampling hook)
* `autotam/drag_tam.py` (prior seeding at `optimize` entry)
* `autotam/pipeline/context.py` (the tensor ranking surrogate and Fourier `m=` parsing)
