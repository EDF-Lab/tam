# AutoTAM Orchestrator Pipeline: Engineering Architecture

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/08_auto_orchestrator.md)
* **Related AutoTAM architecture:** [Data Topology Code](09_auto_data_topology_code.md)

> Mirror-Architecture companion to `math/meta/08_auto_orchestrator.md`.
> This document details the software engineering implementations governing the `autotam` module. It focuses on the programmatic enforcement of the 7-Step Pipeline: Bi-Level Optimization, Topological Sanitization, Dynamic Annealing, and Deterministic State-Space Expansion.

---

## 1. Pipeline Context and Data Management

To avoid passing excessive arguments across nested classes, a common anti-pattern in AutoML software, the framework utilizes a central `PipelineContext`. This Dataclass securely transports the temporal Cross-Validation folds, the physical metadata, and the exact complexity estimator.

The structural degrees of freedom ($k$) are calculated directly from parsed term dictionaries rather than by blindly searching the formula text. Tensor products are ranked using the discounted surrogate defined in the orchestrator theory, while the linear solver retains their true Kronecker width for physical matrix allocation. Formula parsing is cached per string, as the pipeline scores the identical formula for several experts. Furthermore, a combined dynamic formula's cost is computed strictly as the sum of its static and dynamic parts.

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/context.py
:language: python
:start-after: "#: <context_complexity_helpers>"
:end-before: "#: </context_complexity_helpers>"
:caption: src/tam/model/autotam/pipeline/context.py (Parsed-Term Complexity)

```

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/context.py
:language: python
:start-after: "#: <context_complexity_methods>"
:end-before: "#: </context_complexity_methods>"
:caption: src/tam/model/autotam/pipeline/context.py (Complexity Entry Points)

```

### Parser Signatures and Structural Complexity

The context parser estimates the structural complexity $k$ of each term from its signature, before any model is trained (the equations are in the mirror `math/` page):

| Topology / Effect | Parser Signature | Exact Structural Complexity ($k$) |
| :--- | :--- | :--- |
| **Linear / Categorical** | `l()`, `c()` | $1$ (per continuous variable / $n\_cat - 1$) |
| **Splines** | `s(k=N, deg=D)` | $N + D$ (Knots + Spline Degree) |
| **Fourier** | `f(m=N)` | $2 \times N$ (pairs of sine and cosine harmonics) |
| **Chebyshev / Poly** | `p(deg=N)` | $N$ (Degree of polynomial) |
| **Wavelets** | `w(n_scales=S, n_locs=L)`| $S \times L$ (Scales $\times$ Locations) |
| **RBF / Geospatial** | `rbf(n_centers=C)` | $C$ (Number of centroids) |
| **Trees / Forests** | `t(n_trees=T, max_depth=D)` or `t(n_trees=T, max_leaves=M)` | $T \times 2^D$, or $T \times M$ for a histogram (total terminal leaves) |
| **Linear Trees** | `lt(max_depth=D)` or `lt(max_leaves=M)` | $2 \times 2^D$ or $2M$ (one tree for the level, one for the slope) |
| **PID** | `pid(w=W)` | $3$ (the lag, its rolling mean over $W$ steps, its first difference) |
| **Neural** | `n(n_neurons=N, n_hidden=L)`| $N \times L \times 5$ (Heavy penalty for deep VRAM footprint) |
| **Tensor Products** | `te(eff_1, eff_2)` | ranked at $\min(\prod k_i,\ \sum k_i + \sqrt{\prod k_i})$; true width $\prod k_i$ (see the note below) |

`k=N` is also accepted as a legacy spelling of `f(m=N)`. A tensor product allocates its true width (`TensorProductEffect.get_n_coeffs`, for example `te(s(k=10), f(m=5))` has 130 columns) and is ranked by the discounted surrogate given in the math page.


The `DataManager` prepares the expanding window folds. It strictly enforces chronological integrity, guaranteeing that no future data leaks into validation blocks during the evolutionary search.

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/data_manager.py
:language: python
:start-after: "#: <data_manager_cv>"
:end-before: "#: </data_manager_cv>"
:caption: src/tam/model/autotam/pipeline/data_manager.py (Expanding Window Validation Mapper)

```

### Pre-Flight Constraints and Mandatory Validation (`data_manager.py`)

Before any evolutionary exploration commences, `DataManager.prepare` enforces strict pre-flight validation on user-supplied constraints:
* **Canonical Deduplication:** Mandatory terms are parsed directly from the formula RHS and normalized into canonical representations using `canonicalize_term`. If any redundant or duplicate terms are present in the formula (even if formatted with permuted kwargs, distinct quotes, or reordered tensor sub-terms), a fail-fast `ValueError` is raised immediately.
* **Strict Covariate Capacity Cap:** AutoTAM enforces a strict limit of `MAX_ACTIVE_EFFECTS_PER_FEATURE = 2` active mathematical effects per feature across all candidate formulas. If a user supplies $\ge 3$ mandatory terms targeting the same feature, `DataManager.prepare` rejects the configuration with an immediate fail-fast `ValueError`.
* **Tensor Term Ceiling:** Tensor product interactions are similarly bounded at `MAX_TENSOR_TERMS = 2` per formula to preserve numerical stability and avoid dimensionality explosion.

---

## 2. The Bayesian Brain (`knowledge_graph.py`)

The Estimation of Distribution Algorithm (EDA) requires a structural memory to navigate the infinite combinatorial space of mathematical formulas. The `KnowledgeGraph` implements a Bipartite tracker.

![Knowledge Graph: features on the left, bases on the right, edge weight = historical score of the pair](../../_static/knowledge_graph_bipartite.png)

It enforces strict parsimony via variance decomposition: terms that fail to explain variance or exhibit extreme collinearity (>0.98) are purged before they can populate the historical registry.

**Parameter-Order Invariant Protection:** During `update_and_prune`, candidate terms are checked against `mandatory_terms` via `terms_are_equivalent(term, mt)`. Any candidate term matching a mandatory term—regardless of keyword ordering, whitespace, or tensor sub-term permutation—is strictly exempt from variance decomposition and collinearity pruning. Additionally, the sole remaining term representing any feature listed in `mandatory_variables` is protected from variance pruning.

```{literalinclude} ../../../../src/tam/model/autotam/knowledge_graph.py
:language: python
:start-after: "#: <knowledge_graph_update_prune>"
:end-before: "#: </knowledge_graph_update_prune>"
:caption: src/tam/model/autotam/knowledge_graph.py (Variance Decomposition and Pruning)

```

When new genomes are requested, the graph balances exploration and exploitation via Softmax temperature scaling.

```{literalinclude} ../../../../src/tam/model/autotam/knowledge_graph.py
:language: python
:start-after: "#: <knowledge_graph_sampling>"
:end-before: "#: </knowledge_graph_sampling>"
:caption: src/tam/model/autotam/knowledge_graph.py (Epsilon-Greedy and Softmax Sampling)

```

---

## 3. Distributed Search Spaces (`population_nodes.py`)

To guarantee extreme mathematical heterogeneity, formula generation is physically distributed across specialized "Islands."

* **Deep Islands:** Islands such as `NeuralIsland`, `RBFIsland`, and `TreeIsland` condition a deep basis on categorical partners through the `others=` parameter. They select one to five partners (`MAX_INTERACTION_PARTNERS`), each chosen by the Knowledge Graph's interaction score, provided each has at least two levels in the fitting window (creating a conditional curve per level). One deep-basis term in four (`PLAIN_DEEP_TERM_PROBABILITY = 0.25`) is drawn without any partner.
* **Specialized Terms:** `TreeIsland` uniquely proposes linear trees (`lt`). All Islands may propose a `pid` term, restricted to the most recent autoregressive lag. The `_get_safe_params` function enforces a choice between `max_depth` and `max_leaves` on tree terms, as `TreeEffect` ignores depth once explicit leaves are defined.

```{literalinclude} ../../../../src/tam/model/autotam/population_nodes.py
:language: python
:start-after: "#: <interaction_islands>"
:end-before: "#: </interaction_islands>"
:caption: src/tam/model/autotam/population_nodes.py (Deep Island interaction handlers)
```

* **Tensor Products:** `te` terms are open to every Island (capped at `MAX_TENSOR_TERMS` = 2 per formula). When the Knowledge Graph draws a `te` for a continuous feature diagnosed as non-linear, the partner must be either a categorical (with $\ge$ 2 levels) or another continuous feature.
* With a categorical partner: The continuous side retains its profiled capacity, utilizing the Island's own basis if applicable (`s`, `f`, `p`, or `w`).
* With a continuous partner: Both sides default to a small, fixed basis (`CONTINUOUS_TENSOR_BASES`), ensuring the resulting surface never exceeds 25 columns.

**Mutation Protection & Canonical Deduplication:** The `BaseIsland.mutate` and `_clean_and_join_terms` methods maintain structural hygiene and protect mandatory formula components:
* **Canonical Term Deduplication:** Newly mutated or joined candidate terms are normalized via `canonicalize_term`, preventing duplicate terms with permuted arguments from ever appearing within a formula.
* **Verbatim Mandatory Preservation:** When a candidate term is canonically equivalent to a user-supplied mandatory term, the exact verbatim string from `mandatory_terms` is restored, ensuring user formatting and parameter conventions remain pristine.
* **Covariate Lock & Immutability:** A term is **immutable** if `any(terms_are_equivalent(term_str, mt) for mt in mandatory_terms)` and cannot be modified or deleted regardless of parameter ordering. A term is **hyper-immutable** if it is the last remaining term for a feature listed in `mandatory_variables`; it cannot be deleted, but its hyperparameters may be mutated. Furthermore, candidate generation strictly respects `MAX_ACTIVE_EFFECTS_PER_FEATURE = 2` and `MAX_TENSOR_TERMS = 2`.

```{literalinclude} ../../../../src/tam/model/autotam/population_nodes.py
:language: python
:start-after: "#: <tensor_term>"
:end-before: "#: </tensor_term>"
:caption: src/tam/model/autotam/population_nodes.py (Tensor products open to every Island)
```

* **Evolutionary Anchoring:** To maintain baseline stability, the `SmallContinent` enforces the generation of highly diverse but mathematically minimal topologies (e.g., $k=1$, $deg=1$).

```{literalinclude} ../../../../src/tam/model/autotam/population_nodes.py
:language: python
:start-after: "#: <smallcontinent>"
:end-before: "#: </smallcontinent>"
:caption: src/tam/model/autotam/population_nodes.py (The Bacteria Anchor for Evolutionary Diversity)
```

---

## 4. The Evolutionary Outer Loop (`drag_tam.py`)

`DragTAM` controls the EDA loop via Hub-and-Spoke Bi-Level Optimization: it handles the discrete topological search iteratively across generations, completely delegating the continuous hyperparameter optimization to the inner PyTorch GCV solvers.

**Mandatory Terms & Variables:** The engine extracts `mandatory_terms` directly from the formula RHS (e.g. `load ~ s(temp, k=10) + AutoPipe(temp)`) while `mandatory_variables` (feature names) can be supplied via `__init__`. These constraints are strictly enforced:
* **Generation & LIFO Non-Mandatory Eviction:** The `_ensure_mandatory_constraints` method injects any missing mandatory terms or variables into newly generated formulas. When injecting missing mandatory constraints causes an active feature to exceed `MAX_ACTIVE_EFFECTS_PER_FEATURE = 2`, non-mandatory terms assigned to that feature are evicted in Last-In-First-Out (LIFO) order. This guarantees that user-defined mandatory terms are strictly preserved without ever exceeding the covariate capacity limit.
* **Canonical Invariant Matching:** Matching existing candidate terms against mandatory terms uses `terms_are_equivalent`, ensuring parameter-order and sub-term invariance.
* **Mutation:** `BaseIsland.mutate` prevents the deletion or mutation of mandatory terms under canonical equivalence. It also protects the last remaining term for any mandatory variable from being deleted.
* **Ablation:** The parsimony-driven ablation step in the main evolutionary loop is forbidden from removing any term present in `mandatory_terms`.

Crucially, it utilizes **Dynamic Annealing** on the complexity penalty. The penalty multiplier starts at 1.0 (encouraging structural exploration) and scales up to 5.0 over the generations, applying crushing evolutionary pressure to extract only the most parsimonious architectures.

**Island Selection:** Handled by `KnowledgeGraph.select_island_ucb`, every Island is guaranteed a minimum budget of 3 pulls. Subsequently, it draws uniformly 15% of the time (`DragTAM.ucb_min_pulls`, `DragTAM.ucb_epsilon`) and otherwise follows UCB-E. If a candidate fails to fit, it is securely registered as a zero-reward pull.

```{literalinclude} ../../../../src/tam/model/autotam/knowledge_graph.py
:language: python
:start-after: "#: <knowledge_graph_island_ucb>"
:end-before: "#: </knowledge_graph_island_ucb>"
:caption: src/tam/model/autotam/knowledge_graph.py (Island selection)
```

```{literalinclude} ../../../../src/tam/model/autotam/drag_tam.py
:language: python
:start-after: "#: <dragtam_optimize>"
:end-before: "#: </dragtam_optimize>"
:caption: src/tam/model/autotam/drag_tam.py (Dynamic Annealing Optimization Loop)
```

---

## 5. Top Formula Extraction & State-Space Expansion

Once the `DragTAM` engine concludes, the `BaseDiscoverer` acts as the parser and extractor. It analyzes the evaluation cache, mathematically classifies every evaluated formula back into its originating Island, and extracts the champions.

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/base_discoverer.py
:language: python
:start-after: "#: <base_discoverer_search>"
:end-before: "#: </base_discoverer_search>"
:caption: src/tam/model/autotam/pipeline/base_discoverer.py (Island Champion Extraction)
```

These extracted champion topologies are routed to the `ExpertExpander`. This module physically instantiates the tracking environments, building local Adaptive Error Correction Models (ECM) and mapping the base architectures to the Extended Kalman Filter (EKF).

The state of a Kalman expert holds one coefficient per effect of its base model, so its formula has one linear term per effect column returned by `decompose_prediction` (function terms only: the formula grammar rejects bare names):

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/expert_expander.py
:language: python
:start-after: "#: <expert_expander_kalman_formula>"
:end-before: "#: </expert_expander_kalman_formula>"
:caption: src/tam/model/autotam/pipeline/expert_expander.py (Kalman State Formula)
```

The ensemble stage writes every expert name into a formula and reads its weight back by that name, so a name holds letters, digits and underscores only. The process-noise rate that distinguishes the Kalman experts is written into the name (`8.9e-05` becomes `8_9e_05`); a `-` or a `+` would be read as an operator and the weight of the expert would be lost:

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/expert_expander.py
:language: python
:start-after: "#: <expert_expander_kalman_name>"
:end-before: "#: </expert_expander_kalman_name>"
:caption: src/tam/model/autotam/pipeline/expert_expander.py (Kalman Expert Name)
```

The grid-search experts choose their hyperparameters on one fold and are scored on all of them. The folds of a time series are expanding windows, so the choice is made on the last one, the period closest to the test rows:

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/expert_expander.py
:language: python
:start-after: "#: <expert_expander_grid_fold>"
:end-before: "#: </expert_expander_grid_fold>"
:caption: src/tam/model/autotam/pipeline/expert_expander.py (Grid-Search Fold)
```

A candidate that cannot be fitted is dropped and the search goes on, but never silently: every place that survives a failure logs what failed, its formula and the error through one helper (a warning for the experts of the final pool, a debug record for the many candidates the evolution discards):

```{literalinclude} ../../../../src/tam/model/autotam/failures.py
:language: python
:start-after: "#: <failures_log>"
:end-before: "#: </failures_log>"
:caption: src/tam/model/autotam/failures.py (Logged Failures)
```

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/expert_expander.py
:language: python
:start-after: "#: <expert_expander_generate>"
:end-before: "#: </expert_expander_generate>"
:caption: src/tam/model/autotam/pipeline/expert_expander.py (Continuous and Adaptive Expansions)
```

---

## 6. OPERA Aggregation (`ensemble_selector.py`)

The final layer deploys the `OperaTAM` MLpol algorithm via the `EnsembleSelector`. Instead of selecting a single model, it constructs specialized sub-leagues (Static, Kalman, Adaptive, Island Federation, and the global Apex Ensemble), weighting predictions sequentially via cumulative regret bounds.

The adaptive and Kalman experts rebuild their windows and their state from the frame they are given, so the forecast of a row depends on the rows that precede it. `fit()` keeps the training rows (`history_`); a frame that starts after them gets them in front of it, and the output has exactly the rows given (a forecast on the forecast rows alone equals the same rows inside past + future):

```{literalinclude} ../../../../src/tam/model/autotam/auto_tam.py
:language: python
:start-after: "#: <auto_tam_history>"
:end-before: "#: </auto_tam_history>"
:caption: src/tam/model/autotam/auto_tam.py (Training Rows in Front of the Forecast Rows)
```

When the search keeps no expert (too few rows for any formula to be estimated), `fit()` raises a `ValueError` that names the rows of each split and says what to change; an empty `df_fit`, `df_dev` or `df_val` is rejected before the search with the row counts of the three splits. `df_train` is split by fractions, so it works from 12 rows on the one-feature test case; explicit splits from 5 fit rows with 2 dev and 2 val rows:

```{literalinclude} ../../../../src/tam/model/autotam/auto_tam.py
:language: python
:start-after: "#: <auto_tam_no_expert>"
:end-before: "#: </auto_tam_no_expert>"
:caption: src/tam/model/autotam/auto_tam.py (Error When No Expert Survives)
```

Past-only covariates (columns known up to the forecast origin, such as an availability that is observed but not planned) are declared in `fit(past_covariates={"availability": [24, 48]})`. The covariate itself is never a feature, because its value at a forecast row is not known; its lags are built per group like the lags of the target (`availability_lag_24`, `availability_lag_48`), added to the candidate features of the search, and rebuilt from the stored history at prediction. A lag below the horizon (the smallest lag of the target), an unknown column, or a covariate that is also named in the pipeline macro is rejected with a message:

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/data_manager.py
:language: python
:start-after: "#: <data_manager_past_covariates>"
:end-before: "#: </data_manager_past_covariates>"
:caption: src/tam/model/autotam/pipeline/data_manager.py (Past-Only Covariates)
```

Seed formulas (`fit(seed_formulas=["y ~ l(x1) + s(x2, k=10)"])`) are formulas the user already trusts. They are validated before the search (target, syntax, columns), then added to the champions of the search under the island name `Seed` and expanded like any champion: prior, Kalman and adaptive experts named `..._Seed_R<k>_...`, which appear in the performance board, in the leagues and, when they pass the quality floor, in the Apex pool. A seed is never forced into the final model: it competes, so the search cannot do worse than a good seed on the validation rows:

```{literalinclude} ../../../../src/tam/model/autotam/auto_tam.py
:language: python
:start-after: "#: <auto_tam_seed_formulas>"
:end-before: "#: </auto_tam_seed_formulas>"
:caption: src/tam/model/autotam/auto_tam.py (Seed Formulas)
```

The extrapolation guards (`AutoTAM(extrapolation_guards=True)`, off by default) keep the worst extrapolations out of the search. A champion is refused when it has more coefficients than half the fit rows, a neural or PID term on fewer than 1,000 rows, or a linear term on a counter (a monotone column such as a trend index); the reasons are counted in the context (`guard_rejections`). The forecast bounds keep the training range plus a 15 % margin and are floored at zero for a target that was never negative; the rows each expert had to be clipped on are counted against it in the selection and reported by `AutoTAM.clipped_rows_` after a prediction. The thresholds are constants of the module, to be tuned on the multi-seed measurement before the guards become the default:

```{literalinclude} ../../../../src/tam/model/autotam/guards.py
:language: python
:start-after: "#: <guards_constants>"
:end-before: "#: </guards_constants>"
:caption: src/tam/model/autotam/guards.py (Thresholds)
```

```{literalinclude} ../../../../src/tam/model/autotam/guards.py
:language: python
:start-after: "#: <guards_rules>"
:end-before: "#: </guards_rules>"
:caption: src/tam/model/autotam/guards.py (Rules)
```

The seasonal anchor (`AutoTAM(seasonal_anchor=True, season=None)`, off by default) puts the "last season repeated" reference in the pool. It is an expert of its own, a regression of the target on its value one season earlier (`y ~ l(anchor_lag_<season>)`, an intercept and a slope calibrate the reference), added to the champions under the island name `Anchor` and expanded like any champion (prior, Kalman and adaptive experts), so that the leagues and the Apex pool can fall back on it. The season is the natural season of the sampling step (24 hourly, 7 daily, 52 weekly, 12 monthly, 4 quarterly) unless `season=` is given; it must be at least the horizon, the smallest lag of the target, because its value one season earlier has to be known at the forecast origin:

```{literalinclude} ../../../../src/tam/model/autotam/anchor.py
:language: python
:start-after: "#: <anchor_rules>"
:end-before: "#: </anchor_rules>"
:caption: src/tam/model/autotam/anchor.py (Seasonal Anchor)
```

Before the search the data manager reads the facts about the training target that make a fit wrong when nobody looks (`AutoTAM.target_profile_`): the sentinels (a value that repeats at an extreme of the range, far outside the robust spread, such as a missing-value code -9999), the physical bounds (a proportion in 0 to 1, a percentage reaching 100), the monotone share (a cumulative series), the integer, zero and negative shares, the skewness and kurtosis, and a one-word family hint: `cumulative`, `bounded`, `count`, `tweedie`, `log`, `asinh` or `gaussian`. A sentinel is always reported with a warning and fitted as a value unless `replace_sentinels=True` turns it into a missing value, because replacing changes the data. The hints are reported only: the search does not use them yet:

```{literalinclude} ../../../../src/tam/model/autotam/target_profile.py
:language: python
:start-after: "#: <target_profile_rules>"
:end-before: "#: </target_profile_rules>"
:caption: src/tam/model/autotam/target_profile.py (Target Profile)
```

The search is reproducible with `AutoTAM(seed=42)`. The search draws from the global generators of `random`, `numpy` and `torch` (the knowledge graph, the population nodes and the evolutionary loop): `fit()` seeds the three, runs, and puts them back as it found them, so the same seed on the same data gives the same experts whatever was drawn before and whatever the hash seed of the interpreter, and the program around the fit is not disturbed. Candidates are ranked on their score rounded to 10 significant digits and then on their label, so a tie does not depend on the order they arrived in nor on the last digits a processor rounds differently, and the sets of column names that reach a formula are iterated in sorted order. The seed is written in the export next to the run (`AutoTAM_run_info_<timestamp>.json`). The generators are global, so the method is not thread-safe; the numerical solver of each static model is deterministic by itself, and a CPU and a GPU run are not made identical:

```{literalinclude} ../../../../src/tam/model/autotam/seeding.py
:language: python
:start-after: "#: <seeding_rules>"
:end-before: "#: </seeding_rules>"
:caption: src/tam/model/autotam/seeding.py (Seeding and Ranking)
```

`predict()` returns one column per expert in search order, then the league ensembles, then the final model; the first column is an expert, not the final model. `AutoTAM.final_model_column_` (set by `fit()`) names the final model column (`AutoTAM.FINAL_MODEL_COLUMN`, `AutoTAM_Apex_Ensemble`; `AutoTAM.FINAL_MODEL_COLUMN_ONLINE`, `AutoTAM_Apex_Online`, for `predict_online()`).

Frozen inference keeps the members of a league whose final weight reaches `ensemble_sparsity_threshold`. The weights of a league sum to one, so the Apex league, which aggregates the whole dynamic pool (several hundred members), can have no weight above the absolute threshold; the members at or above their uniform share `1/n` are then kept, so a league never ends empty:

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/ensemble_selector.py
:language: python
:start-after: "#: <ensemble_selector_sparse_weights>"
:end-before: "#: </ensemble_selector_sparse_weights>"
:caption: src/tam/model/autotam/pipeline/ensemble_selector.py (Sparse League Weights)
```

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/ensemble_selector.py
:language: python
:start-after: "#: <evaluate_and_refit>"
:end-before: "#: </evaluate_and_refit>"
:caption: src/tam/model/autotam/pipeline/ensemble_selector.py (Sequential Expert Evaluation)
```

**Apex Quality Floor:** Before aggregation, the Apex pool of dynamic experts must pass a quality floor (`apex_quality_ratio`, default 1.5). An expert is admitted only if its validation error is at most 1.5x the best dynamic expert's error, guaranteeing the two best are always kept. This filter acts as a vectorized mask over the pool's scores. The admitted pool is stored in `apex_members_`: `AutoTAM.predict_online` dynamically aggregates all of it, whereas the frozen `predict()` averages only the members still weighted above the sparsity threshold at the very end of validation (`weights_top10`).

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/ensemble_selector.py
:language: python
:start-after: "#: <apex_quality_filter>"
:end-before: "#: </apex_quality_filter>"
:caption: src/tam/model/autotam/pipeline/ensemble_selector.py (Apex Quality Floor)
```

---

## 7. Master API, Telemetry, and Automated Reporting

The `AutoTAM` object provides the unified public interface, cleanly hiding the nested pipelines from the user.

```{literalinclude} ../../../../src/tam/model/autotam/auto_tam.py
:language: python
:start-after: "#: <auto_tam_fit>"
:end-before: "#: </auto_tam_fit>"
:caption: src/tam/model/autotam/auto_tam.py (The execution pipeline sequence)
```

To guarantee auditability in industrial environments, the `EvolutionReporter` completely decouples diagnostic logging from mathematical inference, dumping the evolutionary history, collinearity purges, and sequential OPERA weight trajectories into CSV artifacts.

```{literalinclude} ../../../../src/tam/model/autotam/evolution_reporter.py
:language: python
:start-after: "#: <reporter_export_diagnostics>"
:end-before: "#: </reporter_export_diagnostics>"
:caption: src/tam/model/autotam/evolution_reporter.py (Telemetry and MLOps logging)
```

### Dashboard Generation (`autotam_report_generator.py`)

Analyzing dozens of CSVs manually defeats the purpose of an automated pipeline. The `autotam_report_generator.py` script parses these artifacts to compile a comprehensive 3x3 Matplotlib diagnostic dashboard. This allows Data Scientists to audit why the AI selected the final ensemble and view the scale-free variance shares ($\mathrm{Var}(h_j) / \mathrm{Var}(\hat{Y})$) driving the predictions.

**I/O Safeguards:** The report generator strictly sanitizes inputs to prevent path traversal or injection vulnerabilities:

* Run identifiers are validated via regex (only allowing letters, digits, `_`, and `-`).
* Every artifact path is resolved using `pathlib.Path.resolve()` and explicitly refused if it attempts to resolve outside the designated export directory.
* The `matplotlib` module is imported securely inside the function scope, allowing the pipeline to gracefully skip the report instead of crashing if the visualization library is missing in headless environments.

```{literalinclude} ../../../../src/tam/model/autotam/evaluation/autotam_report_generator.py
:language: python
:start-after: "#: <autotam_report_generator_helpers>"
:end-before: "#: </autotam_report_generator_helpers>"
:caption: src/tam/model/autotam/evaluation/autotam_report_generator.py (Run-ID validation, path confinement, driver scores)
```

```{literalinclude} ../../../../src/tam/model/autotam/evaluation/autotam_report_generator.py
:language: python
:start-after: "#: <autotam_report_generator_main>"
:end-before: "#: </autotam_report_generator_main>"
:caption: src/tam/model/autotam/evaluation/autotam_report_generator.py (Automated 3x3 Diagnostic Dashboard)
```
