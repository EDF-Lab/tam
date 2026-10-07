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

```{literalinclude} ../../../../src/tam/model/autotam/pipeline/expert_expander.py
:language: python
:start-after: "#: <expert_expander_generate>"
:end-before: "#: </expert_expander_generate>"
:caption: src/tam/model/autotam/pipeline/expert_expander.py (Continuous and Adaptive Expansions)
```

---

## 6. OPERA Aggregation (`ensemble_selector.py`)

The final layer deploys the `OperaTAM` MLpol algorithm via the `EnsembleSelector`. Instead of selecting a single model, it constructs specialized sub-leagues (Static, Kalman, Adaptive, Island Federation, and the global Apex Ensemble), weighting predictions sequentially via cumulative regret bounds.

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
