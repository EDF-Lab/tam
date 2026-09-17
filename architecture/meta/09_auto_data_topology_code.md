# AutoTAM Data Topology: Stateful Architecture & Pandas Vectorization

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)

* **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/09_auto_data_topology.md)

* **Related AutoTAM architecture:** [Orchestrator Code](08_auto_orchestrator_code.md)


> Mirror-Architecture engineering document detailing the data topology modules.
> Implementing dynamic data processing for an automated evolutionary pipeline presents a significant engineering challenge. The pipeline must handle millions of rows efficiently while strictly guarding against target leakage between training and validation folds. This document explains the software engineering choices, specifically stateful dictionary tracking, regular expression caching, and vectorization, used to implement these modules.

---

## 1. Data Profiler: Stateful Bounding & Vectorized Clipping

The `DataProfiler` is responsible for cleaning data and enforcing group-aware anomaly bounds.

**Engineering Choice: The Stateful Metadata Dictionary**
To prevent data leakage, the class is designed with a strict `fit/transform` paradigm. During the `profile_and_clean` phase (Training), the algorithm learns the specific IQR boundaries and frequency deltas, storing them exclusively in a `self.metadata` state dictionary. When `transform` is called on unseen data (Inference), the profiler refuses to recalculate any statistics; it strictly applies the frozen `metadata` rules.

**Engineering Choice: Pandas Vectorization for Panel Data**
Applying different clipping bounds to millions of rows based on their specific group (e.g., substation ID) would cause a massive bottleneck if executed using Python `for` loops. Instead, the `_apply_clipping` method leverages highly optimized Pandas indexing (`groupby().groups.items()`) and the vectorized `.clip()` function to process massive panel datasets at C-level speeds.

```{literalinclude} ../../../../src/tam/model/autotam/data_profiler.py
:language: python
:start-after: "#: <data_profiler_helpers>"
:end-before: "#: </data_profiler_helpers>"
```

---

## 2. Feature Engineer: Efficient Collinearity Filtering

The `FeatureEngineer` constructs rolling windows and mathematical interactions. Its most critical engineering task is filtering out highly correlated features to protect the downstream Conjugate Gradient solver.

**Engineering Choice: Upper Triangle Masking**
Computing a full correlation matrix on a heavily augmented dataset is computationally expensive. To optimize this, the algorithm calculates the absolute correlation matrix (`.corr().abs()`) and immediately masks it using `np.triu` (the upper triangle offset by $k=1$). This halves the search space and strictly prevents a feature from being compared against itself or double-counted.

**Engineering Choice: Persistent Purge Logging**
If a generated feature (e.g., `temp_rolling_mean_24`) is deemed collinear and dropped during training, it must also be dropped during production inference, even if the correlation drops in the new data stream. The engineer tracks these in a `dropped_features` list, executing a blind, vectorized `.drop(columns=...)` on all future passes.

```{literalinclude} ../../../../src/tam/model/autotam/feature_engineer.py
:language: python
:start-after: "#: <feature_engineer_filter>"
:end-before: "#: </feature_engineer_filter>"
```

---

## 3. Parser: Pre-Compiled Semantic Routing

The `FormulaParser` translates the human-readable formula string into the configuration dictionaries utilized by the evolutionary engine.

**Engineering Choice: Regex Pre-Compilation**
Because the pipeline might parse thousands of distinct formulas during an evolutionary search generation, compiling Regular Expressions on the fly would introduce severe overhead. The `FormulaParser` explicitly compiles the `equation_regex` and `pipeline_regex` during the `__init__` phase.

**Engineering Choice: Graceful Degradation on Lags**
The parser is designed to handle specialized syntax like `Feature@Lag`. It extracts these into a dedicated `lags` dictionary. If a user attempts to pass a stateful lag injection into a static solver (like `StaticTAM`), the parser does not crash; instead, it raises a structured Python warning to guide the user toward the appropriate dynamic engine (`AdaptiveTAM` or `KalmanTAM`).

```{literalinclude} ../../../../src/tam/model/autotam/parser.py
:language: python
:start-after: "#: <parser_init>"
:end-before: "#: </parser_init>"
```

```{literalinclude} ../../../../src/tam/model/autotam/parser.py
:language: python
:start-after: "#: <parser_parse_method>"
:end-before: "#: </parser_parse_method>"
```

---

## 4. Effect Selector: $O(N)$ Covariate Lock Validation

The `EffectSelector` guarantees that a single feature isn't overwhelmed by competing mathematical bases (e.g., applying a spline, a tree, and a Fourier series to the exact same column).

**Engineering Choice: Dictionary-Based Counting**
The Covariate Lock validation is called constantly by the genetic engine to score generated genomes. It must be exceptionally fast. Instead of utilizing complex graph traversals, the `validate_covariate_lock` method uses a simple $O(N)$ loop over the genome terms, employing a standard Python dictionary (`feature_counts`) with `.get(feat, 0)` to track occurrences. As soon as a feature breaches the `max_active_effects` threshold, the function short-circuits and returns `False`, efficiently saving computational cycles.

```{literalinclude} ../../../../src/tam/model/autotam/effect_selector.py
:language: python
:start-after: "#: <covariate_lock_validator>"
:end-before: "#: </covariate_lock_validator>"
```

---

## 5. Performance Anti-Patterns

The panel datasets AutoTAM ingests (one series per smart meter, substation, or time-of-day slot) make per-group Python code the primary place where preparation time grows. The following software patterns are strictly not accepted in `src/tam/model/autotam/`:

* **Python callbacks per group:** Methods like `df.groupby(g)[col].transform(lambda x: ...)` and `.apply(...)` call back into the interpreter once per group, meaning their computational cost grows linearly with the number of groups. Use native grouped operations instead (`groupby(g)[col].rolling(...)`, `.ewm(...)`, `.shift(...)`, `.cumsum()`). At 48 groups, the two approaches are equivalent in practice (measured: 0.61 s with lambdas vs. 0.73 s natively for 34 columns over 52,608 rows); however, this rule exists to protect scaling when the number of groups grows massively.


* **Label-aligned write-back of grouped results:** A grouped window operation returns rows in group order under a `(group, index)` MultiIndex. Assigning it back by label misaligns rows as soon as the index holds duplicate labels (for instance, after concatenating explicit folds). Developers must compute on a positional copy and write back by position, exactly as `FeatureEngineer._create_temporal_features` does.


* **Row loops:** Utilizing `iterrows()` or per-row `.loc` assignments in a data path is banned. Use column arithmetic, `np.where`, `.clip()`, or vectorized merges.

* **Redundant copies of large frames:** Operations like `dropna`, boolean masks, `.loc` with an index array, and `pd.concat` already return new frames. Because every TAM model copies its input before adding helper columns (`_ensure_dummies`), a trailing `.copy()` duplicates the frame for no reason. Keep an explicit copy only when the frame will be written to afterwards.

* **Retained per-model working state:** A model kept in an expert pool must not also retain the tensors and frames used to score it. Measured on the national load, one single adaptive expert retained 51 MB to 498 MB of simulation state (7- to 365-period windows). Across a pool of several hundred experts, this accumulated to the 60-73 GB RAM peaks observed in earlier framework runs. Such state must be released as soon as its predictions are read (e.g., `EnsembleSelector._get_expert_predictions`).

```{literalinclude} ../../../../src/tam/model/autotam/feature_engineer.py
:language: python
:start-after: "#: <feature_engineer_transformations>"
:end-before: "#: </feature_engineer_transformations>"
:caption: src/tam/model/autotam/feature_engineer.py (Native grouped window operations with positional write-back)
```
