
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
- **Mandatory Terms Use Case:** `use_cases/autotam_mandatory_terms.py` runs AutoTAM free, with a mandatory term (`s(x1, k=10)`) and with a mandatory variable (`x10`) on the cheatsheet data, and fails if a static model of the search loses the term or the variable.

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
- **Visible Expert Failures:** an expert whose cross-validation evaluation fails is logged (`logging` warning with the formula and the fold) instead of being scored infinite silently.

### Fixed

- **Knowledge Graph Pruning Fix:** Fixed term contribution lookups so redundant terms are correctly identified, rewarded, and pruned.
- **Simulation Memory Leak:** `EnsembleSelector` now aggressively releases AdaptiveTAM simulation tensors (`m_ref.simulation_data_ = None`) after scoring, resolving 60-73 GB RAM peaks during wide grid searches.
- **Time-Series Integrity:** Fixed Kalman/Adaptive prediction backtest alignment, patched a `groupby` crash caused by Daylight Saving Time (DST) duplicates, and prevented `__dummy_date__` overflows on massive inputs.
- **Grid-Search Experts:** the grid-search experts of `ExpertExpander` were never fitted (the `grid_search_fit(cv_folds=...)` call did not match the engine's `grid_search_fit(data_train, data_val, grid_search_config)`); they are now fitted on the first cross-validation fold.
- **Parenthesis Mismatch:** combining mandatory terms and mandatory variables no longer produces formulas with unbalanced parentheses.
- **Kalman Experts:** the search never trained a KalmanTAM expert: the formula it gave each one used bare effect names, which the formula grammar rejects, and the error was swallowed. Each expert now tracks one linear term `l(effect_...)` per effect of its base model, so Kalman experts enter the candidate pool and AutoTAM forecasts change.
- **Kalman Expert Names:** the name of a Kalman expert holds its process-noise rate, and a rate such as `8.9e-05` put a `-` in the name: the ensemble formula read it as a subtraction, the expert got no weight and the search failed with a `KeyError` (or ended without `AutoTAM_Apex_Ensemble`). Expert names now keep letters, digits and underscores only, and a missing weight raises an error naming the expert. Forecasts of searches that did not fail are unchanged.
- **Large Apex Leagues:** frozen inference kept only the league members whose final weight reached the absolute sparsity threshold (0.01). The Apex league aggregates the whole dynamic pool, so with several hundred members no weight reached it, the league kept nothing and `predict()` returned no `AutoTAM_Apex_Ensemble`. When no member reaches the threshold, the members at or above their uniform share `1/n` are kept. Leagues that already kept members give the same forecasts.

---

## [1.4.1] - 2026-10-08

Patch release for probabilistic forecasting: new methods and options only, plus one fix. Existing calls give the same forecasts; the only change of behaviour is that a distributional or mixture `StaticTAM` with the default log target now raises on a training target with values `<= 0` instead of returning a meaningless model (see "Fixed").

- The `DCO` sign-off is required on every commit of a pull request (`DCO`, `CONTRIBUTING.md`, `docs/contributing/dco_howto.md`); a `REUSE` workflow runs `reuse lint` on each push and pull request.
- The `pid()` page of `architecture/core/06_the_spectrum_api.md` says when `d_pen` acts and that the derivative is computed along the frame given to `predict()`; three tests pin `pid` behaviour.

### Added
- **Limiting the CPU threads is documented and tested** (`architecture/core/04_hardware_memory.md`, README): set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before starting Python. The page also describes a hang of `torch.set_num_threads(n)` seen on one PyTorch / oneMKL build, which reproduces without `tam`; the results of `tam` are unchanged.
- **`loss="negative_binomial"` and `loss="tweedie"`** on `StaticTAM` (log link): overdispersed counts, and positive targets with exact zeros (sparse retail, rain, night-time solar). The negative binomial dispersion is estimated by maximum likelihood after the IRLS fit (or fixed with `loss_kwargs={"dispersion": alpha}`); the Tweedie power is `loss_kwargs={"power": p}` with `1 < p < 2` (1.5 by default) and its dispersion is the Pearson estimate. The estimate is `model.dispersion_`. A negative target raises a `ValueError`.
- **`StaticTAM.predict_quantiles(data, taus)` for `poisson`, `negative_binomial` and `tweedie`**: discrete quantiles for the counts and the quantiles of the compound Poisson-gamma law for Tweedie, which are `0` where the probability of zero exceeds the level; non-negative and non-decreasing in the level by construction. The Poisson loss is otherwise unchanged.
- **`KalmanTAM.predict_quantiles(data, taus)`**: Gaussian quantiles of the online forecast from the predictive variance of the filter, `x' P x + R` in the target scale (for `horizon_steps > 1`, the covariance of the delayed state grown by the process noise of the delay). Returns a DataFrame with `q<tau>` columns, like the distributional `StaticTAM`; like `predict_online`, `data` holds the target. The constructor, `predict_online`, `predict` and `fit` are unchanged. The intervals are as calibrated as `observation_noise_var` and `process_noise_var` are.
- **`AdaptiveTAM.predict_quantiles(data, taus)`**: online quantile forecasts when the `base_model` is a distributional `StaticTAM` (a dict formula). The adaptive correction moves the location; the scale of the base is rescaled at each update by the root mean square of the standardized residuals of the last `training_window_periods` rows that are at least `horizon_steps` old (past residuals only), and the quantiles go back through the target transform of the base. Returns a DataFrame with `q<tau>` columns, like the distributional `StaticTAM`; like `predict_online`, `data` holds the target. A plain base raises a `ValueError`. `predict_online`, `predict` and `fit` give the same results as before for a plain base, and now also accept a distributional one.
- **`dist_kwargs={"target_transform": "asinh" | "logit"}`** on a distributional `StaticTAM`: the location-scale law sits on `arsinh((y - c) / s)` (centre and robust scale estimated on the training target; a skewed, heavy-tailed target with negative values, such as electricity prices) or on the logit of `(y - a) / (b - a)` (a target bounded in `bounds=(a, b)`, estimated from the training range when not given). `predict_median`, `predict_quantile(s)`, `cdf`, `crps`, `anomaly_score` and the conformal intervals work through the transform, and quantiles stay inside the bounds for the logit. `target_transform="log"` is the default and the same as `log_target=True`; `"none"` the same as `log_target=False`. A mixture accepts `"none"` and `"log"` only.
- **`f(x, ..., period=...)`**: the range of one cycle of a Fourier input, on which the feature is normalised in `StaticTAM` and in each rolling window of `AdaptiveTAM`. `period=None` (default): the minimum and maximum of the training rows, as before. `period='auto'`: first value to last value plus one step, the step read from the training rows (hours 0 to 23 give (0, 24), months 1 to 12 give (1, 13)). `period=(low, high)`: given, for a history covering only part of the cycle (`(0, 1)` for a time of year trained on half a year). Without it the training range is still taken as one period, which glues the last value of an integer-coded position onto the first (`f(hour, cyclic=True)` with hours 0 to 23 forces 23:00 = 0:00; Sunday = Monday for 0 to 6; December = January for 1 to 12) and reads a history covering half a year (or a 60-day adaptive window) as a whole year: use `'auto'` for the first case and a given range for the second. Results without `period` are unchanged.
- **`OperaTAM(..., loss_type='pinball', tau=0.9)`**: online aggregation of quantile forecasts with the pinball loss of level `tau` (MLpol and EWA). `tam.model.opera.aggregate_quantiles(df, target_col, expert_cols_by_level)` aggregates every level and sorts them on each row. The square and absolute losses are unchanged.
- **`SafetyTAM.predict_quantiles(y_pred, levels, scale=None, signed=False)`**: conformal quantiles at several levels in one call (`q<level>` columns, sorted on each row). By default the symmetric bands of `predict_intervals`; `signed=True` calibrates each tail on the signed residuals kept by `calibrate()` (one-sided split conformal), for skewed errors. `calibrate(..., verbose=False)` silences the sample-count line, printed by default as before.

### Fixed
- **A distributional or mixture `StaticTAM` with the default log target rejects a training target with values `<= 0`** (`ValueError` naming the count and `dist_kwargs={"log_target": False}`). They were clipped to a tiny constant before the log, i.e. silently turned into huge negative outliers, and the fitted location, scale and quantiles were meaningless. Positive targets are unchanged.

## [1.4.0] - 2026-10-06

First release on PyPI and Zenodo since 1.3.1; it also carries everything of 1.3.2 to 1.3.4, which were tags only. **`AdaptiveTAM`, `KalmanTAM` and `NeuralTAM` [EXP] forecasts change** (see 1.3.3, 1.3.4 and the sections below); `StaticTAM` changes only for `te()` with GCV (-2%), a mixture mean (-0.6%) and a prediction without the group column. To get the 1.3.1 behaviour: `pip install tam-ml==1.3.1`.

### Changed
- **A formula with an empty argument is rejected** (`s(x,,k=5)`, `s(x, k=5,)`, `te(s(x),, s(z))`): `split_args_respecting_parentheses` now raises a `ValueError` on a leading, trailing or doubled delimiter and on unbalanced parentheses (it used to ignore a trailing empty argument silently). Contributed by Amaury Durand.
- **`te()` has one smoothing parameter per margin** (mgcv semantics): `auto_fit` (GCV) tunes the weight of each margin separately, so a tensor product can be smooth along one axis and rough along another (before, the margins were multiplied by one weight GCV could barely move: on a plane `z = x + y` the surface used 76 effective degrees of freedom, now 4). A weight given to the tensor product itself (`te(..., ap=)` or the default) is folded into each margin once; **with fixed weights the penalty is unchanged** (forecasts are bit-identical), but `auto_fit` results with a `te()` change, because the search can now move the penalty: a larger `gamma` (2.0) can now select an over-smoothed surface, `gamma=1.4` (the default, kept) selects the same fit as before in the cases measured. `auto_fit` reports and `summary()` show one value per margin. New effect API: `penalty_coordinates()`, `set_penalty_coordinates()`, `n_penalty_coordinates` (one coordinate for every effect but a tensor product).
- **`auto_fit` default `gamma=1.4` kept after re-tuning** on the fixed engine: no other value improves the validation RMSE by more than 0.5% on most cases, and 1.0 can be much worse (see the v1.4.0 Pull Request description on GitHub for the full benchmark and engine diagnosis).
- **`NeuralTAM` is reproducible** (`NeuralTAM(..., seed=42)`): the validation split, the epoch shuffling and the network initialisation use their own generators, so a fit depends on `seed` and not on the random draws made before it, and it neither reads nor advances the global torch generator. **Neural results change once** (the draws are not the same as before); they are then stable.
- **`NeuralTAM(guard=True)`** keeps a network only where it beats, on the validation rows, the closed-form effect it replaces; `network_used_` says where, a `UserWarning` is raised when no network is kept (the model then equals its `StaticTAM`), and `guard=False` keeps every network as before.

### Added
- **API contract tests** (`tests/test_api_contracts.py`): the signatures of the main models and of the formula parser, the `decompose_prediction` column names and the `summary()` keys are pinned; adding a keyword argument with a default passes, changing an existing parameter fails.
- **`KalmanTAM(process_noise_var={term: q, "offset": q, "default": q})`**: one process noise per formula term (a diagonal `Q`); every design column of a term gets the term's `q`, the terms not named use `"default"` (1e-4). A float works as before, bit for bit. `KalmanTAM.term_columns()` lists the terms and their columns; `tune_hyperparameters` accepts one candidate dict per grid entry.
- **A use-case workflow in CI** (`.github/workflows/use_cases.yml`): every script of `use_cases/` runs headless on each push and pull request of `main`, `automl`, `feat/**` and `release/**`, one job per script, and one `Use cases` check fails when any of them does.
- **Regression checks, one script per functionality** (`tests/regression/`): 38 scripts (every family of effects, every argument of every effect, mixed formulas and their decomposition, interactions, fixed and GCV fits, losses, distributional and conformal, the prediction API, `AdaptiveTAM`, `KalmanTAM`, `OperaTAM`, `HierarchicalTAM`, `NeuralTAM`, the errors and warnings a user meets, hyper-parameter search, a small `AutoTAM`, plots, the extrapolation of splines) run on one frozen year of the FORCE dataset and write `tests/regression/RESULTS.txt` (stable modules) and `RESULTS_EXP.txt` (beta and experimental ones), text files that are committed. A `Regression` workflow reruns them on every pull request and fails when `RESULTS.txt` changes: the diff is what the maintainer reviews; a change of the experimental file is only a warning. They need only tam, take about two minutes, and do not depend on the datasets shipped with the package.
- **One license header in every source and test file**: `SPDX-FileCopyrightText` lines, one `SPDX-FileContributor` line per author (the free-text `# Author(s) :` lines became contributor lines, nothing was lost), then `SPDX-License-Identifier`; `REUSE.toml` covers the files that carry no header and `LICENSES/LGPL-3.0-or-later.txt` holds the license text (REUSE layout). `tests/test_spdx_headers.py` checks the shape.
- **Contributor files**: `.github/CODEOWNERS` (every path reviewed by the maintainer), a pull request template (one checklist row per CHANGELOG line, plus the pre-push verdict), issue templates (bug, feature, research), and `CONTRIBUTING.md` sections on shared feature branches (fork, pull request into `feat/*`, two review slots a week with a three-working-day turnaround) and on `print()` in new code. The README, `AUTHORS.md` and the citation file no longer mention a journal evaluation: the citation file has no placeholder identifier.
- **`AdaptiveTAM` warns when a forecast level was not in the training rows of its window** (`TAMExtrapolationWarning`, once per categorical feature and model): a holiday seen for the first time, or a weekday missing from a short window, gets a penalised near-zero column silently; the message gives the level(s) and in how many windows it happens. Silent without a categorical term; `collect_extrapolation()` collects it like the range warning.
- **`compact()`** on `StaticTAM`, `AdaptiveTAM`, `KalmanTAM` and `OperaTAM`: drops what grows with the data (`predictions_`, the state and weight histories except their last row) and returns the model; `predict()` is unchanged to the last bit. See "Model size and persistence" in the architecture notes.

### Fixed
- **`AdaptiveTAM` frees its simulation cache** when `predict_online()`, `fit()` or `grid_search_fit()` ends. Previously, every overlapping window was kept stacked in memory, accounting for the vast majority of the model's pickled size. The model footprint is now drastically reduced, and `compact()` makes the size of a fitted `AdaptiveTAM`, `KalmanTAM` or `OperaTAM` independent of the data. `prepare_simulation()` alone still fills `simulation_data_`.
- **`OperaTAM` and `NeuralTAM` accept a frame whose columns are stored in another order than the formula lists them**: with pandas 3, selecting the expert (or network input) columns in the reverse of their storage order gave a view with negative strides that torch refuses (`ValueError: At least one stride in the given numpy array is negative`); the tensors are now built from a contiguous copy, and the forecasts do not depend on the column order.
- **`NeuralTAM.decompose_prediction` names a network's effect like the other models** (`effect_n_x` when `x` also has a `s()` or `l()` term, as `StaticTAM` does): it added a stray `effect_x` column next to the `effect_n_x` one, so the effects no longer summed to the forecast and an `AdaptiveTAM` behind a `NeuralTAM` saw a different set of columns than behind a `StaticTAM`. Formulas where the network is the only term on its feature are unchanged.
- **`NeuralTAM` backfitting started from zero for a network on a feature shared with another effect** (`l(x) + n(x)`): it looked for the closed-form contribution under the wrong column name, so the network learned the residual left after that effect and then replaced it; a network on a shared feature was several times worse than the closed-form model. It now starts from the closed-form contribution.
- **`NeuralTAM` networks no longer stall at the start of the training**: a plateau scheduler halved the learning rate while the validation loss was still flat (the last layer starts at zero), so a network with the default training ended up worse than the closed-form effect it replaces (on a line, three times worse). The rate now stays at `lr` and early stopping alone ends the training.
- **Predictions on a frame with duplicate index labels** (`pd.concat` without `ignore_index`), interleaved groups or rows not in date order: every prediction and every `decompose_prediction` effect is placed on its own row (by position, in date order inside each group); the engine used to fail on repeated labels, and `decompose_prediction` put the effects of a group in frame order, wrong when the rows were not in date order.

---

## [1.3.4] - 2026-10-02

Patch release (tag on `main`; the latest release on PyPI and Zenodo stays 1.3.1): `AdaptiveTAM` now gives what an operational refit would, and `StaticTAM` says when a forecast leaves the trained range. **`AdaptiveTAM` forecasts change**; `StaticTAM` changes only for a categorical level absent from training. To get the 1.3.3 behaviour, use the tag `v1.3.3`.

### Changed
- **`AdaptiveTAM` equals a rolling `StaticTAM`** (`fit` on the training rows of a window, `predict` on the next rows, to numerical precision): each window is normalised with its own training rows (it used the whole period), forecasts are no longer clipped to the whole-period range (`clip_to_train_range=True` clips each window to its own), windows are anchored on the start of the data, features and targets are `float64`, and the rows added to balance groups are never trained on. Rows before the first window are `NaN` in `Estimated...` (they were 0); `AdaptedEstimated...` keeps the base model forecast there when there is a base model.
- **`AdaptiveTAM.fit()` then `predict()`** uses the window a refit holds after the last row, with its own normalisation: `fit(history up to day t).predict(day t+1)` equals the simulation's forecast for day t+1.
- **Categorical codes** (`c()` with an `n_cat` given in the formula) are normalised on the full level range `0 .. n_cat - 1`, so a level absent from training keeps its own column instead of landing on an edge level. Codes `1 .. n_cat`, non-integer codes and an inferred `n_cat` keep the min/max rule.

### Fixed
- **Spline knots** (`s()`): set from the full training tensor, no longer from the solver's memory probe, which could cache degenerate knots when the first row of every group sat on the upper edge of its range.

### Added
- **`ta.TAMExtrapolationWarning`**: once per model and feature, when a feature read by a non-linear effect (`s`, `f`, `p`, `w`, `rbf`, `n`, `t`, `phys`, `te` margin) leaves its trained range by more than 1% of its half-range, or a categorical level was not seen in training (`AdaptiveTAM`: one warning for all windows; `AutoTAM` lists them in `summary()`). `l()` never warns; the default extrapolation does not change.
- **`TAM [Info]` at training**: a `c()` term with a given `n_cat` says which levels the training rows lack.
- **`ta.rolling_windows(...)`** yields the `(train_rows, forecast_rows)` of each `AdaptiveTAM` window, to write the reference loop.
- **`AdaptiveTAM(clip_to_train_range=False)`** clips each window's forecast to the range of its own training rows when `True`.

## [1.3.3] - 2026-10-01

Patch release (tag on `main`; the latest release on PyPI and Zenodo stays 1.3.1): `KalmanTAM` no longer looks ahead and updates its state at every step.

**`KalmanTAM` forecasts change**; the only other change is the fix of `s()` on one-row frames and on a single out-of-range value below. Everything else is identical to 1.3.2. To get the 1.3.2 behaviour, use the tag `v1.3.2` (or `block_size=128` for the update rule alone).

### Changed
- **Causal scaling** (`KalmanTAM`): the feature normalisation and the target scale (`y_max - y_min`) were computed on the whole online period, so every forecast depended on observations after it, and the scale set the effective observation and process noise. They now come from a reference period: the first `calibration_steps` rows of each group, or a separate `calibration_data`. A forecast at or before row r no longer changes when later rows change. With data whose range never changes after the reference rows, the forecasts are identical to 1.3.2 (checked to the last bit, with and without a base model).
- **Update at every step** (`KalmanTAM`): with `horizon_steps=1` the default `block_size` was 128, so the state moved only every 128 steps (every 128 days with `group_col="tod"`) while the docstring promised standard online filtering. The default is now the exact sequential filter (`block_size=None`, i.e. 1), which tracks the load more closely on the FORCE national load.
- **`fit()` then `predict()`** (`KalmanTAM`): the frozen state was the one the last row was forecast with, i.e. before that row's update, so forecasting the next day from `fit(history)` ignored the last observation. `fit()` now stores the state after the last update: with `block_size=1`, `fit(history up to day t).predict(day t+1)` equals the online simulation's forecast for day t+1 (checked on the FORCE national load, to numerical precision).

### Fixed
- **`s()` predicted wrongly in two situations** (`StaticTAM`, `AdaptiveTAM`, `KalmanTAM` base features): a spline treated any one-element input as the solver's one-row memory probe and rebuilt its knots from that single point instead of using the trained ones. It happened (1) on a frame with one row per group (tomorrow's forecast for every half-hour with `group_col="tod"`), and (2) with the default linear extrapolation when **exactly one value** of a frame was outside the training range, because the slope of the out-of-range values is computed on a tensor holding only those values (one element for one value). On the FORCE national load test year, a single half-hour just above its group's training maximum was predicted orders of magnitude off. Splines now always use their trained knots. Frames with no out-of-range value, or with several, and frames with several rows per group are unchanged; the other effects (`l()`, `c()`, `f()`, `w()`, `rbf()`, `t()`) were already row-wise.

### Added
- **`KalmanTAM(calibration_steps=None, calibration_data=None)`**: `calibration_steps` is the number of rows per group, at the start of the data, that set the scaling (default `min(365, half the rows)`). A message says those rows are not causal and should not be scored. `calibration_data` takes the scaling from a separate historical DataFrame (for example the training set), so every online row can be scored. A `calibration_steps` that leaves no row to track raises a `ValueError`. The scaling is stored at `fit` and reused by `predict()`.
- **`KalmanTAM.tune_hyperparameters(..., calibration_steps=None, calibration_mask=None)`**: the RMSE is computed on the time steps you choose (a slice, a list of positions or a boolean mask), not on the end of the data.

### Deprecated
- **`tune_hyperparameters(lookback_days=...)`** scored the last `lookback_days * 24` steps of the data you passed, which is look-ahead when that is the period you report (and assumes hourly steps). It still works with a `FutureWarning`, as does calling it with no period; it is removed in 1.5.0.
- An explicit **`block_size > 1`** now warns that the state is updated only every `block_size` steps.

---

## [1.3.2] - 2026-10-01

Patch release (tag on `main`; the latest release on PyPI and Zenodo stays 1.3.1): two crashes fixed. **No prediction changes**: every model that worked in 1.3.1 gives the same numbers.

### Fixed
- **Grouped models predict a frame holding only some of their groups** (`StaticTAM` with `group_col`): `predict` raised `ValueError: Length of values (n) does not match length of index (0)` and `decompose_prediction` returned NaN. The groups stacked from the frame were paired with the fitted group list by position, and the per-group coefficients were sliced by position in the stacked tensor, so a one-group frame would have used the first group's coefficients. Prediction and decomposition now use the groups actually present: rows keep the input order and index, and a group never seen in training raises a `ValueError` naming it. `plot_component` no longer replicates its grid over every group.
- **Silent overflow in log-target quantiles** (`predict_quantile`, `predict_quantiles`, `predict_median`, `anomaly_score`, mixture `component_means` and `predict_mean`): a scale prediction far outside the training range gave `exp(...) = inf` and a silent numpy `RuntimeWarning: overflow encountered in exp`. The value is still `inf` (it is beyond float64, and the columns stay ordered) but it comes with an explicit `UserWarning` naming the quantity and the first rows.

---

## [1.3.1] - 2026-09-30

Patch release: fixes that made some models wrong (formulas with `te()` or several features in one term) or not reproducible (`rbf()`, trees).

**Predictions change** for the models below; everything else is identical to 1.3.0. To get the 1.3.0 behaviour: `pip install tam-ml==1.3.0`.

### Changed
- **Formulas with `te()` or several features in one term**: they were fitted with mislabelled terms and now fit the formula as written; their errors drop sharply, and the ensembles built on them follow. The THEORY benchmark and its figures are regenerated with 1.3.1.
- **`auto_fit` (GCV)**: now fits exactly the model GCV selected. GCV minimises an in-sample criterion, not the holdout error, so holdout results move in both directions.
- **`rbf()`**: centres are seeded and drawn from the whole training set; RBF models change once and are then reproducible.
- **Trees on grouped data** (`t()`, `lt()` with `group_col`): splits and leaf counts come from the whole training set; results change, mostly for the better. Ungrouped trees do not change.
- **NeuralTAM** (experimental): may change slightly in scripts that fit `rbf()` models before it, because both used to share the global torch random generator. A NeuralTAM fitted on its own after the same `torch.manual_seed` is unchanged.
- Fixed-penalty models without these terms are identical to 1.3.0.

### Added
- **`tam.plot_component(model, data, component, kind="auto")`**: plots one additive component with a view chosen from its dimension: a curve for one feature, one curve per level for `te(x, c)`, a 3D surface (or `kind="heatmap"` with the observed points overlaid) for `te(x1, x2)`, evaluated on a regular grid for one group (`group=`, default the most frequent), a 3D scatter coloured by the contribution for `te(x1, x2, x3)`. Rows with a non-finite contribution are skipped.
- **`tam.common.plotting.resolve_component(model, feature, component=None, color_by=None)`**: the component a feature maps to under the new decomposition names.
- **`rbf(x, ..., seed=42)`**: seed of the centre sampling (default 42, like `n()` and `t()`).

### Fixed
- **Terms renamed by position** (`StaticTAM._prepare_data`): each effect's `feature_name` was overwritten from the deduplicated feature list by position, so with a `te()` or several features in one term, the effects after it were relabelled (e.g. `c(day_type_week)` became `toy`) and **the fitted model changed**, not only the labels (e.g. a national-load model with `te(temperature, toy)`: test RMSE 3254 → 2432). The renaming is removed. `decompose_prediction` names components with the new `decomposition_names()`: unique feature names are kept, collisions get a basis prefix (`s_x`, `l_x`), and a collision that remains (two `te()` over the same features) gets an occurrence suffix, so no contribution overwrites another.
- **Plotting helpers after the renaming fix**: `plot_effect_with_model_and_data` and `plot_effect_with_data_decomposed` raised `KeyError: 'effect_x1'` for a feature inside a `te()` or used by several effects. They now resolve the feature to its component: the only one using it, the tensor product whose other margin is `color_by`, or the new `component=` argument; a still-ambiguous feature raises a `ValueError` listing the candidates. Single-component features plot exactly as before.
- **`add_base_effects` in AdaptiveTAM and KalmanTAM**: the base model's components were added as `l(effect_<feature>)`, a name that does not exist when two effects share a feature (`s(x) + l(x)` gives `effect_s_x`, `effect_l_x`), so the model crashed with a `KeyError`; and with a `te()`, the renaming bug above fed mislabelled components. The components are now added under their `decomposition_names()` columns.
- **GCV scored a different penalty from the one it stored** (`auto_fit`, `smart_solve_gcv`): each trial rescaled a block that already carried the formula's own weight, so the search scored λ_formula × λ_GCV but stored λ_GCV alone; a later `fit()` on the selected weights returned a different model, and with the default `ap=-9` the nine highest decades of the search range were unreachable. Each trial block is now rebuilt by the effect at λ = 10^α, exactly as `fit()` assembles it; the trial weight is restored even if the search raises, and the initial alphas are clipped to the bounds. **Models selected by `auto_fit` change**.
- **GCV scored a slightly different system from the one fitted** (`auto_fit`): the solver adds a ridge floor `1e-6·n·I`, GCV added `1e-6·I`. Both now use one helper (`_ridge_floor`), so the selected penalties are scored on the model `fit()` returns; a negative residual sum of squares from rounding is clamped at 0 instead of folded with `abs()`. Penalties below `ap = -6` are dominated by that floor (documented). **Models selected by `auto_fit` can change**; fixed-penalty fits do not.
- **Trees and RBF centres were initialised on a memory probe, not on the training data** (`t()`, `lt()`, `rbf()`): their data-dependent state was set by the first design matrix built, which is the solver's size probe (one row per group). With 48 half-hourly groups, quantile splits came from 48 points and the `sp_alpha` leaf counts summed to 48 × n_trees instead of every training row; `rbf()` centres were drawn from those 48 points. `StaticTAM` now calls `initialize_effects()` on the full training tensor before any design matrix; a probe or a later chunk never changes that state. **Tree, linear-tree and RBF predictions change for grouped models**; ungrouped trees were already initialised on the full data and do not change (ungrouped `rbf()` changes through its new seed only).
- **`rbf()` centres were not reproducible**: they were drawn from the global torch generator, so an RBF model changed with whatever code ran before it (two identical fits could differ). `rbf(x, ..., seed=42)` now seeds a local generator, and fitting no longer touches the global random state. **RBF predictions change once** (new, fixed centres).
- **Sparsity-adaptive tree penalty** (`t(..., sp_alpha>0)`): `empirical_counts` summed only the first batch axis, so the leaf-density penalty was built from the wrong counts and could not be formed. It now counts every sample and group, one value per leaf in design-matrix order.
- **Linear tree weight** (`lt()`): assigning `lambda_p` (as GCV does on every candidate) did not reach the intercept tree and the slope surface, so GCV could not tune `lt()`. The weight now propagates to both sub-blocks.
- **Dummy date overflow**: without `date_col`, the internal dummy date was spaced one day apart and ran past the year 2262 after ~95,000 rows, which overflows pandas 2.x nanosecond datetimes (pandas 3 tolerates it). It is now spaced one second apart.
- **CI**: the test workflow runs on every push and pull request (any branch) and on demand, and checks `import tam` first on every Python version (3.10-3.14).

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

### Added

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

### Changed

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

### Added

* **Evolutionary Orchestrator (`AutoTAM`)**: A multi-fidelity AutoML engine for automated GAM discovery. It utilizes a Hub-and-Spoke evolutionary architecture, strict topological sanitization, and bi-level optimization (GPU MSP-GCV) to solve the combinatorial explosion of adaptive models, ultimately deploying orthogonal experts into a Dual OPERA arena.

* **OPERA (`OperaTAM`)**: A new expert aggregation meta-learner featuring a fast GPU implementation natively optimized via `@torch.jit.script`.
* **Kalman Filter (`KalmanTAM`)**: Dynamically tracks coefficient drift over time via a Fast Dynamic Extended Kalman Filter (EKF), highly optimized using the Woodbury matrix identity (reducing inversion complexity to $\mathcal{O}(T_{block}^3)$) and compiled with `TorchScript`.
* **DeepGAM (`NeuralTAM`)**: A new Deep-GAM hybrid model (Additive + Deep Learning) implementing Group-wise Orthogonal Backfitting.
* **Tree Effect (`TreeEffect`)**: Added the Tree / Random Forest effect (`t(...)`) designed for GPU, based on Oblivious Random Trees and Random Binning Features approximation.
* **Hardware Manager (`HardwareManager`)**: Centralized hardware abstraction layer to dynamically manage the capabilities of different compute backends.
* **`_dispatcher.py` (Mathematical Solver Dispatcher)**: Created an intelligent routing layer between statistical modeling abstractions and PyTorch linear algebra engines. It dynamically routes resolution to either a chunked direct solver or a Matrix-Free Sparse Conjugate Gradient (CG) solver based on topological complexity and available VRAM.
* **`_memory.py` (Hardware Memory Management and Estimation)**: Completely isolated low-level hardware interactions into a dedicated module. It estimates the byte footprint of massive matrices and calculates safe algorithmic chunk sizes.

### Changed

* **Neural Effect Improvements (`NeuralEffect`)**: Added support for multiple hidden layers to project variables into higher dimensions.
* **Native GPU Acceleration**: Complete migration of intensive CPU to GPU calculation for Splines (`s(...)`), Wavelets (`w(...)`), and RBF (`rbf(...)`) effects, improving performance of design matrix construction.
* **Memory Management (Safeguards & Smart Chunking)**: 
    * Overhauled memory safety to prevent and correct CUDA Out of Memory bugs via a smart chunking system.
    * Implemented a memory safeguard for CPU / group-chunking by independent series.
    * Established strict dynamic RAM & VRAM safety margins to guarantee the stability of large matrix inversion operations.

---

## [Internal] 1.2.1 - 2025-12-18 (Not publicly released)

This version represents a complete architectural overhaul, introducing advanced functional bases (Spectrum), Conformal Prediction, and a full benchmark suite.

### Added

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

### Changed

* **Math Engine (Primal Solver):** Formally validated the exact **Primal Ridge Solver** utilizing block-diagonal covariance accumulation. Corrected performance tracking to accurately reflect the framework's time complexity of $\mathcal{O}(G \times T \times D^2 + G \times D^3)$, ensuring isolated mathematical resolution per group $G$.
* **Effect Architecture:** Refactored the core around `BaseEffect`, establishing the `List[BaseEffect]` as the standard configuration.
* **Modularization:** Monolithic `_effects.py` was entirely split into the `spectrum` package, improving modularity.
* **Normalization Domain:** Changed global feature normalization from the Fourier-centric $[-\pi, \pi]$ to the strictly orthogonal **$[-1, 1]$** domain in `_data.py`. Basis functions now apply internal scaling (e.g., Fourier rescales to $[-\pi, \pi]$).
* **Decomposition Robustness:** Implemented **collision detection** in `_math.py` to automatically prefix feature effects (e.g., `l_time`, `s_time`) when multiple bases share the same input variable.

### Fixed

* **Recursive Parsing:** Implemented an architectural fix in `parse_formula_to_terms` to correctly **identify and preserve string tokens** (like `ga_te` or `grid_k`) during the recursive parsing of `te(...)` terms.
* **Syntax Stability:** Converted all docstrings containing LaTeX math commands to **raw strings** (`r"""..."""`) to eliminate Python `SyntaxWarning`s.

---

## [Internal] 1.1.1 - 2025-11-21 (Not publicly released)

This version introduced the Formula API and the first object-oriented refactoring.

### Breaking Changes
- Removed legacy dictionary-based API (`m_orders`, `s_orders`, `alpha_list`)
- Introduced formula-based API as the primary interface

### Added

* **Formula-based API (`model/additive.py`):** Implemented a new, intuitive R-like formula API (e.g., `Load ~ s(temp, k=10) + l(day_type)`) as the new standard for model initialization.
* **Spline Effects (`model/_effects.py`):** Added `SplineEffect` (P-splines) as a new core effect type, available via `s(...)`.
* **Formula Parser (`common/utils.py`):** Added a `parse_formula_to_terms` function to support the new API.
* **`StaticTAM` & `AdaptiveTAM`:** Implemented the full object-oriented API (`.fit()`, `.predict()`) and the online error correction model.
* **Multi-Start Grid Search:** The `grid_search_fit` method now uses a **Multi-Start Coordinate Descent** strategy (Conservative, Median, Aggressive) to avoid local minima.
* **`diagnostics` Module:** Added a module for model analysis, including t-tests and feature importance visualization.

### Changed

* **Legacy API Removed:** Removed the old `m_orders`, `s_orders`, `alpha_list` dictionary-based configuration from `v0.0.6`.
* **Package Structure:** The codebase was refactored into a modular package structure (`common`, `model`).
* **Internal Math:** Math functions (`_math.py`) were cleaned of all effect-specific logic and made robust to 2D/3D tensor inputs.
* **Hardcoded Names Removed:** Removed dependencies on specific column names (`tod`, `timestamp`, `Load`).

---

## [0.0.6] - 2025-05-27

### Added
* Initial project setup based on the original `weakl` v0.0.6 package.

[Unreleased]: https://github.com/EDF-Lab/tam/compare/v1.4.1...HEAD
[1.4.1]: https://github.com/EDF-Lab/tam/releases/tag/v1.4.1
[1.4.0]: https://github.com/EDF-Lab/tam/releases/tag/v1.4.0
[1.3.4]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.4
[1.3.3]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.3
[1.3.2]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.2
[1.3.1]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.1
[1.3.0]: https://github.com/EDF-Lab/tam/releases/tag/v1.3.0
[1.2.6]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.6
[1.2.5]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.5
[1.2.4]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.4
[1.2.3]: https://github.com/EDF-Lab/tam/releases/tag/v1.2.3
[0.0.6]: https://pypi.org/project/weakl/0.0.6/