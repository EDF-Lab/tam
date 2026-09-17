# AutoTAM Probabilistic Selection: Engineering Implementation Plan

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/11_autotam_probabilistic.md)
* **Related AutoTAM architecture:** [Orchestrator Code](08_auto_orchestrator_code.md)

> Mirror-Architecture companion outlining the planned engineering for `math/meta/11_autotam_probabilistic.md`.
> This document specifies the "how": where each planned modification change will land within the codebase and the strict engineering constraints it must follow. It serves as an implementation roadmap and does not re-derive the underlying math.

---

## 1. Planned Mode Routing (No New Genome)

The modifications will introduce an `objective` axis rather than adding a second genome branch. The `DragTAM._evaluate_candidate` method will be updated to construct the per-candidate model based on the active mode:

* **Legacy Point Path (`objective="point"`):** Will route to `StaticTAM(formula, loss="l2")`. This path will remain completely unchanged and bit-identical to current releases.
* **Probabilistic Path (`objective="quantile"`):** For each requested expectile level `τ`, it will route to `StaticTAM(formula, loss="expectile", loss_kwargs={"tau": expectile_level_for_quantile(τ)})`.
* **Architecture Re-use:** The structural `formula` (the genome's right-hand side) will remain identical across all levels; only the loss and expectile level will differ. The engine will explicitly reuse the existing Iteratively Reweighted Least Squares (IRLS) schedule over the `BaseTAM._solve_pwls_step` atom, meaning no new mathematical solver needs to be written.

---

## 2. Planned Vectorized Ablation Importance (Pruning)

The `KnowledgeGraph.update_and_prune` method will gain a new probabilistic branch to execute parsimony pruning. Since the additive contributions are already materialized and cached via a single call to `model.decompose_prediction(df)`, the ablation will be implemented as a highly optimized, **vectorized** tensor operation rather than a costly refit:

```python
# C represents contributions: shape [N, L] (per-term centred effect), where f = C.sum(1)
# R_tau(f) represents the pinball risk of the full surface on the D_dev dataset
# R_tau(f - C_l) subtracts one column yielding an [N] residual shift, reusing the pinball kernel

I = (R_tau(f[:,None] - C) - R_tau(f)) / (R_tau(f) + eps)  # Yields [L] ablation importances
# Prune terms where I < gamma

```

* **Performance Guardrail:** There will be no per-term refit and strictly no Python loops over the terms; the `[N, L]` subtraction will be broadcasted natively.
* **Fallback Mechanism:** The system will gracefully fall back to the legacy `Var(h_l)/Var(f)` calculation whenever `loss == "l2"`.

---

## 3. Planned UCB Reward Engineering

The `DragTAM.optimize` method will be updated to compute the new skill-score reward evaluated against the unconditional-$\tau$-quantile baseline, denoted as `R_τ(f₀)`.

* To ensure optimal computational efficiency, this scalar baseline will be computed exactly once per fold from the `D_dev` block and cached directly on the pipeline context.
* This new reward scaling will replace the legacy `rmse/target_std` formula exclusively on the probabilistic path.

---

## 4. Planned CQR Calibration (Intervals)

Calibration will be executed on the minimax holdout block (`D_val`) immediately following OPERA aggregation. The implementation steps will be:

1. **Gathering:** Extract the aggregated `q̂_lo` and `q̂_hi` arrays directly from the per-$\tau$ OPERA fans.
2. **Error Calculation:** Compute `E = np.maximum(q_lo - y, y - q_hi)` vectorially on the `D_val` block.
3. **Quantile Extraction:** Calculate `Q = conformal_quantile(E, 1-alpha)` utilizing the existing primitive within `SafetyTAM` (developers are strictly instructed not to hand-roll this quantile calculation).
4. **Serving Predictions:** The final static intervals will be served as `[q_lo - Q, q_hi + Q]`. Streaming deployments will route through `risk.aci.adaptive_conformal_intervals`.
5. **Monotonicity Enforcement:** To prevent crossing quantiles, the engine will rearrange the fan prior to calibration using a single vectorized call: `np.sort(q_fan, axis=1)`. Operating on an `[N, K]` array of quantile columns, this runs in $\mathcal{O}(NK \log K)$ within the compiled C++ backend. Python loops over rows are strictly banned, as they would cost one interpreter iteration per holdout row.

*Note:* The existing `StaticTAM.calibrate_conformal` and `predict_intervals` (the point-residual path) will be constrained to run **only** for `objective="point"` champions where `needs_conformal=True`.

---

## 5. Strict Engineering Constraints (Carried Over)

Per `CONTRIBUTING` §3, the implementation must adhere to the following hardware safety and performance guidelines:

* **No Python Loops:** Iterating over rows or terms is prohibited; operations must utilize `[N, L]` or `[N, K]` tensor broadcasting.
* **OOM Safety:** Ablation logic must reuse the already-allocated contribution tensors to prevent a catastrophic `[N, L, L]` Out-Of-Memory blow-up.
* **Chronological Alignment:** The $\tau$-fits and calibration processes must inherit the existing `date_col` sorting guarantees; absolutely no new tensor stacking sequences may be introduced.
* **Regression Guard:** The `objective="point"` path must remain the default and must stay mathematically bit-identical, safeguarded by the existing `_ladder.py` point test cases.

---

## 6. Planned File Touch List

To successfully implement the modifications, they are scheduled for the following files:

* `evaluation/metrics.py`
* `pipeline/context.py` (to add `ForecastObjective` and the baseline `R_τ(f₀)` cache)
* `data_profiler.py` (to serve as an advisory gate)
* `parser.py` (to handle `quantiles=` and `interval=` arguments)
* `drag_tam.py` (to integrate mode routing, the ablation prune, and the skill reward)
* `pipeline/expert_expander.py` (to generate per-$\tau$ quantile experts)
* `pipeline/ensemble_selector.py` (to manage per-$\tau$ OPERA aggregation and CQR)
* `evaluation/autotam_report_generator.py` (to add reliability and fan visualization panes)
* `auto_tam.py` (to expose the `predict_quantiles` and `predict_intervals` APIs)
