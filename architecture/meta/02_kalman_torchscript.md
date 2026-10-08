#  Kalman TorchScript Optimizations & Block Updates [BETA]

> 🚧 **Beta – Research Module**

⚠️ **Status: Active development**

- Core functionality is implemented but still evolving  
- Some features may be incomplete or subject to change  
- API is not yet stable  
- Intended for research use only (not production-ready)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/02_kalman_filter.md)

**Causal scaling (verified by `tests/test_kalman_causal.py`):** `KalmanTAM.prepare_data` computes the tracked features, then `_fit_reference` fixes the feature normalisation (fitted by the internal `StaticTAM` extractor on the reference rows only) and stores the target centre and scale per group (`scale_dict_` for `predict()`); `_design` builds the design matrix with that normalisation. The reference is the first `calibration_steps` rows of each group (burn-in message, `calibration_steps_used_`) or `calibration_data`. `block_size=None` means 1, the exact sequential filter. The TorchScript kernel returns the state after the last update (`final_state_`), which `fit` stores as `last_state_dict_` for `predict()`.

**Process noise per term (verified by `tests/test_kalman_per_term_noise.py`):** the TorchScript kernel takes the diagonal of $Q$ as a tensor (`process_noise_diag`, one variance per design column) instead of a scalar. `KalmanTAM._term_spans` maps each formula term (the names of `decomposition_names`, plus `"offset"` for the two constant columns: the filter's systematic offset and the formula's intercept) to its design columns, and `_noise_diagonal` fills the diagonal from `process_noise_var` (a float, or a dict `{term: q, "offset": q, "default": q}`). The boost on the offset's noise (`offset_q_boost`) is `offset_boost` unless the offset has a variance of its own; the initial covariance of the offset keeps `offset_boost`. A float call builds the same diagonal as before, so its forecasts are unchanged.

**Base-model components (verified by `tests/test_base_effect_columns.py`):** with `add_base_effects=True`, one `l(effect_<name>)` term per base-model component is appended to the Kalman formula, with the column names `decompose_prediction` emits (`decomposition_names()`).

## Predictive quantiles

The compiled loop can also return the predictive variance of every forecast row (`with_variance`, off by default: the filter and its outputs are then identical to before). Per block it adds one batched product `X P Xᵀ` on the a priori covariance, and for `horizon_steps > 1` the covariance of the delayed state grown by `(h - 1) Q`, written at the row it forecasts. `_run_filter` keeps it in `_predictive_variance_`; `_online` is the former body of `predict_online`, which now calls it, and `predict_quantiles` converts the standard deviation to the target scale with the reference scale and adds `z_tau` standard deviations to the online forecast:

```{literalinclude} ../../../../src/tam/model/kalman.py
:language: python
:start-after: "#: <kalman_quantiles>"
:end-before: "#: </kalman_quantiles>"
:caption: src/tam/model/kalman.py (Gaussian quantiles from the filter variance)
```

The constructor and the outputs of `predict_online`, `predict` and `fit` are unchanged. One limit of the filter carries over: a row without a target leaves the state unchanged but still reduces the covariance, so the variance after a long gap of missing targets is understated.
