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

**Base-model components (verified by `tests/test_base_effect_columns.py`):** with `add_base_effects=True`, one `l(effect_<name>)` term per base-model component is appended to the Kalman formula, with the column names `decompose_prediction` emits (`decomposition_names()`).
