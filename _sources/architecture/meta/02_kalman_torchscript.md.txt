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

**Base-model components (verified by `tests/test_base_effect_columns.py`):** with `add_base_effects=True`, one `l(effect_<name>)` term per base-model component is appended to the Kalman formula, with the column names `decompose_prediction` emits (`decomposition_names()`).
