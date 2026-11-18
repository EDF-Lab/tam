# Form B Temporal Operators: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See TEMPORAL_OPERATOR.md](../../math/spectrum/TEMPORAL_OPERATOR.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

Form B Temporal Operators wrap an inner functional basis (`'spline'`, `'fourier'`, or `'linear'`) and apply a causal time-mixing operator directly to the evaluated inner basis matrix $\Phi_{\text{base}}$:

```python
# Causal exponential attention operator over B-spline basis
formula = "Y ~ op_attn(temperature, inner='spline', k_base=6, decay=0.92)"

# Causal Legendre HiPPO operator over Fourier or B-spline basis
formula = "Y ~ op_hippo(Load_d1, inner='spline', k_base=6)"

# Causal Hankel SSA operator over B-spline basis
formula = "Y ~ op_ssa(Load_d1, inner='spline', k_base=6, window=12)"

# Stateful out-of-sample boundary continuation
preds = model.predict(df_test, continue_state=True)
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `op` | `str` | `'attn'` | Operator type: `'attn'`, `'hippo'`, or `'ssa'`. |
| `inner` | `str` | `'spline'` | Inner basis family: `'spline'`, `'fourier'`, or `'linear'`. |
| `k_base` | `int` | `8` | Inner basis dimension $K$. |
| `decay` | `float` | `0.92` | Exponential decay factor $\gamma \in (0, 1)$ for `'attn'`. |
| `window` | `int` | `12` | Trailing Hankel window length for `'ssa'`. |

## Complexity & Performance

- **End-to-End Statefulness**: `StaticTAM` caches `_effect_states_` after fitting and passes `state_in` into `TemporalOperatorEffect` when `predict(..., continue_state=True)` is called.
- **Penalty Preservation**: Reuses the exact $K \times K$ inner penalty matrix $P_{\text{base}}$ (e.g., $D_2^\top D_2$ for splines), keeping the normal equations $\mathcal{O}(T K^2)$.

## State, Complexity and Limits

- **Class and file**: `TemporalOperatorEffect`, `src/tam/model/spectrum/_temporal_operator.py`. Label: (BETA).
- **State** (`stateful`): `attn` `{"S": (B, m, K), "Z": (B, m)}`, `hippo` `{"h": (B, K)}`, `ssa` `{"hist": (B, L - 1, K)}`.
- **Layout**: `(B, T, d)` (`keeps_feature_dim`); `(T,)` is one series. Only the first column feeds the inner basis; `others=` columns are read (they enter `extrapolating_features`) but not used by the three operators.
- **Time and memory**: $\mathcal{O}(B\,T\,m\,K)$ for `attn`, $\mathcal{O}(B\,T\,K)$ for `hippo`, $\mathcal{O}(B\,T\,L\,K)$ for `ssa`; no $T \times T$ matrix. Fifty thousand steps run in a few seconds.
- **Penalty coordinates**: `base` and `op`.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction; the inner basis is pointwise and keeps its own rule.
- **Smoothing criterion**: the effect has several penalty coordinates (see above). Prefer `auto_fit(criterion="reml")` to the default coordinate GCV. Signature: `auto_fit(self, data_train, alpha_p_bounds=(-30.0, 6.0), number_of_steps=10, alpha_p_list=None, gamma=1.4, verbose=False, criterion="gcv")`. This is a recommendation, not a measured result for this effect: the repository's REML tests check the analytic gradient, the penalty basis of the multi-coordinate effects and that REML is more stable than GCV on autocorrelated noise (`test_reml_criterion`); no benchmark compares the two criteria on this effect.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `TemporalOperatorEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_temporal_operator.py
:language: python
:start-after: "#: <temp_op_feature_map>"
:end-before: "#: </temp_op_feature_map>"
```

Penalty of `TemporalOperatorEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_temporal_operator.py
:language: python
:start-after: "#: <temp_op_penalty_matrix>"
:end-before: "#: </temp_op_penalty_matrix>"
```

Parsing of the `op_attn / op_hippo / op_ssa(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_temporal_operator>"
:end-before: "#: </parse_temporal_operator>"
```
