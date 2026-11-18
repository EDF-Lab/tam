# Universal Foundation Basis: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See FOUNDATION.md](../../math/spectrum/FOUNDATION.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

The foundation basis is exposed as the composite token `foundation()`, which carries its components internally (`PatchBasisEffect`, `QuantileHistEffect`, `MultiRateFilterbankEffect`, `StructuralPriorEffect`; the components are not formula tokens):

```python
# Composite universal foundation basis
formula = "Y ~ foundation(temperature, others='Load_d1', patch_len=12, n_tokens=8, n_bins=8, scales='1|4|16', n_struct=8)"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `others` | `str` or `list` | `None` | Secondary features separated by `\|` (read by the patch basis Lévy areas). |
| `patch_len` | `int` | `16` | Causal sliding patch length $P$ (the histogram window is `max(patch_len, 8)`). |
| `n_tokens` | `int` | `16` | Number of random-phase orthogonal patch tokens $M$. |
| `include_levy` | `bool` | `True` | Append Lévy-area hysteresis tokens. |
| `n_bins` | `int` | `16` | Number of quantile histogram bins $B$. |
| `scales` | `Sequence[int]` or `str` | `(1, 4, 16, 64)` | Time scales of the multi-rate filterbank. |
| `n_struct` | `int` | `16` | Size of the structural prior dictionary. `n_pfn` is accepted as an alias; giving both with different values raises an error. |
| `seed` | `int` | `42` | Seed of the patch projections. |

## State, Complexity and Limits

- **Class and file**: `FoundationEffect`, `src/tam/model/spectrum/_foundation.py`. Label: (BETA).
- **State**: `{"patch": (B, P - 1, D), "qhist": (B, W - 1, 1), "ema": (B, S), "env": (B, S)}`.
- **Layout**: `(B, T, D)` (`keeps_feature_dim`); the primary column feeds all blocks, the other columns only the patch basis.
- **Time and memory**: linear in $T$ for every block (the filterbank is a pair of linear recurrences, not a $T \times T$ operator): fifty thousand steps run in a few seconds.
- **Penalty coordinates**: `patch`, `hist`, `mrfb`, `struct`; `set_penalty_coordinates` accepts four values or one.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction.
- **Closed form**: no network is trained; the fit is one penalized weighted least-squares solve.

- **Smoothing criterion**: the effect has several penalty coordinates (see above). Prefer `auto_fit(criterion="reml")` to the default coordinate GCV. Signature: `auto_fit(self, data_train, alpha_p_bounds=(-30.0, 6.0), number_of_steps=10, alpha_p_list=None, gamma=1.4, verbose=False, criterion="gcv")`. This is a recommendation, not a measured result for this effect: the repository's REML tests check the analytic gradient, the penalty basis of the multi-coordinate effects and that REML is more stable than GCV on autocorrelated noise (`test_reml_criterion`); no benchmark compares the two criteria on this effect.
- **Penalty ridge**: `_build_d2_penalty(n)` is `D2.T @ D2 + 1e-6 * I`; the ridge is kept on purpose: it makes the second-difference blocks positive definite so that the REML log-determinant is defined, and it is part of the numbers of the benchmarks.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `FoundationEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_foundation.py
:language: python
:start-after: "#: <foundation_feature_map>"
:end-before: "#: </foundation_feature_map>"
```

Penalty of `FoundationEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_foundation.py
:language: python
:start-after: "#: <foundation_penalty_matrix>"
:end-before: "#: </foundation_penalty_matrix>"
```

Parsing of the `foundation(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_foundation>"
:end-before: "#: </parse_foundation>"
```
