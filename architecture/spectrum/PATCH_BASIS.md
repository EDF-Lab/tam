# Causal RevIN Patch Basis: PyTorch Architecture

**Navigation:**
  * **Mathematical Theory:** [See PATCH_BASIS.md](../../math/spectrum/PATCH_BASIS.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Tokens

The causal patch basis is exposed via `patch_basis()`:

```python
# Causal RevIN patch basis
formula = "Y ~ patch_basis(temperature, others='Load_d1', patch_len=12, n_tokens=12, include_levy=True)"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `others` | `str` or `list` | `None` | Secondary features separated by `\|` (used by `patch_basis` Lévy areas). |
| `patch_len` | `int` | `16` | Causal sliding patch length $P$. |
| `n_tokens` | `int` | `16` | Number of random-phase orthogonal patch tokens $M$. |
| `include_levy` | `bool` | `True` | Append Level-2 antisymmetric Lévy area hysteresis tokens for multivariate inputs. |

## Complexity & Performance

- **Zero future leakage**: the instance statistics are those of the patch ending at the current step.

## State, Complexity and Limits

- **Class and file**: `PatchBasisEffect`, `src/tam/model/spectrum/_patch_basis.py`. Label: (BETA).
- **State**: `{"hist": (B, patch_len - 1, D)}`. **Layout**: `(B, T)` for one column, `(B, T, D)` with `others=`.
- **Time and memory**: $\mathcal{O}(B\,T\,P\,M)$ for the tokens and $\mathcal{O}(B\,T\,D\,P)$ for the Lévy areas; all groups in one batch.
- **Penalty**: ridge weighted by filter roughness on the $M$ tokens, plain ridge on the other columns (one coordinate; inside `foundation` it is the `patch` coordinate).
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction; the tokens are bounded, so extreme values saturate.
- **Limits**: a pass shorter than the patch uses the replicated first row as history.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `PatchBasisEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_patch_basis.py
:language: python
:start-after: "#: <patch_basis_feature_map>"
:end-before: "#: </patch_basis_feature_map>"
```

Penalty of `PatchBasisEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_patch_basis.py
:language: python
:start-after: "#: <patch_basis_penalty_matrix>"
:end-before: "#: </patch_basis_penalty_matrix>"
```

Parsing of the `patch_basis(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_patch_basis>"
:end-before: "#: </parse_patch_basis>"
```
