# Path Signature and Log-Signature Effect: PyTorch Architecture & Implementation

**Navigation:**
  * **Mathematical Theory:** [See SIGNATURE.md](../../math/spectrum/SIGNATURE.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

The Path Signature effect is exposed in formulas via the `sig()` token:

```python
# Univariate sliding-window path signature with time augmentation
formula = "Y ~ sig(Temperature, w=9, N=3)"

# Compressed Log-Signature on the Free Lie algebra via Lyndon basis
formula = "Y ~ sig(Temperature, w=9, N=3, log_sig=True)"

# Cross-channel causal lead-lag discovery with Hoff's transform
formula = "Y ~ sig(Temperature, others='Load_d1', w=9, N=2, lead_lag=True)"

# Combined with non-linear spline and tensor interaction
formula = "Y ~ s(Temperature, k=10) + sig(Temperature, w=7, N=3) + te(sig(Temperature, w=5, N=2), c(day_type_week))"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Primary feature column name. |
| `w` | `int` | `9` | Sliding window length in rows/steps. |
| `N` | `int` | `3` | Iterated integral truncation level (maximum 4). |
| `lag` | `int` | `0` | Delay rows applied before the end of the sliding window. |
| `time_augment` | `bool` | `True` | Appends normalized time coordinate $\tau \in [0, 1]$ across the window ($D = d + 1$). |
| `log_sig` | `bool` | `False` | Compresses signature into Free Lie algebra via tensor log and Lyndon basis. |
| `lead_lag` | `bool` | `False` | Applies Hoff's Lead-Lag embedding ($2w - 1$ steps in $\mathbb{R}^{2d}$) for quadratic covariation. |
| `others` | `str` / `list[str]` | `None` | Pipe-delimited or list of additional multivariate feature columns. |
| `ap` | `float` | `default_alpha_p` | Log10 of the regularization parameter $\lambda_p = 10^{\text{ap}}$. |
| `extrapolate` | `str` | `'continue'` | OOD extrapolation strategy beyond $[-1, 1]$. |

## Complexity & Performance

- **Memory Complexity**: The feature map requires $\mathcal{O}(T \cdot K)$ floats where $K = \sum_{k=1}^N D^k$ for signatures or $\sum_{k=1}^N \beta(D, k)$ for log-signatures.
- **Compute Complexity**: Each window of length $w$ requires $w - 1$ segment exponentials and Chen tensor products, executing in $\mathcal{O}(T \cdot w \cdot D^N)$.
- **Free Lie Algebra Compression**: Setting `log_sig=True` reduces dimensionality from $\frac{D^{N+1}-1}{D-1}$ to the Lyndon basis dimension $\sum_{k=1}^N \beta(D, k)$, eliminating shuffle-product collinearity and improving design matrix conditioning.
- **Factorial Penalty**: Level-dependent factorial scaling $(k!)^2$ regularizes higher-order terms proportionally to their natural decay scale.

## State, Complexity and Limits

- **Class and file**: `SignatureEffect`, `src/tam/model/spectrum/_signature.py` (iterated-integral algebra in `_signature_ops.py`). Label: (BETA).
- **State**: `{"hist": (B, w - 1 + lag, d)}`, the last rows of every group, detached. `StaticTAM.predict(data, continue_state=True)` passes it back; a state with another number of groups raises a `ValueError`.
- **Layout**: `(B, T)` for one column, `(B, T, d)` with `others=`; one signature per step and per group.
- **Time and memory**: $\mathcal{O}(B\,T\,w\,D^{N})$ time and $\mathcal{O}(B\,T\,K)$ memory ($K$ = number of coefficients); the windows of all groups are processed as one batch, with no Python loop over groups.
- **Extrapolation**: the effect is temporal, so `'linear'` and `'saturation'` are refused at construction; use `'continue'` (default) or `'constant'`.
- **Limits**: the truncation level is at most 4; the windows read the standardized increments of the training period, so a regime far from the training range gives large (but finite) coefficients.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `SignatureEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_signature.py
:language: python
:start-after: "#: <sig_feature_map>"
:end-before: "#: </sig_feature_map>"
```

Penalty of `SignatureEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_signature.py
:language: python
:start-after: "#: <sig_penalty_matrix>"
:end-before: "#: </sig_penalty_matrix>"
```

Parsing of the `sig(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_signature>"
:end-before: "#: </parse_signature>"
```
