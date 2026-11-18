# Transformer FAVOR+ Attention Effect: PyTorch Architecture & Implementation

**Navigation:**
  * **Mathematical Theory:** [See TRANSFORMER_FAVOR.md](../../math/spectrum/TRANSFORMER_FAVOR.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

The Transformer attention effect is exposed in formulas via the `attn()` token:

```python
# Univariate causal multi-head self-attention
formula = "Y ~ attn(Load_yesterday, m_features=16, n_heads=4)"

# Multi-scale causal decay with custom decay rates
formula = "Y ~ attn(Temperature, m_features=16, n_heads=3, decay=(1.0, 0.95, 0.80))"

# Multivariate interaction using others
formula = "Y ~ attn(Load_yesterday, others='Temperature|NetLoad', m_features=32, n_heads=4)"

# Tensor product interaction with spline
formula = "Y ~ te(attn(Load_yesterday, m_features=16, n_heads=2), s(Temperature, k=8))"
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Primary feature column name. |
| `m_features` | `int` | `16` | Number of positive random features $m$. |
| `n_heads` | `int` | `4` | Number of multi-scale attention heads $H$. |
| `causal` | `bool` | `True` | Whether attention is strictly causal ($\tau \le t$). |
| `decay` | `tuple[float]` | `(1.0, 0.99, 0.95, 0.85)` | Exponential memory decay rates $\gamma_h \in (0, 1]$ per head. |
| `mode` | `str` | `'head'` | Output representation: `'head'` ($\mathbb{R}^{T \times (H \cdot d)}$) or `'bilinear'` ($\mathbb{R}^{T \times (m \cdot d)}$). |
| `orthogonal` | `bool` | `True` | Whether to use Orthogonal Random Features (ORF) via Mezzadri QR for variance reduction. |
| `seed` | `int` | `42` | Deterministic PRNG seed for the random projection matrix. |
| `others` | `str` / `list[str]` | `None` | Pipe-delimited or list of additional multivariate feature columns. |
| `k` | `int` | `None` | Total coefficient budget ($H \cdot d$). Overrides `n_heads` if provided. |
| `ap` | `float` | `default_alpha_p` | Log10 of the regularization parameter $\lambda_p = 10^{\text{ap}}$. |
| `extrapolate` | `str` | `'continue'` | OOD extrapolation strategy beyond $[-1, 1]$. |

## Complexity & Performance

- **Memory Complexity**: The feature map requires $\mathcal{O}(T \cdot H \cdot d)$ floats in `'head'` mode. No $T \times T$ attention matrix is ever materialized.
- **Compute Complexity**: Feature map evaluation executes in strictly linear time $\mathcal{O}(T \cdot m \cdot d)$. For heads with $\gamma = 1.0$, evaluation uses a single vectorized `torch.cumsum` in $\approx 5\text{ ms}$ for $T = 5000$.
- **Orthogonal Random Features**: Uses Mezzadri QR phase correction on Gaussian blocks with $\sqrt{d}$ norm scaling, achieving lower empirical variance than independent random Fourier features while maintaining unbiasedness.

## State, Complexity and Limits

- **Class and file**: `TransformerEffect`, `src/tam/model/spectrum/_transformer.py`; formula token `attn()`. Label: (BETA).
- **State** (causal form): `{"S": (B, H, m, d), "Z": (B, H, m)}`; none for `causal=False`.
- **Layout**: `(B, T)` for one column, `(B, T, d)` with `others=`.
- **Time and memory**: $\mathcal{O}(B\,H\,T\,m\,d)$ for the features and the recurrences; no $T \times T$ matrix.
- **Penalty**: in `head` mode $\lambda\operatorname{diag}(1 + h) \otimes I_d$; the per-head projections differ but share these penalty weights. In `bilinear` mode $\lambda(1 + \|\omega_j\|^2)\otimes I_d$ with the first head's projection.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction for the causal form.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `TransformerEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_transformer.py
:language: python
:start-after: "#: <attn_feature_map>"
:end-before: "#: </attn_feature_map>"
```

Penalty of `TransformerEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_transformer.py
:language: python
:start-after: "#: <attn_penalty_matrix>"
:end-before: "#: </attn_penalty_matrix>"
```

Parsing of the `attn(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_transformer>"
:end-before: "#: </parse_transformer>"
```
