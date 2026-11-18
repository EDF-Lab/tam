# Bispectrum Effect: PyTorch Architecture & Implementation

**Navigation:**
  * **Mathematical Theory:** [See BISPECTRUM.md](../../math/spectrum/BISPECTRUM.md)
  * **Spectrum API Overview:** [See The Spectrum API](../core/06_the_spectrum_api.md)

## Overview & Formula Token

The Bispectrum effect is exposed in formulas via the `bispec()` token:

```python
# Default bispectrum with K=8 canonical triads
formula = "Load ~ f(Hour, m=12) + bispec(Hour, k=8)"

# Custom FFT window length and higher Sobolev order
formula = "Load ~ bispec(Hour, k=12, n_fft=128, s=2.0)"

# Explicit triads from a diagnostic pre-screening are given to the class (a formula cannot hold a list of pairs):
# BispectrumEffect("Hour", triads=[(12, 18), (6, 12)], n_fft=128)

# Interaction with ambient temperature via tensor product
formula = "Load ~ te(bispec(Hour, k=4), s(Temperature, k=8))"
```

## Diagnostic Pre-Screening: Hinich Bispectrum Test

The module provides an automated diagnostic tool to detect quadratic phase coupling (QPC), test Gaussianity, and select top coupled frequency triads:

```python
from tam.model.diagnostics import hinich_bispectrum_test

# Analyze series with Welch-averaged bicoherence
diag = hinich_bispectrum_test(
    data["Load"].values,
    n_fft=128,
    overlap=0.5,
    top_k=8,
)

print(f"Gaussianity p-value: {diag['p_value_gaussianity']:.4e} (Is Gaussian: {diag['is_gaussian']})")
print(f"Linearity IQR: {diag['linearity_iqr']:.4f}")

# Extract top coupled triads
top_triads = diag["top_triads"]
for t in top_triads:
    print(f"Coupled: (k1={t['k1']}, k2={t['k2']}) -> k3={t['k3']}, b^2={t['bicoherence']:.3f}")
```

## Parameter Specification

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_name` | `str` | *required* | Primary feature column name. |
| `k` | `int` | `8` | Number of coupled frequency triads ($4K$ primal columns). |
| `n_fft` / `window` | `int` | `128` | Window length for FFT segmentation and discrete frequency scaling (`window` is a formula alias for `n_fft`). |
| `sobolev_order` / `s` | `float` | `1.0` | Exponent of the bi-frequency Sobolev penalty $(1 + \sum \omega_i^2)^s$. |
| `triads` | `list` | `None` | Optional sequence of discrete harmonic pairs $(k_1, k_2)$ or diagnostic dicts. |
| `bicoherences` | `list` | `None` | Optional bicoherence weights $b^2$ for the specified triads. |
| `ap` | `float` | `default_alpha_p` | Log10 of the regularization parameter $\lambda_p = 10^{\text{ap}}$. |
| `extrapolate` | `str` | `'continue'` | OOD extrapolation strategy beyond $[-1, 1]$. |

## Complexity & Performance

- **Memory Complexity**: The feature map requires $\mathcal{O}(T \cdot 4K)$ floats, identical in memory footprint to standard Fourier features.
- **Compute Complexity**: Evaluated by short-time convolutions in $\mathcal{O}(B\,T\,n_{\text{fft}}\,F)$ operations ($F$ distinct frequencies); the effect carries the last $n_{\text{fft}}-1$ samples as its state (see the state section).
- **Penalty Matrix**: Diagonal $4K \times 4K$ matrix with $\mathcal{O}(K)$ non-zero entries, adding zero fill-in overhead to the Primal normal equations.
- **Hardware Acceleration**: Full compatibility with mixed-precision, float32/float64, and chunked Cholesky/QR solvers.

## State, Complexity and Limits

- **Class and file**: `BispectrumEffect`, `src/tam/model/spectrum/_bispectrum.py`. Label: (BETA).
- **State**: `{"hist": (B, n_fft - 1, 1)}`. **Layout**: `(B, T)`; one column only (`others=` is refused).
- **Columns**: `4 k`, four per triad: $\operatorname{Re}B/s$, $\operatorname{Im}B/s$, $|z_1z_2|/s_a$ and the cosine of the biphase; their scales are frozen at initialization. With the default triads and noise the design has full column rank.
- **Time and memory**: $\mathcal{O}(B\,T\,n_{\text{fft}}\,F)$ for the convolutions ($F$ = number of distinct frequencies of the triads), $\mathcal{O}(B\,T\,k)$ for the columns.
- **Extrapolation**: `'linear'` and `'saturation'` are refused at construction.
- **Limits**: the window is $n_{\text{fft}}$ samples long (alias `window`), so a triad needs a series much longer than that; the first $n_{\text{fft}}$ steps of a pass without a state read replicated history.
- **Smoothing weight**: in a formula `bispec(x)` starts from `ap = -5` (log10 of the weight), the value the benchmarks use, instead of the model default of `-9`; an explicit `ap=` wins. The columns of a very smooth series (a seasonal and a diurnal sinusoid with little noise) are nearly collinear (condition number of the Gram matrix up to 1e7), and with the default `-9` the unpenalised solve amplified small differences between the training and the test chunk: on such a frame the test RMSE was about 90 times that of the spline-only baseline (388.6 against 4.3; 146 with `ap=-5`, 6.5 with `ap=-1`). `auto_fit` tunes `ap`; `-5` is the value used in the benchmarks, not an optimum.
- **Short groups**: `initialize` warns (`UserWarning`) when the window is longer than the shortest training group: no window ever fills and the columns come from zero-padded start-ups (the smoke frame of 40 rows per group with the default window of 128 gave a test RMSE of 4437 for this reason). The warning is raised at `initialize` only; a short chunk continued with `continue_state=True` carries its history and is valid.
- **Benchmark note**: on the national load benchmark the effect shows no static gain (the 2026-11-14 retest: static test +1.4 % against the spline reference, adaptive test +0.5 %). No gain was measured on national load, static or adaptive; the effect ships as BETA for other series and for Opera pools.
- **Chunk equivalence bound**: `tests/spectrum/test_stateful_contract.py` checks that the second of two chunks, started from the state of the first, equals the matching rows of the one-pass features within `atol=1e-8` (float64), and that every group computed alone equals the same group inside a batch within `atol=1e-10`.

## Code

Feature map of `BispectrumEffect` (`build_feature_map`):

```{literalinclude} ../../../../src/tam/model/spectrum/_bispectrum.py
:language: python
:start-after: "#: <bispec_feature_map>"
:end-before: "#: </bispec_feature_map>"
```

Penalty of `BispectrumEffect` (`build_penalty_matrix`):

```{literalinclude} ../../../../src/tam/model/spectrum/_bispectrum.py
:language: python
:start-after: "#: <bispec_penalty_matrix>"
:end-before: "#: </bispec_penalty_matrix>"
```

Parsing of the `bispec(...)` token in the factory:

```{literalinclude} ../../../../src/tam/model/spectrum/_factory.py
:language: python
:start-after: "#: <parse_bispectrum>"
:end-before: "#: </parse_bispectrum>"
```
