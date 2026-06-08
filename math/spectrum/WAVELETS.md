# The Wavelet Effect (Ricker)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## Foundational Theory: Time-Frequency Localization

Classical spectral bases, such as the `FourierEffect`, are globally supported across the entire temporal domain. While exceptionally powerful for extracting rigid, deterministic seasonalities, Fourier transforms struggle with non-stationary singularities, such as sudden structural shocks, local transients, or brief operational anomalies. Modeling these transients with global sine waves induces severe "ringing" artifacts (Gibbs phenomenon) across the entire time series.

While wavelets are proven forecasters {cite:p}`amato2021forecasting`, TAM bypasses traditional discrete, sequential pipelines by constructing an **Continuous Wavelet Transform dictionary** {cite:p}`torrence1998practical, mallat1999wavelet`. Unlike Fourier harmonics, wavelets possess compact or rapidly decaying support, allowing them to perfectly isolate local shocks without mathematically corrupting distant time steps.

## Formula Definition: The Continuous Wavelet Grid ($\Phi$)

To construct the finite-dimensional feature map, the input variable $x$ (strictly normalized to the domain $[-1, 1]$) is mapped against a pre-defined bidimensional grid of scales (dilations) and locations (translations), following standard Continuous Wavelet Transform (CWT) protocols {cite:p}`torrence1998practical`:
* **Locations ($b$)**: A set of $n_{locations}$ linearly spaced centers across the empirical $[-1, 1]$ interval.
* **Scales ($a$)**: A set of $n_{scales}$ logarithmically spaced dilations, ranging from $0.01$ (capturing high-frequency, narrow spikes) to $1.0$ (capturing low-frequency, broad waves).

For each specific combination of scale $a_i$ and location $b_j$, the input is shifted and scaled to compute the localized dimensionless context $t = \frac{x - b_j}{a_i}$.

The framework explicitly evaluates the **Ricker Wavelet**. Originally derived to model the laws of propagation for seismic shocks {cite:p}`ricker1951form`, it corresponds mathematically to the negative normalized second derivative of a Gaussian function:

$$\psi(t) = (1 - t^2) e^{-t^2 / 2}$$

To rigorously guarantee that wavelets across all scales maintain constant spectral energy (a fundamental requirement for operating within a valid Reproducing Kernel Hilbert Space), the TAM tensor engine strictly scales the projection by the $L_2$ normalization factor $1/\sqrt{a_i}$ {cite:p}`mallat1999wavelet`.

The explicit finite-dimensional feature map extracted for the Primal matrix $\Phi$ is the concatenated block of these normalized wavelet evaluations across the entire grid:

$$\phi_{wavelet}(x) = \left[ \frac{1}{\sqrt{a_i}} \psi\left(\frac{x - b_j}{a_i}\right) \right]_{i=1, \dots, n_{scales}}^{j=1, \dots, n_{locations}}$$

## The Scale-Dependent Structural Penalty ($P$)

A dense grid of localized wavelets is exceptionally prone to overfitting, particularly to high-frequency stochastic noise. The foundational theory of **Wavelet Shrinkage** {cite:p}`donoho1994ideal` dictates that to achieve ideal spatial adaptation, wavelet coefficients must be aggressively thresholded, shrinking small, noisy coefficients exactly to zero.

Because TAM operates via an exact, closed-form Ridge ($L_2$) resolution rather than iterative $L_1$ thresholding (Lasso), it mathematically adapts Wavelet Shrinkage into a continuous, **scale-dependent structural penalty**. This penalization acts as a "whitening" operator that aggressively suppresses small-scale (high-frequency) basis functions unless they are strongly supported by the empirical loss gradient.

The structural penalty sub-matrix $P_{wavelet}$ is defined as a strictly diagonal matrix where the penalty multiplier is inversely proportional to the squared scale $a_i^2$:

$$P_{wavelet} = \lambda \cdot \text{diag}\left( \frac{1}{a_i^2} \right)$$

Where $\lambda$ is the global regularization strength. As $a_i \to 0$ (narrow, high-frequency spikes), the penalty scales toward infinity. This algebraically enforces a rigorous sparsity prior, ensuring the model only activates high-frequency wavelets for genuine, statistically significant structural shocks, exactly mimicking the oracle performance proven by Donoho and Johnstone {cite:p}`donoho1994ideal`.

## RKHS Eligibility & Exact Resolution

By equipping the continuous wavelet frame with this strictly positive, scale-dependent diagonal penalty, the localized basis functions are formally bounded in the resulting Reproducing Kernel Hilbert Space (RKHS). Because $1 / a_i^2 > 0$ strictly for all defined scales, the penalty matrix is strictly positive definite. 

This formulation structurally isolates the wavelet null space, allowing these localized transient features to be mathematically flattened alongside continuous global polynomials (e.g., Chebyshev) and deep neural random features (NNGP) into the global block-diagonal matrix $P$. This object-oriented geometry guarantees that local transients are solved simultaneously within the exact Primal Ridge solver without disrupting the global convexity of the normal equations.