# The Fourier Effect (Sobolev Spectral Basis)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## Foundational Theory & Structural Evolution

To model rigid, deterministic seasonalities and cyclical phenomena (e.g., daily load profiles or yearly temperature variations), the framework projects the continuous input space into a frequency domain using a truncated Fourier series {cite:p}`harvey1989forecasting`. 

In its foundational formulation for Weak Kernel Learning (WeaKL), Doumèche et al. (2025) defined this spectral basis within a Reproducing Kernel Hilbert Space (RKHS) using complex exponentials ($e^{ikx}$). While mathematically elegant for proving convergence in Sobolev spaces {cite:p}`doumeche2025forecasting`, managing complex-number arithmetic within deep tensor autograd engines (like PyTorch) introduces severe memory and performance bottlenecks. 

To achieve Gigadata scalability while strictly preserving the theoretical guarantees of WeaKL, the TAM framework re-architects this spectral projection. It maps the complex formulation into an exact, equivalent real-valued basis using strictly harmonic sines and cosines, maximizing native GPU matrix multiplication throughput.

## Formula Definition: The Real Spectral Basis ($\Phi$)

As established in the data preparation phase, the input variable $x$ is normalized to the domain $[-1, 1]$. To properly evaluate the trigonometric basis, this domain is linearly rescaled:

$$x_{scaled} = \pi x$$

The explicit finite-dimensional feature map $\phi(x)$ evaluates a sequence of $m$ harmonic frequencies ($k = 1, \dots, m$). To handle both strictly periodic phenomena and transient wave dynamics, the framework introduces a topological `cyclic` toggle:

**1. Strict Periodic Boundary (`cyclic=True`):**
The basis is evaluated over the full cycle, guaranteeing that the endpoints meet perfectly ($f(-1) = f(1)$):

$$\phi_{fourier}(x) = \left[ \cos(k \pi x), \sin(k \pi x) \right]_{k=1}^m$$

**2. Relaxed Boundary (`cyclic=False`):**
The angular frequency is halved, mapping the $[-1, 1]$ domain to $[-\pi/2, \pi/2]$. This allows the solver to capture wave-like dynamics without mathematically forcing the edges to connect, preventing artificial distortion at the boundaries of the time series:

$$\phi_{fourier}(x) = \left[ \cos\left(\frac{k \pi x}{2}\right), \sin\left(\frac{k \pi x}{2}\right) \right]_{k=1}^m$$

## The Sobolev Structural Penalty ($P$)

Unpenalized Fourier series are highly susceptible to overfitting high-frequency noise. In kernel learning, this is mitigated by evaluating the function within a Sobolev space, which mathematically penalizes the roughness (derivatives) of the function {cite:p}`scholkopf2002learning`.

The TAM framework explicitly operationalizes this Sobolev norm {cite:p}`doumeche2025forecasting, smola2004tutorial`. The structural penalty sub-matrix $P_{fourier}$ is defined as a strictly diagonal matrix where the penalty applied to each harmonic pair scales with its frequency $k$, governed by a smoothness parameter $s$:

$$P_{fourier} = \lambda \cdot \text{diag}\left( [1 + k^{2s}]_{k=1}^m \oplus [1 + k^{2s}]_{k=1}^m \right)$$

Because $1 + k^{2s} > 0$ strictly for all $k \ge 1$, the penalty matrix is strictly positive definite. This guarantees an empty null space (the unpenalized constant $k=0$ is intentionally excluded from the effect and handled globally by the framework's intercept). This rigorously satisfies Aronszajn's conditions for a valid RKHS {cite:p}`aronszajn1950theory`, ensuring the perfect invertibility of the global normal equations.

## Theoretical Integration & Hardware Translation

Classical spectral convolutions typically rely on the Fast Fourier Transform (FFT). However, FFTs strictly require perfectly spaced, equidistant temporal grids {cite:p}`boyd2001chebyshev`, which frequently fails in real-world time series marred by missing data or irregular sampling.

By formulating the Fourier effect strictly within the continuous Primal RKHS dictionary, TAM bypasses this limitation entirely. The feature map $\Phi$ is constructed via massively parallelized tensor operations directly on the arbitrarily spaced continuous input $x$. Furthermore, because the penalty matrix $P_{fourier}$ is strictly diagonal, it injects minimal memory overhead into the global block-diagonal system. This allows highly expressive, high-frequency seasonal models ($m > 100$) to be solved instantly alongside neural features and splines within the exact $\mathcal{O}(N_{total} D^2)$ Primal solver.