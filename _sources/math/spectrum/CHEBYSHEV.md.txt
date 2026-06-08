# The Chebyshev Effect (Polynomial Minimax Basis)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)
  
## Foundational Theory: Runge's Phenomenon & Minimax Approximation

To model smooth, long-term secular trends and continuous macro-phenomena, classical statistical models often rely on standard polynomial regression using a monomial basis ($1, x, x^2, \dots, x^n$). 

However, modern approximation theory dictates that evaluating high-degree monomials yields highly ill-conditioned Vandermonde matrices. More critically, uniformly spaced polynomial interpolation systematically triggers **Runge's phenomenon**, catastrophic, unbounded oscillations at the boundaries of the interval {cite:p}`trefethen2019approximation`. 

To mathematically resolve this, the TAM framework abandons the monomial basis and instead projects the continuous input space onto a basis of orthogonal **Chebyshev polynomials of the first kind**, $T_n(x)$. Chebyshev polynomials are mathematically proven to achieve near-optimal **minimax approximation** (minimizing the maximum absolute error across the domain), guaranteeing that the interpolation error is distributed evenly without explosive boundary oscillations {cite:p}`trefethen2019approximation`.

## Topological Constraints & Formula Definition ($\Phi$)

As established in the data preparation pipeline, the input variable $x$ is strictly normalized to the domain $[-1, 1]$ via an affine transformation. This is not merely a scaling heuristic; it is a fundamental topological requirement. Chebyshev polynomials only maintain their minimax approximation properties and their strict orthogonality (with respect to the singular weight function $(1-x^2)^{-1/2}$) within this specific bounded interval {cite:p}`rivlin1990chebyshev`.

The explicit finite-dimensional feature map $\phi(x)$ evaluates these polynomials up to a specified `degree` $D$. However, evaluating Chebyshev polynomials via direct trigonometric definitions ($T_n(x) = \cos(n \arccos x)$) is computationally expensive, and expanding them into standard algebraic powers induces severe numerical instability.

Instead, the framework utilizes the mathematically exact **three-term recurrence relation** {cite:p}`rivlin1990chebyshev`:

$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x) \quad \text{for } n \ge 2$$

The Primal mapping matrix is the horizontal concatenation of these recursive evaluations:

$$\phi_{chebyshev}(x) = \left[ T_n(x) \right]_{n=1}^D$$

## Numerical Stability & Hardware Translation

A core engineering novelty of the TAM framework is the translation of this recurrence relation natively into GPU tensor operations. Higham proved that evaluating polynomial bases via direct exponentiation leads to catastrophic floating-point cancellation and massive roundoff errors on modern hardware {cite:p}`higham2002accuracy`. 

By strictly utilizing the three-term recurrence relation, the `_chebyshev.py` module explicitly bounds the floating-point arithmetic. Furthermore, by evaluating this recursively in-place (updating $T_n$ using only the active $T_{n-1}$ and $T_{n-2}$ vectors), the tensor engine entirely avoids materializing dense, ill-conditioned intermediate matrices, ensuring perfectly stable gradients even at high polynomial degrees ($D > 50$).

## The Spectral Sobolev Penalty ($P$)

Even with the optimal minimax basis, high-degree polynomials can overfit empirical noise. To prevent this, the framework equips the Chebyshev basis with a degree-dependent **Sobolev structural penalty**. 

The structural penalty sub-matrix $P_{chebyshev}$ is defined as a purely diagonal matrix where the penalty applied to each polynomial term scales geometrically with its degree $n$, governed by a user-defined smoothness parameter $s$:

$$P_{chebyshev} = \lambda \cdot \text{diag}\left( n^{2s} \right)_{n=1}^D$$

Where $\lambda$ acts as the global regularization strength. Operationally, this acts as a mathematically exact low-pass filter {cite:p}`boyd2001chebyshev`. As the polynomial degree $n$ increases, the penalty scales aggressively, forcing the global solver to heavily penalize high-degree, high-frequency "wiggles" in favor of smooth, low-degree structural trends.

## RKHS Eligibility

By equipping the orthogonal polynomial basis with this strictly positive, degree-dependent diagonal penalty, the functional space is formally bounded within a valid Reproducing Kernel Hilbert Space (RKHS). Because $n^{2s} > 0$ strictly for all evaluated degrees $n \ge 1$, the penalty matrix is strictly positive definite. 

This formulation perfectly isolates the null space (the unpenalized constant $T_0$ is intentionally excluded from the effect and handled by the global intercept). This satisfies Aronszajn's conditions for RKHS invertibility {cite:p}`aronszajn1950theory`, guaranteeing that the global macro-trend can be resolved exactly and simultaneously alongside complex local transients (wavelets) and deep neural features without mathematically corrupting the global convexity of the Ridge solver.