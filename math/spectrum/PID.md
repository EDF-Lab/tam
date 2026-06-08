# The PID Effect (Autoregressive Control Dynamics)

**Navigation:**

* **Theory introduction:** [See the Intro](../../THEORY.md)
* **Related code architecture:** [See the Code Architecture](../../architecture/core/06_the_spectrum_api.md)

## Theoretical Critique: AR($p$) Over-parameterization vs. Physical Momentum

In time-series forecasting, standard models traditionally treat autoregressive target lags (e.g., $y_{t-1}, y_{t-2}, \dots, y_{t-p}$) as independent linear terms. While the Box-Jenkins methodology for ARMA modeling {cite:p}`box2015time` relies on statistical correlation to estimate these weights, this purely data-driven approach possesses a severe structural flaw when dealing with noisy physical systems (e.g., power grids or thermal masses).

An unconstrained AR($p$) process treats every lag as a free parameter, frequently overfitting to high-frequency stochastic noise and failing to explicitly capture the underlying physical momentum and inertial dynamics.

To mathematically constrain the model's internal memory, the TAM framework projects the autoregressive lags into a discrete **Proportional-Integral-Derivative (PID)** space. Originally formalized mathematically by Nicolas Minorsky in 1922 for the automatic steering of naval vessels {cite:p}`minorsky1922directional`, this acts as a highly interpretable, physics-informed inductive bias. It forces the hypothesis space to map the target's past states into three explicit dynamic components {cite:p}`aastrom2021feedback`:

1. **Proportional ($P$):** The immediate reaction to the most recent state ($y_{t-1}$).
2. **Integral ($I$):** The accumulation of past states to capture long-term baseline drift and physical inertia. To maintain numerical parity, this is computed as a sliding rolling mean over a predefined window $w$.
3. **Derivative ($D$):** The reaction to the trajectory or rate of change, anticipating sudden shocks ($y_{t-1} - y_{t-2}$).

For a target sequence $y$, the explicit Primal feature mapping evaluated at time $t$ is geometrically concatenated as:

$$\phi_{pid}(y_{t-1}) = \left[ y_{t-1}, \quad \frac{1}{w} \sum_{k=1}^w y_{t-k}, \quad y_{t-1} - y_{t-2} \right]^\top$$

This embeds the classical AR($p$) process into a strictly 3-dimensional control space, where the global Primal solver perfectly fits the corresponding control gains: $\theta_{pid} = [K_p, K_i, K_d]^\top$.

## Optimal Penalization: Bounding Derivative Noise Amplification

Because the feature map scales the Integral term via a rolling mean ($1/w$) rather than an absolute rolling sum, the numerical magnitudes of the $P$, $I$, and $D$ vectors remain roughly isotropic. This allows the framework to apply a standard $L_2$ Ridge penalty uniformly across the PID coefficients without structural bias.

**The Derivative Pathology:** However, optimal control theory dictates that pure derivative action is pathologically sensitive to high-frequency stochastic measurement noise {cite:p}`aastrom2021feedback`. In a purely unconstrained Ridge regression, the solver might assign a massive weight to $K_d$ to perfectly fit a historical shock, which would subsequently amplify minor noise into massive resonant swings during future extrapolation.

To prevent the GAM from amplifying this spurious noise, the structural penalty sub-matrix $P_{pid}$ applies an artificial stiffness multiplier $d_{mult}$ strictly to the derivative term:

$$P_{pid} = \lambda \cdot \text{diag}(1, 1, d_{mult})$$

Minimizing this specific anisotropic quadratic form structurally forces the controller to act as a stable low-pass filter. It heavily penalizes aggressive extrapolations of the derivative trajectory while allowing the proportional and integral momentum to freely guide the global forecast.

## Frequency Response & Structural Stability (Z-Transform)

A unique capability of the TAM framework is its ability to mathematically guarantee the physical stability of the learned machine learning weights. Because the PID effect is an endogenous autoregressive constraint, it forms a closed-loop dynamical system.

The exact discrete digital filter applied by the GAM to the sequence is:

$$\hat{y}_t = K_p y_{t-1} + K_i \left( \frac{1}{w} \sum_{k=1}^w y_{t-k} \right) + K_d (y_{t-1} - y_{t-2}) + \text{Exogenous}(t)$$

To evaluate the structural stability of how the model filters external exogenous shocks (like weather or calendar events), we map these learned coefficients to the frequency domain using the **Z-transform**. The characteristic polynomial of the denominator determines the system's poles.

Expanding the rolling mean and isolating the autoregressive coefficients $a_k$ for each lag $z^{-k}$ yields:

* $a_1 = K_p + \frac{K_i}{w} + K_d$
* $a_2 = \frac{K_i}{w} - K_d$
* $a_k = \frac{K_i}{w} \quad \text{for } 3 \le k \le w$

The corresponding closed-loop discrete transfer function $H(z)$ from the exogenous variables to the target is characterized by the denominator:

$$D(z) = z^w - a_1 z^{w-1} - a_2 z^{w-2} - \dots - \frac{K_i}{w}$$

By evaluating $H(z = e^{j\omega})$, the framework explicitly generates **Bode plots** of the learned machine learning model. This frequency-response visualization technique, pioneered by Hendrik W. Bode in the 1940s for amplifier stability {cite:p}`bode1945network`, allows us to inspect the exact gain and phase margins of the GAM.

If the maximum magnitude of the complex roots (poles) of $D(z)$ is strictly less than $1.0$, the model is mathematically proven to be **BIBO (Bounded-Input Bounded-Output) stable** {cite:p}`aastrom2021feedback`. This guarantees that the machine learning forecaster will not inject resonant instabilities if deployed inside a live Model Predictive Control (MPC) loop.

## RKHS Eligibility & Theoretical Integration

The PID effect is a direct, finite-dimensional linear mapping of endogenous lag states. When equipped with the strictly positive definite penalty matrix $P_{pid}$, it rigorously satisfies Mercer’s conditions, operating as a valid Reproducing Kernel Hilbert Space (RKHS) {cite:p}`aronszajn1950theory`.

By isolating the autoregressive target dynamics into this specific functional space, the TAM framework cleanly separates the endogenous inertia of the system from the exogenous drivers (like Splines for temperature or Fourier series for seasonality). Both spaces are concatenated and solved simultaneously within the exact, closed-form inversion of the global Primal matrix on the GPU. This allows continuous hyperparameter optimization to perfectly balance statistical likelihood with rigorous physical stability.
