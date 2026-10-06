# Dynamic Tracking via Extended Kalman Filtering [BETA]

> 🚧 **Beta – Research Module**

⚠️ **Status: Active development**

- Core functionality is implemented but still evolving  
- Some features may be incomplete or subject to change  
- API is not yet stable  
- Intended for research use only (not production-ready)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/02_kalman_torchscript.md)
  * **Alternative tracking:** [Adaptive Online Theory](01_adaptive_online.md)

The tracker is the Kalman recursion used as a stochastic online optimizer, in its static extended form {cite:p}`de2021stochastic`: the effects of an already trained model (a GAM, a neural network) are frozen and only their coefficients move, as the state of a state-space model {cite:p}`de2022state`.

## Causal scaling and update step

The filter works in a normalised space: each tracked feature is mapped to $[-1, 1]$ and the target to $(y - c)/s$ with $c$ the centre and $s$ the half-amplitude of the target, per group.
These constants come from a **reference period**, never from the whole online period (they would otherwise carry future information into every forecast and set the effective observation and process noise $R$ and $Q$): the first rows of each group, whose forecasts are not causal and must not be scored, or a separate historical period.

With a horizon of one step the state is updated at every step ($B = 1$, the exact sequential filter): the forecast at $t+1$ uses the target at $t$. A block of $B > 1$ observations applies the Woodbury block update and moves the state only every $B$ steps.
The state after the last update forecasts the next, unseen steps exactly as the online filter would (with $B = 1$). No forecast at or before row $r$ changes when later rows change, and $B = 1$ is the plain Kalman filter.

## Process noise per formula term

The tracked coefficients follow a random walk $\theta_{t+1} = \theta_t + w_t$, $w_t \sim \mathcal{N}(0, Q)$ with $Q$ diagonal. By default every coefficient gets the same variance $q$ (in the standardised target space, where the observation noise is $R = 1$), the offset a larger one. Terms do not drift at the same speed (a temperature response moves slowly, a holiday effect or the offset quickly), so $Q$ can take one variance per formula term:

$$Q = \mathrm{diag}\big(q_{\text{offset}},\; q_{1}\,\mathbf{1}_{d_1},\; q_{2}\,\mathbf{1}_{d_2},\; \dots\big)$$

All the $d_k$ design columns of term $k$ share its variance $q_k$. Variational Bayesian variance tracking {cite:p}`vilmarest2024viking` estimates such variances dynamically, online; the variances of `KalmanTAM` stay static: set by the user or tuned offline on a calibration period. A variance of $0$ lets the covariance of the term shrink with every observation, so the term follows a changing coefficient less and less. The variances are best chosen on a calibration period that ends before the period that is reported.
