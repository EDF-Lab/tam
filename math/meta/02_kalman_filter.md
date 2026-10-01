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

## Causal scaling and update step

The filter works in a normalised space: each tracked feature is mapped to $[-1, 1]$ and the target to $(y - c)/s$ with $c$ the centre and $s$ the half-amplitude of the target, per group.
These constants come from a **reference period**, never from the whole online period (they would otherwise carry future information into every forecast and set the effective observation and process noise $R$ and $Q$):
the first `calibration_steps` rows of each group (burn-in: their forecasts are not causal, do not score them), or a separate `calibration_data`. They are stored at `fit` and reused by `predict()`.

With `horizon_steps = 1` the state is updated at every step ($B = 1$, the exact sequential filter): the forecast at $t+1$ uses the target at $t$.
An explicit block size $B > 1$ applies the Woodbury block update and moves the state only every $B$ steps.
`fit` stores the state after the last update, so `fit(history up to t).predict(step t+1)` forecasts the next step exactly as the online simulation does (with $B = 1$).
Verified by `tests/test_kalman_causal.py`: no forecast at or before row $r$ changes when later rows change; $B = 1$ equals a plain NumPy Kalman filter to $10^{-10}$; the operational loop equals the simulation to $10^{-9}$.
  
