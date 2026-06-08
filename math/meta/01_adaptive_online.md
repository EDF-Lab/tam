# Adaptive Online Learning (AdaptiveTAM)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/01_adaptive_code.md)
  * **Alternative tracking:** [Kalman Filter Theory](02_kalman_filter.md)
  
Standard Generalized Additive Models (GAMs) assume the underlying data generating process is globally stationary {cite:p}`hastie2017generalized`. However, real-world time series, particularly in energy grids and financial markets, frequently experience concept drift, structural breaks, and unobserved localized shocks {cite:p}`doumeche2025forecasting`. 

To address this without continuously retraining the entire global topology, the TAM framework introduces a parallel "Online" sliding-window strategy, operationalizing the foundational principles of WeaKL-Online into a highly scalable architecture {cite:p}`doumeche2025forecasting`. 

For alternative approaches dealing with state-space stochasticity rather than deterministic windows, please refer to the continuous tracking methods ([See Dynamic Tracking via Extended Kalman Filtering](02_kalman_filter.md)).

## The Two-Stage "Expert-Corrector" Architecture

The `AdaptiveTAM` engine operates on a strict two-stage residual correction paradigm, conceptually dividing the forecasting task between a slow-moving "Long-Term Expert" and a highly reactive "Local Corrector". The final aggregated prediction at any time step $t$ is mathematically demonstrated as the sum of two distinct projections:

$$\hat{Y}_{t} = \underbrace{\Phi_{base, t} \hat{\theta}_{base}}_{\text{Long-Term Expert}} + \underbrace{\Phi_{adapt, t} \hat{\theta}_{adapt, t}}_{\text{Local Corrector}}$$

### 1. The Base Model (The Long-Term Expert)
A primary `StaticTAM` model is fitted on the entire deep historical dataset. Its purpose is to learn the fundamental "physics" of the series-the stable, long-term secular trends, macroscopic weather responses, and rigid global seasonalities via the Primal exact resolution. It projects the inputs through a dictionary $\Phi_{base, t}$ and computes a static, globally optimized weight vector $\hat{\theta}_{base}$ ([See The Exact Primal Resolution](../core/01_primal_model.md)). 

### 2. The Adaptive Model (The Local Corrector)
A secondary `StaticTAM` model is dynamically fitted exclusively to the residuals of the long-term expert. Because the base model already handles the macro-structures, the local corrector only needs to learn how the current, immediate environment is temporarily deviating from the global norm. The target for this adaptive layer is explicitly defined as the localized error:

$$\epsilon_t = Y_t - \Phi_{base, t} \hat{\theta}_{base}$$

## Mathematical Formulation of the Parallel Sliding-Window

Unlike sequential recursive filters which update state vectors incrementally, `AdaptiveTAM` utilizes a "Parallel-in-Time" sliding window mechanism.  This mathematically decouples the optimization steps across time, allowing the system to be mapped to modern GPU tensorization ([See 3D Tensorization & Data](../core/02_tensorization.md)).

For a given group and a specific forecast step $t$, the adaptive model strictly bounds its observation matrix to a local temporal window of size $W$. To maintain strict causality and prevent target leakage in industrial pipelines, the framework enforces an information delay $H$ (horizon steps). 

The local empirical risk minimization (ERM) problem for the corrector at time $t$ over the window $[t-W-H, t-H]$ is formulated as minimizing the regularized loss functional $\mathcal{L}_t(\theta)$:

$$\mathcal{L}_t(\theta) = || \Lambda_{W} (\epsilon_{W} - \Phi_{adapt, W}\theta) ||^2_2 + W \cdot \theta^\top P \theta$$

Where:
* $\Phi_{adapt, W} \in \mathbb{R}^{W \times D}$ represents the heterogeneous spectral dictionary mapping the local features over the window $W$.
* $\Lambda_W \in \mathbb{R}^{W \times W}$ is a diagonal weight matrix applying localized observation confidences (often an identity matrix in standard formulations).
* $P \in \mathbb{R}^{D \times D}$ is the structured block-diagonal penalty matrix defining the structural prior (e.g., Sobolev norm for Fourier maps, finite differences for Splines) ([See The Heterogeneous Spectral Dictionary](../spectrum/01_spectrum_intro.md)).
* $W$ acts as a structural scaling scalar to ensure the penalty $P$ remains perfectly balanced relative to the truncated sample size of the window, preventing under-regularization.

### The Exact Local ERM Solution

To find the optimal localized weights $\hat{\theta}_{adapt, t}$, we compute the gradient of the loss functional with respect to $\theta$ and set it to zero:

$$\nabla_\theta \mathcal{L}_t(\theta) = -2 \Phi_{adapt, W}^\top \Lambda_{W}^\top \Lambda_{W} (\epsilon_{W} - \Phi_{adapt, W}\theta) + 2 W P \theta = 0$$

Solving for $\theta$ yields the exact local primal minimizer computed via direct linear algebra {cite:p}`doumeche2025forecasting`:

$$\hat{\theta}_{adapt, t} = \left( \Phi_{adapt, W}^\top \Lambda_{W}^\top \Lambda_{W} \Phi_{adapt, W} + W \cdot P \right)^{-1} \Phi_{adapt, W}^\top \Lambda_{W}^\top \Lambda_{W} \epsilon_{W}$$

By flattening the group and window dimensions into a single massive 3D tensor batch, the framework computes these dynamic corrector weights simultaneously across all groups and time steps.  This isolates the regularization scale for each distinct topological window and neutralizes concept drift, bypassing the computational overhead typically associated with rolling dense matrix inversions.