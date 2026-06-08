# Online Prediction by Expert Aggregation (OPERA)

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/05_opera_gpu.md)
  
While the Generalized Additive Models (GAMs) and hierarchical solvers within the Time series Additive Model (TAM) framework provide exceptionally robust point predictions, industrial forecasting pipelines often rely on multiple, fundamentally distinct algorithms (e.g., a Primal TAM model, a legacy SARIMA model, and a Deep Neural Network). 

Because different algorithms excel under different structural regimes, statically selecting a single "best" model inevitably leads to catastrophic failures during Concept Drift. To solve this mathematically, TAM integrates the **Online Prediction by Expert Aggregation (OPERA)** framework {cite:p}`gaillard2016opera`. 

Instead of picking a single winner, OperaTAM dynamically computes a convex combination of all available experts, adjusting the weights sequentially based on their real-time predictive performance.

---

## The Game Theory Framework and Regret

Online aggregation is formulated as a sequential, deterministic game between the forecaster and the environment {cite:p}`cesa2006prediction`. At each time step $t=1, \dots, T$:

1. A set of $K$ experts provides their predictions $x_{k,t}$.
2. The forecaster aggregates these predictions using a weight vector $w_{k,t}$ (where $\sum w_{k,t} = 1$) to form the global ensemble prediction: 

   $$\hat{y}_t = \sum_{k=1}^K w_{k,t} x_{k,t}$$

3. The true observation $y_t$ is revealed.
4. The forecaster suffers a loss $\ell(\hat{y}_t, y_t)$ and updates the weights for the next step based on the individual losses suffered by the experts, $\ell(x_{k,t}, y_t)$.



The mathematical objective is to minimize the **Regret** ($R_{k,T}$). Regret bounds the difference between the forecaster's cumulative loss and the cumulative loss of the best possible constant expert chosen in hindsight {cite:p}`cesa2006prediction`:

$$R_{k,T} = \sum_{t=1}^T \ell(\hat{y}_t, y_t) - \sum_{t=1}^T \ell(x_{k,t}, y_t)$$

---

## Exponentially Weighted Average (EWA)

The most fundamental strategy to minimize regret is the Exponentially Weighted Average (EWA) algorithm. The weight assigned to expert $k$ at time $t+1$ is decayed exponentially according to its cumulative loss $L_{k,t} = \sum_{s=1}^t \ell(x_{k,s}, y_s)$, scaled by a fixed learning rate $\eta > 0$:

$$w_{k,t+1} = \frac{\exp(-\eta L_{k, t})}{\sum_{j=1}^K \exp(-\eta L_{j, t})}$$

### GPU Hardware Translation (The Softmax Trick)
While EWA provides excellent theoretical guarantees, evaluating standard exponentials of cumulative losses on modern Float32 GPU architectures frequently results in `NaN`s due to exponential overflow or underflow. 

The `OperaTAM` module enforces two strict stabilization tricks directly inside its compiled TorchScript loop:
1. **Target Scaling:** All targets and experts are divided by the group's maximum absolute magnitude $\max(|Y|)$ to restrict the continuous loss evaluation to a bounded, numerically stable domain.
2. **Shifted Exponentials:** Before computing the exponential, the maximum scaled loss across all experts is subtracted:

   $$w_{k,t+1} \propto \exp\left( -\eta (L_{k, t} - \max_j L_{j, t}) \right)$$

   This ensures the maximum exponent is always exactly 0, algebraically eliminating overflow while mathematically preserving the exact proportional weight distribution {cite:p}`gaillard2016opera`.

---

## Polynomial Minimax Strategy (MLpol)

The primary limitation of EWA is its reliance on a fixed hyperparameter $\eta$. To provide a fully automated, parameter-free ensemble, TAM natively implements the **MLpol (Polynomial Minimax)** strategy {cite:p}`gaillard2016opera`.

Rather than tracking absolute losses, MLpol tracks the **Linearized Pseudo-Regret** $r_{k,t}$ of each expert against the aggregated consensus:

$$r_{k,t} = \nabla \ell(\hat{y}_t, y_t) \cdot (x_{k,t} - \hat{y}_t)$$
$$R_{k,t} = \sum_{s=1}^t r_{k,s}$$

According to the polynomial potential bounds established by Cesa-Bianchi and Lugosi {cite:p}`cesa2006prediction`, the optimal weight update is proportional to the gradient of the squared positive regret. The TAM implementation strictly translates this as applying a Rectified Linear Unit (ReLU) to the cumulative regret, multiplied by an adaptive learning rate $\eta_{k,t}$:

$$w_{k,t+1} = \frac{\eta_{k,t} \max(0, R_{k,t})}{\sum_{j=1}^K \eta_{j,t} \max(0, R_{j,t})}$$

**Adaptive Learning Rate Decay:**
To mathematically guarantee that the algorithm operates within optimal theoretical minimax bounds, the expert-specific learning rates $\eta_{k,t}$ are dynamically decayed. As implemented in the core engine, the denominator is updated using the maximum observed squared pseudo-regrets:

$$\eta_{k,t} = \left( \frac{1}{\eta_{k,t-1}} + r_{k,t}^2 + \max_j \left( w_{j,t} [r_{j,t}]_+^2 \right) \right)^{-1}$$

This analytically guarantees convergence without requiring the user to tune any parameters {cite:p}`gaillard2016opera`.

---

## 3D Tensor Batching (Gigadata Scaling)

The original OPERA implementation sequentially loops through the time series step-by-step. However, executing a standard `for t in range(T):` loop in Python over millions of smart-meter rows triggers severe CPU-GPU kernel launch bottlenecks, effectively paralyzing the hardware.



To solve this, `OperaTAM` re-architects the online aggregation into a **3D Tensor Batching** paradigm. 
The historical data is padded and reshaped into a contiguous GPU tensor of dimensions `(Batch, Time, Experts)`, where the `Batch` dimension represents independent spatial groups (e.g., thousands of distinct energy substations). 

The sequential mathematical loop is then decorated with `@torch.jit.script`. This compiles the sequential MLpol evaluation, simultaneously tracking the regret sequences for thousands of independent groups in a single pass without standard Python loop overhead. This fuses the theoretical purity of the OPERA framework {cite:p}`gaillard2016opera` with the raw computational velocity of modern deep learning infrastructures.
