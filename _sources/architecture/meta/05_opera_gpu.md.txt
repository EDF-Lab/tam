# OPERA GPU Tensor Batching

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/meta/05_opera_aggregation.md)
  
This chapter explores the software engineering of the `OperaTAM` module. The core challenge in online aggregation is the inherently sequential nature of the time loop. While standard Primal solvers vectorize entire time series matrices instantaneously, online learning dictates that the weights at $t+1$ depend strictly on the observation at time $t$.

---

## The Bottleneck of Sequential Python Loops

If the aggregation loop $t=1, \dots, T$ were written natively in Python for a dataset containing thousands of time series (Groups), the performance would collapse. 

At every time step, Python would have to dispatch minuscule array operations to the GPU. This triggers massive CPU-GPU kernel launch bottlenecks, leaving the massive parallel capabilities of the CUDA cores completely starved. 

---

## 3D Tensor Batching & Padding

To achieve GPU saturation despite the sequential constraint, `OperaTAM` engineers the data into a strict **3D Tensor format** prior to execution.


**Architectural Choice (Uniform Temporal Alignment):**
During the preparation phase in the `predict_online` method:
1. The framework uses the internal `_balance_groups` utility (with `method="fill"`) to artificially pad any asynchronous or missing time series data with "fake dates". This padding rigorously guarantees that every single group has the exact same temporal length $T$.
2. The data is stacked into continuous tensors `X_tensor_3d` and `Y_tensor_3d` with the shape `(Groups, Time, Experts)` or $(B \times T \times K)$.

By perfectly aligning the temporal dimension across all groups, the framework transforms $G$ independent sequential loops into a single, massive parallel operation. Once the compiled loop finishes simulating the historical timeframe, the framework seamlessly re-associates the outputs with their original indices and uses a boolean mask (`_cleanup_dummies`) to automatically strip away the artificial padding.

---

## TorchScript C++ Compilation

Once the 3D tensors are assembled, they are passed into the specific algorithmic loops `_mlpol_loop_optimized_3d` or `_ewa_loop_optimized_3d`.

**Architectural Choice (Bypassing the GIL):**
To eliminate the Python Global Interpreter Lock (GIL) overhead from the sequential $T$ iteration, these functions are decorated with `@torch.jit.script`. PyTorch strictly compiles the entire sequential logic-including the regret tracking, the adaptive learning rates, and the weight normalizations-into a single C++ computational graph.

### Numerical Stability Engineering

Within these C++ compiled loops, extreme care is taken to prevent floating-point disasters common to GPU hardware:

**1. Scale Invariance (MLpol):** In `_mlpol_loop_optimized_3d`, the polynomial minimax strategy tracks squared regrets. To ensure the dynamic learning rates don't exponentially explode when evaluating massive industrial targets, the entire input tensor `X_scaled` and target `Y_scaled` are divided by a `scale_factor` (the per-group maximum absolute value) before entering the loop. 
A `torch.where` mask safely intercepts any groups with perfectly zero targets to prevent division-by-zero (`NaN`) crashes during this scaling.

```{literalinclude} ../../../../src/tam/model/opera.py
:language: python
:start-after: "#: <torch_jit_mlpol>"
:end-before: "#: </torch_jit_mlpol>"
:caption: src/tam/model/opera.py (TorchScript Compiled MLpol Loop)
```

**2. Safe Softmax (EWA):** In `_ewa_loop_optimized_3d`, computing standard exponentials ($\exp(-x)$) for massive cumulative losses triggers immediate `inf` or `NaN` values. The code utilizes the stable log-sum-exp trick (`scaled_losses - max_val`) to safely bound the maximum exponent to exactly 0, preventing exponential overflow without altering the mathematical weight distribution.

```{literalinclude} ../../../../src/tam/model/opera.py
:language: python
:start-after: "#: <torch_jit_ewa>"
:end-before: "#: </torch_jit_ewa>"
:caption: src/tam/model/opera.py (TorchScript Compiled EWA Loop)
```

---

## Causal Boundary Enforcement (Horizon Shifting)

In production forecasting, online algorithms must not leak future information. 

**Architectural Choice (Information Delay):**
After the optimized loops evaluate the performance weights natively, the `OperaTAM.predict_online` method intercepts the raw `weights_np` array. If the user specifies a multi-step forecasting scenario (`horizon_steps > 1`), the algorithm explicitly shifts the learned weights forward by $H - 1$ steps. 
The "blind" initial steps are overridden with uniform weights (`1.0 / len(experts)`), and the historical weights are shifted causally to the right: `shifted_weights[:, shift:, :] = weights_np[:, :-shift, :]`. This ensures the aggregated prediction at time $t$ strictly relies on expert performance evaluated *prior* to the blind forecast horizon.
