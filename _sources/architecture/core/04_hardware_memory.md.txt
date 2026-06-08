
# Hardware Memory Dispatch & Anti-OOM Systems

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/04_complexity.md)

This chapter details the hardware-aware engineering of the TAM framework. It explores how the theoretical complexity established in [Computational Complexity](../../math/core/04_complexity.md) is policed at runtime by dynamic PyTorch memory oracles to prevent catastrophic Out-Of-Memory (OOM) crashes during massive tensor calculations.

---

## The Hardware Abstraction Layer (HAL)

The foundation of the framework's stability is the `HardwareManager` (instantiated as the `hw` singleton) located in `hardware.py`. 

Before any mathematical operation begins, this layer dynamically probes the host machine to detect the most capable compute backend, routing tensor operations in descending order of preference: NVIDIA CUDA, Apple MPS, Intel XPU, and finally the Host CPU. 

Crucially, it manages disaster recovery via the `handle_oom` method. When an operation exceeds physical capabilities, this method intercepts the failure, executes a low-level cache purge (`torch.cuda.empty_cache()`), and computes a diminished workload batch size to allow the system to seamlessly retry the computation.

```{literalinclude} ../../../../src/tam/common/hardware.py
:language: python
:pyobject: HardwareManager.handle_oom
```

---

## The Memory Oracle and Safe Chunking

To proactively avoid invoking the OOM handler, the framework utilizes `_memory.py` as an advanced predictive oracle. 

Before the `_dispatcher.py` attempts to allocate the massive global Covariance Matrix $\Phi^T \Phi$, it queries `can_fit_dense_matrix` and evaluates the theoretical byte footprint against a strict **Multi-Tiered Memory Waterfall**:

1. **The Dense Inversion Limit:** The globally exact $\mathcal{O}(D^3)$ solver is only authorized if the exact theoretical byte footprint of the dense inversion (accounting for Float64 precision) requires $< 90\%$ of available VRAM, *and* the primal dimension is $D \le 7500$. If either threshold is breached, the workload is routed to the Matrix-Free Conjugate Gradient solver.
2. **Standard Group Chunking:** For static data processing, the oracle bounds spatial tensor chunks to $90\%$ of free VRAM (or $70\%$ of system RAM) to maximize GPU compute occupancy without triggering PyTorch out-of-memory states.
3. **Sliding Window Buffer (`AdaptiveTAM`):** Because online learning models require recursive history tracking, the oracle enforces a stricter $80\%$ VRAM limit ($60\%$ CPU RAM) to preserve buffer space for continuous state-space updates.

```{literalinclude} ../../../../src/tam/model/_memory.py
:language: python
:pyobject: can_fit_dense_matrix
```

---

## The OOM Safety Net in the Dispatcher

Despite predictive calculations, unpredictable memory spikes can still occur during tensor decompositions or highly concurrent batching. The `_dispatcher.py` script shields these vulnerable linear algebra blocks inside robust `try/except` fallback loops.



If a `torch.OutOfMemoryError` is caught during the chunked processing of a group, the loop immediately invokes `hw.handle_oom()`. The dispatcher then smoothly re-attempts the exact same calculation with the halved batch size returned by the hardware manager, guaranteeing eventual convergence regardless of the hardware's scale.

```{literalinclude} ../../../../src/tam/model/_dispatcher.py
:language: python
:start-after: "#: <smart_solve_router>"
:end-before: "#: </smart_solve_router>"
```

---

## Sparse Routing and In-Place Memory Tricks

For algorithmic structures like Random Forests (Random Binning Features), the theoretical feature dimension $D$ expands drastically, creating severe matrix parallelization bottlenecks {cite:p}`wu2016revisiting`. The framework utilizes aggressive low-level PyTorch optimizations inside `_tree.py` to prevent these models from crashing the server upon instantiation:

* **In-Place Bounding:** The binary leaf allocations naturally produce massive tensors. Instead of allocating a secondary normalized tensor to apply the $1/\sqrt{B}$ RKHS bound, the framework strictly enforces an in-place mutation using `.mul_(self.scale)`. This minor optimization physically prevents PyTorch from allocating an extra redundant gigabyte in VRAM.

```{literalinclude} ../../../../src/tam/model/spectrum/_tree.py
:language: python
:start-after: "#: <feature_map>"
:end-before: "#: </feature_map>"
```

* **Sparse COO Tensors:** High-dimensional symmetric penalties (like those bounding 7,000 algorithmic leaves) would natively consume massive, contiguous memory blocks as dense diagonal matrices. The `TreeEffect` class constructs its structural penalty exclusively as a sparse coordinate (`torch.sparse_coo_tensor`) object. 



This architectural choice forces the global linear algebra engine to utilize specialized sparse sub-routines, mathematically eliminating the storage of zeros and entirely circumventing the $\mathcal{O}(D^2)$ physical allocation limitation.

```{literalinclude} ../../../../src/tam/model/spectrum/_tree.py
:language: python
:start-after: "#: <penalty_matrix>"
:end-before: "#: </penalty_matrix>"
```
