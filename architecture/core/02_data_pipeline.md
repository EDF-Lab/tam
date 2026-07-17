# The Engineering of Normalization and Padding

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related mathematical theory:** [See the Mathematical Theory](../../math/core/02_tensorization.md)
  
This chapter covers the technical aspects of data preparation. It explains how the theoretical principles of N-Dim Broadcasting are implemented in Python via PyTorch and Pandas, ensuring numerical stability and compatibility with massively parallel GPU architectures.

## Tensor Uniformization: Group Balancing

The Primal solver operates exclusively on structured tensors of shape $(G \times N \times D)$ for base static training. A major technical constraint emerges when dealing with panel data econometrics {cite:p}`arellano2003panel`: in reality, isolated time series (groups) rarely share the exact same length $N$. 

The `utils.py` script resolves this issue via the `_balance_groups` function, which enforces the mathematical uniformity of the temporal dimension through two distinct strategies:

* **The `drop` method (Truncation):** Truncates all series to the length of the shortest series present in the dataset, formally guaranteeing a perfectly balanced panel. 
* **The `fill` method (Temporal Padding):** Identifies the longest series and pads shorter series by cloning their last known row. To prevent indexing conflicts, the script calculates a time step (`delta`) and generates incremental fake dates (`fake_date`).

```{literalinclude} ../../../../src/tam/common/utils.py
:language: python
:start-after: "#: <balance>"
:end-before: "#: </balance>"
```

A boolean mask is strictly preserved during this padding phase. Once the model outputs its final predictions, this mask is applied to strip away the artificially padded rows, ensuring the final output exactly matches the user's initial input dimensions.

## Affine Normalization and 3D Stacking



Once balanced, the data must be projected into the mathematical $[-1, 1]$ interval. This affine transformation is not merely for numerical scaling; it is a strict geometric requirement. Orthogonal bases like Chebyshev polynomials are only mathematically stable and bounded on the $[-1, 1]$ domain, preventing the catastrophic Runge phenomenon {cite:p}`rivlin1990chebyshev`. Furthermore, standardizing the input space guarantees that the resulting global covariance matrix remains numerically well-conditioned during the exact Primal inversion {cite:p}`lecun1998efficient`.

Instead of looping through groups in Python, the `_transform_data_stacked` function uses vectorized Pandas operations to apply the affine transformation:

$$X_{norm} = 2 \times \frac{X - \min}{\max - \min} - 1$$

After normalization, the flat 2D DataFrame is explicitly reshaped into a strictly dimensional 3D PyTorch tensor of shape `(n_groups, n_time_steps, n_features)`.

```{literalinclude} ../../../../src/tam/model/_data.py
:language: python
:start-after: "#: <transform_stacked>"
:end-before: "#: </transform_stacked>"
```

## Vectorized Sliding Windows (Adaptive Online)



For the Adaptive meta-learner, managing the continuous shift of historical bounds is essential for online GAM selection and adaptation to concept drift {cite:p}`das2025automl`. Constructing rolling historical windows iteratively in Python would create a massive CPU bottleneck. 

The `_transform_data_adaptive` function bypasses this by utilizing advanced PyTorch tensor indexing to formalize the adaptive online approach natively on the hardware {cite:p}`doumeche2025forecasting`.

* **Index Calculation:** It calculates the valid starting points by stepping backward from the end of the series.
* **Offset Broadcasting:** It builds 1D `train_offsets` and `predict_offsets` tensors using `torch.arange`.
* **Advanced Indexing:** Complete 4D windows are extracted instantaneously via tensor addition by broadcasting the offsets against the reshaped start indices (`start_indices.view(-1, 1) + train_offsets`).

```{literalinclude} ../../../../src/tam/model/_data.py
:language: python
:start-after: "#: <transform_adaptive>"
:end-before: "#: </transform_adaptive>"
```

**The Flattening Optimization for VRAM Management:** It is important to note that the core mathematical solver (`_math.py`) is intrinsically dimension-agnostic. It relies on PyTorch's N-dimensional broadcasting (`...`) and dynamic shape expansion to resolve linear systems of any rank. 

However, before sending the 4D tensors to the solver, the orchestrator explicitly flattens the `Groups` and `Windows` dimensions into a single combined batch dimension (`total_items = n_groups * n_windows`) using a memory-free `.view()` operation. 

This was a deliberate engineering choice for performance and memory safety. By collapsing the data into a pseudo-3D shape, the hardware manager can easily chunk the massive historical workload into safe 1D slices (`start_idx:end_idx`), entirely avoiding the complex 2D indexing logic that would be required to prevent Out-Of-Memory (OOM) errors.

## Reassembly and Decomposition

After the core engine computes the predictions (or decomposes them per-effect), the multidimensional PyTorch tensors must be safely mapped back to the user's original 2D Pandas DataFrame. 

The `_reassemble_decomposed_predictions` function reverses the stacking process. It flattens the predicted tensors along the batch dimensions and precisely aligns them against the original `unique_groups` order to guarantee absolute data integrity.

```{literalinclude} ../../../../src/tam/model/_data.py
:language: python
:start-after: "#: <reassemble>"
:end-before: "#: </reassemble>"
```

## 🛡️ Chronological Tensor Alignment & Reassembly Protection

A critical architectural requirement for Generalized Additive Models handling sequential data (like autoregressive lags or temporal splines) is that row-adjacency in the PyTorch tensor must strictly equal chronological adjacency. If a user passes a shuffled Pandas DataFrame to the model, the structural penalty matrix ($P$) would penalize random noise instead of chronological differences, causing catastrophic optimization failure.

To guarantee mathematical safety, the data pipeline enforces a strict two-way sorting mechanism using `date_col`:

### 1. Pre-Tensorization Sort (`_transform_data_stacked`)
Before projecting empirical data into the RKHS feature map ($\Phi$), the pipeline intercepts the per-group Pandas DataFrame and forces a chronological sort:
`group_data = data[data[group_col] == group_name].sort_values(date_col).reset_index(drop=True)`
This guarantees that the resulting 3D Tensor `[Groups, Time, Features]` is structurally perfect for the downstream GPU solvers, regardless of the input data's initial condition.

### 2. Reassembly Index Protection (`_reassemble_predictions`)
When the GPU finishes resolving the coefficients and returns the predicted tensor, the pipeline must map these chronological predictions back onto the user's original (potentially shuffled) DataFrame. 
To prevent silent row-scrambling (zipping sorted tensors to unsorted indices), the pipeline dynamically reconstructs the chronological index array prior to assignment:
```python
if date_col is not None and date_col in original_data.columns:
    group_indices_full = original_data.loc[group_indices_full].sort_values(date_col).index

```

This bidirectional safeguard completely neutralizes order-sensitivity vulnerabilities. If the user provides cross-sectional data with no temporal component (`date_col=None`), the pipeline defaults to the internal `__dummy_date__` mechanic, locking the tensors to the user's exact input sequence.
