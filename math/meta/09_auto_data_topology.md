# AutoTAM Data Topology and Structural Safeguards

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/09_auto_data_topology_code.md)
  * **Related AutoTAM topic:** [Evolutionary Orchestrator](08_auto_orchestrator.md)

Before the TAM core engine can solve the mathematical optimization problem, the raw input data must be rigorously structured and bounded. When automating model discovery, algorithms can easily generate mathematically invalid combinations that crash the solver.

This chapter translates the complex scientific mechanisms behind the **Data Profiler**, **Feature Engineer**, and **Effect Selector** into foundational mathematical concepts and understandable algorithms.

-----

## 1\. The Data Profiler: Outliers and Temporal Grids

The Data Profiler acts as the first line of defense. It handles two major risks: extreme statistical anomalies and irregular time tracking.

### Group-Aware Interquartile Range (IQR)

Standard machine learning often removes outliers using the Standard Deviation ($\sigma$). However, this is mathematically fragile because a single massive outlier artificially inflates $\sigma$, hiding other anomalies. Instead, we rely on the **Interquartile Range (IQR)**.

![Tukey fences: the box spans Q1 to Q3 and values beyond 1.5 IQR from the box are flagged](../../_static/iqr_tukey_fences.png)

If we sort our data and find the 25th percentile ($Q_1$) and 75th percentile ($Q_3$), the IQR is simply the distance between them:
$$IQR = Q_3 - Q_1$$

The algorithm establishes strict "safe boundaries" using the standard Tukey method:
$$B_{safe} = \left[ Q_1 - 1.5 \times IQR, \ Q_3 + 1.5 \times IQR \right]$$

**Why "Group-Aware"?** If we are forecasting electricity for an entire country, applying a single global bound is disastrous. A normal baseline load in a major city would be flagged as a massive anomaly if compared to a rural village. The algorithm computes these $B_{safe}$ bounds *independently* for each entity (e.g., per substation) to preserve local physical realities.

### Time Continuity and the Lag Operator

Time series forecasting fundamentally relies on looking at the past to predict the future. In mathematics, this is governed by the **Lag Operator** ($L$):
$$L^k Y_t = Y_{t-k}$$

For this equation to work, $t$ must exist on a strict, equally spaced integer grid (like steps on a ladder). Raw industrial data is often irregular due to sensor failures. The profiler calculates the median time difference between all rows ($\Delta t$) and forces the dataset onto a continuous mathematical grid by inserting empty rows and forward-filling the missing values. If this is not done, a lag of "24 hours" might accidentally fetch data from 3 days ago, destroying the model's physical logic.

-----

## 2\. The Feature Engineer: Mathematical Augmentation and Stability

The Feature Engineer safely enriches the dataset with chronological memory without breaking the linear algebra solver.

### Smoothing via EWMA and Rolling Windows

To capture trends, the algorithm generates smoothing features. The most prominent is the **Exponentially Weighted Moving Average (EWMA)**. Unlike a simple average, EWMA applies an exponential decay to older data, giving more weight to recent events.

Mathematically, the smoothed value $S_t$ at time $t$ is defined as:
$$S_t = \alpha X_t + (1 - \alpha) S_{t-1}$$

Here, $\alpha$ (the decay factor) is dynamically calculated based on the dataset's maximum autocorrelation, ensuring the algorithm learns the natural "memory" of the specific signal without human guessing.

### The Pearson Filter and Multicollinearity

When you automatically generate dozens of rolling windows, you inevitably create features that are almost identical (e.g., a 24-hour average vs. a 25-hour average). This leads to **Multicollinearity**.

In the core TAM solver, we must invert a massive covariance matrix ($\Phi^\top \Phi$). In linear algebra, if two columns in a matrix are nearly perfectly correlated (parallel), the determinant of the matrix approaches zero. Dividing by near-zero causes the matrix inversion to "explode" (singularity), crashing the Conjugate Gradient solver .

To prevent this, the algorithm calculates the Pearson correlation coefficient ($\rho$) between all generated features. If $|\rho| > 0.95$, the features are deemed too mathematically similar, and the redundant feature is strictly purged to guarantee a safe matrix inversion.

-----

## 3\. The Effect Selector: Topological Bounds and Covariate Locks

The Effect Selector decides *which* mathematical equations (Splines, Fourier, Trees) are allowed to be applied to *which* data columns.

### Defining Data Topology

The algorithm classifies raw data into three mathematical topologies to avoid invalid operations:

1.  **Discrete:** If a column has very few unique values (e.g., Day of the Week), it is restricted to Categorical or Tree-based effects. Applying continuous calculus (like a Spline) to discrete steps causes interpolation errors.
2.  **Sparse:** If a column is mostly zeros ($\ge 80\%$), continuous smoothing fails. It is restricted to RBF kernels or Trees.
3.  **Continuous:** Standard decimal data (like Temperature), eligible for all smooth bases (Splines, Wavelets, Fourier).

### The Strict Covariate Lock

In a Generalized Additive Model (GAM), the final prediction is the sum of isolated effects. If we want to understand how temperature affects energy load, we look at the partial dependence curve for temperature, $f(Temp)$.

If the automated evolutionary engine is left unchecked, it might apply a Spline $s(Temp)$, a Fourier series $f(Temp)$, and a Polynomial $p(Temp)$ simultaneously. When this happens, the mathematical subspace overlaps. The model can no longer mathematically isolate *which* base caused the change, destroying the interpretability of the model.

The **Covariate Lock** is a strict algorithm constraint: it counts how many mathematical bases are actively using a single feature in a given formula. If it exceeds a maximum threshold (e.g., 2), the formula is mathematically rejected. This prevents formula bloat and guarantees that the model remains explainable.

### Dynamic Geometric Scaling

Mathematical bases have architectural hyperparameters. For example, P-Splines require $k$ "knots" (anchor points), and Radial Basis Functions (RBF) require $c$ "centers".

If an algorithm sets $k=50$ knots on a dataset that only has 30 rows, the system becomes underdetermined (more variables to solve for than actual data points), resulting in infinite solutions. The Effect Selector dynamically bounds these hyperparameters using the geometric size of the dataset ($N$), ensuring $k \ll N$ so the mathematical problem always remains stable and overdetermined.