# Evolutionary Orchestration and Multi-Fidelity AutoML

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/08_auto_orchestrator_code.md)
  * **Related AutoTAM topic:** [Data Topology & Routing](09_auto_data_topology.md)

This chapter details the complete algorithmic pipeline governing the automated discovery and assembly of TAM models. By unifying discrete topological search with continuous convex optimization, the `AutoTAM` orchestrator resolves the combinatorial explosion inherent to adaptive model discovery.

## The Combinatorial Explosion in Online GAMs

Reliable short-term forecasts for non-stationary signals (such as electricity load) can be obtained by combining a Generalized Additive Model (GAM) with a State-Space tracker or an Error Correction Model (ECM) {cite:p}`das2025automl`.

Because TAM supports advanced multivariate topologies (like Tensor Products, RBFs, and Neural components), the additive effects are not strictly univariate. For a given time $t$, the expectation of the target variable $Y_t$ given the covariates $X_t$ is modeled as:

$$\mu(X_t) = \mathbb{E}[Y_t|X_t] = \sum_{l=1}^L h_l(\mathbf{x}_{l,t})$$

Where $h_l$ are non-parametric smoothing functions (or neural/physical operators) and $\mathbf{x}_{l,t}$ is the specific subset of covariates feeding into the $l$-th effect. 

The optimization task is immense: the orchestrator must simultaneously discover the optimal discrete formula (the physical interactions) and the continuous adaptation parameters governing the state-space. `AutoTAM` achieves this via a mathematically unified 7-step pipeline.

---

## Step 1: Topological Data Splitting (Expanding Window)



Before any learning occurs, the algorithm must establish how it will evaluate the models. 

**The Hypothesis:** Time series data inherently violates the i.i.d. (independent and identically distributed) assumption. Using standard random K-Fold cross-validation would cause "data leakage" (training on the future to predict the past). 

To rigorously simulate real-world online forecasting, `AutoTAM` implements an **Expanding Window Cross-Validation**.

**How the blocks are made:**
1. The dataset is chronologically split into a `Fit` (Train) set, a `Dev` (Development) set, and a `Val` (Validation) set.
2. The algorithm computes the median time-delta between rows to infer the physical step size (e.g., hourly, daily).
3. It sets an initial training block equal to roughly 1/3 of the historical data.
4. It defines a validation block based on the inferred step size (e.g., 30 days if daily data, 12 months if monthly).
5. The window iteratively *expands*: it trains on $T_1$, tests on $T_2$; then trains on $T_1 + T_2$, tests on $T_3$, and so on.

The final **Cross-Validation (CV) Score** is the average predictive error (e.g., RMSE) calculated across all these sequential out-of-sample test blocks. This CV score directly serves as the "fitness metric" for the evolutionary engine.

---

## Step 2: The Exact Primal Engine (The Inner Loop)

Historically, automated GAM selection relied on Penalized Iteratively Reweighted Least Squares (P-IRLS) {cite:p}`das2025automl`. This imposed a severe computational bottleneck, requiring multiple iterative $\mathcal{O}(ND^2)$ inversions until convergence. 

`AutoTAM` bypasses this by translating all topologies into the unified **Primal Spectral Space**. Every candidate architecture proposed by the evolutionary engine is solved directly and exactly via the regularized normal equations:

$$\hat{\theta} = \left( \Phi^\top \Lambda^\top \Lambda \Phi + n P \right)^{-1} \Phi^\top \Lambda^\top \Lambda Y$$

This single-pass, exact analytical evaluation allows the outer evolutionary engine to evaluate thousands of complex architectures in seconds without getting trapped in nested iterative loops.

---

## Step 3: Information Criterion & Complexity Penalties

A model that achieves a perfect CV score by memorizing noise is useless. However, `AutoTAM` must balance two different types of complexity: **Statistical Complexity** and **Structural Complexity**.

* **The Inner Loop (Statistical Complexity):** Handled entirely by the analytical Generalized Cross-Validation (GCV) solver. It calculates the *Effective Degrees of Freedom* (the trace of the Hat matrix) to automatically tune the continuous $\lambda$ penalties and prevent mathematical overfitting.
* **The Outer Loop (Structural Complexity):** Even if a massive 1,000-knot spline is heavily penalized by $\lambda$ to behave like a simple line, it still consumes massive GPU VRAM and slows down the $\mathcal{O}(ND^2)$ Primal solver. 

To enforce absolute parsimony and computational efficiency, the Evolutionary Outer Loop penalizes the **raw structural Degrees of Freedom ($k$)**-the exact physical number of columns added to the Primal matrix $\Phi$.

The raw CV Score is penalized to calculate the final Evolutionary Selection Score:

$$Score_{penalized} = Score_{raw} \times \left( 1 + \alpha_{dynamic} \frac{k}{N} \right)$$

* **$k$ (Structural Degrees of Freedom):** The exact column footprint in the Primal matrix.
* **$N$ (Number of Training Samples):** Complexity is relative. A model with $k=100$ on $N=100$ samples is pure memorization.
* **$\alpha_{dynamic}$ (Dynamic Complexity Factor):** Rather than a static scalar, `AutoTAM` implements **Simulated Annealing**. The penalty multiplier $\alpha$ starts at 1.0 in Generation 0 (allowing wild structural exploration) and linearly scales up to 5.0 in the final generations, applying crushing evolutionary pressure to force absolute parsimony.

### Exact Structural Complexity Estimation
Before training, the `AutoTAM` context parser accurately estimates $k$ based on the strict geometric realities of the underlying PyTorch operators:



| Topology / Effect | Parser Signature | Exact Structural Complexity ($k$) |
| :--- | :--- | :--- |
| **Linear / Categorical** | `l()`, `c()` | $1$ (per continuous variable / $n\_cat - 1$) |
| **Splines** | `s(k=N, deg=D)` | $N + D$ (Knots + Spline Degree) |
| **Fourier** | `f(m=N)` | $2 \times N$ (pairs of sine and cosine harmonics; `k=N` is accepted as a legacy spelling) |
| **Chebyshev / Poly** | `p(deg=N)` | $N$ (Degree of polynomial) |
| **Wavelets** | `w(n_scales=S, n_locs=L)`| $S \times L$ (Scales $\times$ Locations) |
| **RBF / Geospatial** | `rbf(n_centers=C)` | $C$ (Number of centroids) |
| **Trees / Forests** | `t(n_trees=T, max_depth=D)` or `t(n_trees=T, max_leaves=M)` | $T \times 2^D$, or $T \times M$ for a histogram (total terminal leaves) |
| **Linear Trees** | `lt(max_depth=D)` or `lt(max_leaves=M)` | $2 \times 2^D$ or $2M$ (one tree for the level, one for the slope) |
| **PID** | `pid(w=W)` | $3$ (the lag, its rolling mean over $W$ steps, its first difference) |
| **Neural** | `n(n_neurons=N, n_hidden=L)`| $N \times L \times 5$ (Heavy penalty for deep VRAM footprint) |
| **Tensor Products** | `te(eff_1, eff_2)` | ranked at $\min(\prod k_i,\ \sum k_i + \sqrt{\prod k_i})$; true width $\prod k_i$ (see the note below) |

*Note on Tensor Products:* a tensor product multiplies the inner dimensions. Mixing a 10-knot spline with a 5-harmonic Fourier series (`te(s(k=10), f(m=5))`) builds an interaction surface of $d_1 \times d_2 = (10+3) \times (2 \times 5) = 130$ columns, and that true width is what the solver allocates (`TensorProductEffect.get_n_coeffs`). Charged in full to the selection score, it would price interactions out of the search before they are ever evaluated, so the score charges a discounted surrogate instead:

$$k_{te} = \min\Big(\prod_i d_i,\; \sum_i d_i + \sqrt{\textstyle\prod_i d_i}\Big)$$

For the example, $k_{te} = \min(130,\ 23 + \sqrt{130}) = 34$ after truncation. For $d_i \ge 1$ the surrogate satisfies $\max_i d_i \le k_{te} \le \prod_i d_i$, so an interaction is never ranked as cheaper than its largest marginal nor as more expensive than its width; it is non-decreasing in every $d_i$; and when all inner dimensions grow, $k_{te} / \prod_i d_i \to 0$. It is an empirical heuristic chosen to prevent that starvation, not a complexity measure derived from effective degrees of freedom. The proofs are in [Feature Profiling, section 7](12_autotam_feature_profiling.md).

---

## Step 4: Darwinian Evolution via Knowledge Graphs (The Outer Loop)

To navigate the infinite space of possible mathematical formulas, `AutoTAM` acts as an Estimation of Distribution Algorithm (EDA). In plain terms: where a classical genetic algorithm mutates formulas blindly, an EDA keeps a probabilistic record of which building blocks (feature and basis pairs, interactions, parameter values) appeared in good formulas, and samples new formulas from that record. In `AutoTAM` that record is a **Bipartite Bayesian Knowledge Graph**.

Think of this as **Natural Selection driven by Environmental Memory**. The framework divides the search space into specialized generative nodes called **Islands**:

* `FourierIsland`: Only breeds cyclic, repeating waves.
* `TreeIsland`: Only breeds algorithmic if-then splits.
* `NeuralIsland`: Only breeds dense, hidden feature interactions.

**The Continents (Hybridization):**
Islands do not exchange DNA directly. Instead, `AutoTAM` introduces two meta-nodes:

1. **The Small Continent:** Allowed to sample from *all* islands simultaneously, but strictly forces the lowest possible hyperparameters (e.g., $k=1$, $deg=1$). This acts as the evolutionary "bacteria" anchor-guaranteeing the survival of highly diverse, ultra-fast, computationally cheap hybrid baselines.
2. **The Meta-Continent:** The apex breeder. It freely samples and combines the top-performing traits from all specialized islands to create massive, heterogeneous mega-formulas.

When sampling, the Knowledge Graph balances Epsilon-greedy exploration with Softmax temperature-scaled exploitation:

$$P(effect_k) = \frac{\exp(S_k / \tau)}{\sum_{j} \exp(S_j / \tau)}$$

Where $P(effect_k)$ is the probability of breeding a specific effect, and $S_k$ is its historical fitness (a combination of how much variance it explained in past generations minus its structural penalty).

**Evidence accounting and the repopulation budget.** Each distinct formula is one observation: its reward, its survival and its island's bandit pull are recorded once, when it is first evaluated. An elite carried into later generations is served from the evaluation cache and adds no further evidence; crediting it again every generation would let a single early draw accumulate fitness until the sampling distribution collapsed onto it. A generation of size $N$ keeps its best $\lfloor \rho N \rfloor$ formulas (survival rate $\rho$), reserves $\max(1, \operatorname{round}(\phi N))$ slots for fresh island spawns (fresh fraction $\phi$, default $0.25$), and fills the remainder with single-term ablations of the elites that have not been evaluated before. Without the reserved share, elites plus their ablations fill the population, every member is already cached after a few rounds, and the search stops producing candidates.

**Island selection via Mutant-UCB.** The orchestrator frames island selection as an infinite-armed bandit problem, utilizing the **Mutant-UCB** algorithm {cite:p}`bregere2024mutant`. By integrating evolutionary mutation operators with Upper Confidence Bound exploration, Mutant-UCB effectively searches the space without requiring any assumptions about an underlying smooth vector space or reward function {cite:p}`bregere2024mutant`. The next Island to breed is drawn in three steps. An Island pulled fewer than $m$ times is chosen first, uniformly among such Islands ($m = 3$). Otherwise, with probability $\varepsilon = 0.15$, an Island is drawn uniformly. Otherwise the upper confidence bound decides, $I_t \in \arg\max_k \{\mu_k + \sqrt{E/N_k}\}$, with mean reward $\mu_k$, pull count $N_k$ and $E = 2$. The rewards $1/(\mathrm{RMSE}/\sigma_Y)$ lie between about $1$ and $6$, so a bonus of at most $\sqrt{2} \approx 1.41$ stops mattering after a few pulls: measured runs gave $53$ to $69$ of about $95$ pulls to a single Island. The minimum budget and the uniform share keep every topology in the evidence. A candidate that fails to fit counts as a pull with zero reward, so an Island whose candidates fail cannot hold the first step forever.

A tensor product $te(h_a, h_b)$ has no single feature. Its fitness is credited to every feature it crosses (on the feature-to-$te$ edge that decides whether to build an interaction) and to the feature pair $\{a, b\}$ (the edge that chooses the partner).

Every Island may propose tensor products, at most two per formula. A product of a non-linear continuous feature with a categorical of at least two levels keeps the feature's profiled capacity: one curve per level. A product of two continuous features is a surface of $d_1 d_2$ columns, so each side takes a small fixed basis (a spline with $k = 3$ knots of degree 2, a Fourier series with $m = 2$ harmonics, or a Chebyshev polynomial of degree 4) and the surface stays within $5 \times 5 = 25$ columns.

---

## Step 5: Parsimonious Pruning (Variance Decomposition)

Natural selection produces bloat; evolution is not inherently efficient. To explicitly force parsimony, `AutoTAM` acts as an active biological pruner.

After a formula is solved via the exact primal engine, the global prediction vector $\hat{Y}$ is mathematically decomposed into the isolated contributions of each basis $h_l(\mathbf{x})$.

1.  **Variance Check:** If an effect explains less variance than a strict threshold ($\frac{\text{Var}(h_l(\mathbf{x}))}{\text{Var}(\hat{Y})} < \gamma_{prune}$), it is mathematically "killed" and stripped from the genome.
2.  **Collinearity Check:** Terms are tested in decreasing order of their variance share. If an effect competes for the exact same variance as an effect already retained (Pearson correlation $>0.98$), it is deleted to preserve matrix invertibility, so of two near-collinear effects the one explaining less of the prediction variance is removed. A tensor product therefore claims its variance before a marginal it spans, whatever the order in which the formula lists them.

Each term is matched to its own decomposed contribution by its position in the model's effect list, never by feature name: a tensor product has no single feature, and one feature may carry two bases.

**Mathematical Insight for the User:** The pruning threshold $\gamma_{prune}$ (default 0.005, or 0.5%) is configurable. It sits at 0.5% because an autoregressive lag absorbs most of the seasonal variance in-sample, leaving a structural seasonal term with under 1% of the prediction variance even though it is the component that survives a regime shift. If your data is highly noisy, you should *increase* this threshold to actively kill off spurious, low-variance correlations. If your data is highly deterministic, you can *decrease* it to capture subtle micro-effects.

---

## Step 6: Dynamic State-Space Expansion (AutoTAM vs. Das et al.)



Once the static formulas (the Champions) are finalized, they must be expanded to handle Concept Drift. 

**The Paradigm Shift from Das et al. (2025):**
In previous literature {cite:p}`das2025automl`, the evolutionary algorithm attempted to jointly discover the discrete GAM formula *and* the continuous state-space parameters (like the Kalman covariance matrix $Q$) simultaneously. While this coupled approach is highly specialized and extremely effective for capturing the precise dynamics of energy time series, it carries a massive computational cost that bottlenecks the search process.

`AutoTAM` explicitly decouples them (Hub-and-Spoke) for scalability. It finds the globally optimal static GAM *first*, freezes the physical topology, and then systematically expands it into dynamic tracking spaces:

1.  **Adaptive Error Correction Models (ECM):** Fits a local sliding-window model to predict recent residual errors.
2.  **Continuous Kalman Tracking:** Converts static coefficients into a hidden Markov state, drifting via Gaussian process noise.

The user can toggle these expansions via a boolean configuration dictionary (e.g., `{"prior": True, "kalman": True, "adaptive": False}`). 

---

## Step 7: Dual Minimax Aggregation (OPERA)



At the end of the pipeline, `AutoTAM` has generated dozens of static, adaptive, and Kalman models. Rather than selecting a single "best" model (which is mathematically fragile), it deploys a competitive, sequential game {cite:p}`cesa2006prediction`. 

The final predictions are organized into specialized sub-ensembles ("Leagues"), updated in real-time via the parameter-free MLpol (Polynomial Minimax) algorithm {cite:p}`gaillard2016opera`:

* **Ensemble Static:** Aggregates only the static base models.
* **Ensemble Kalman:** Aggregates the State-Space variants.
* **Ensemble Adaptive:** Aggregates the ECM variants.
* **Island Federation:** Takes the top champion from each independent island and aggregates their distinct mathematical topologies (e.g., mixing pure Wavelets with pure Trees).
* **AutoTAM Apex Ensemble:** The master aggregator. When dynamic (adaptive or Kalman) experts exist, it aggregates them without a top-$N$ cut, after a quality floor: an expert whose validation error $L_k$ exceeds $\rho \min_j L_j$ (default $\rho = 1.5$) is left out, and the two best experts are always kept. An adaptive expert re-fits through a regime shift, so its validation error carries over to later periods; a ratio to the best keeps a large pool for MLpol, whose regret grows only logarithmically in the number of experts, while removing experts that validation already rules out. Online aggregation (`predict_online`) weights the whole pool; the frozen `predict()` averages only the experts whose MLpol weight on the last validation row exceeds the sparsity threshold, a one-step snapshot that would otherwise hide from the sequential aggregator the experts that win later. Without dynamic experts, the Apex combines the top $N$ models regardless of their underlying physics.

By dynamically shifting weights $w_{k,t}$ strictly proportional to the positive cumulative regret $[R_{k,t}]_+$ of each expert, the framework provides mathematically guaranteed robustness against structural grid breaks and sudden concept drift.