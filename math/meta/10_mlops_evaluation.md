# Empirical Evaluation and MLOps Diagnostics

**Navigation:**
  * **Theory introduction:** [See the Intro](../../THEORY.md)
  * **Related code architecture:** [See the Code Architecture](../../architecture/meta/10_mlops_tracking_code.md)
  * **Related inference topic:** [Statistical Diagnostics](07_statistical_diagnostics.md)
  
While the Primal exact solver and the `DiagnosticsTAM` module provide "Glass-Box" statistical inference on the training data (e.g., T-statistics and EDoF), modern Machine Learning Operations (MLOps) require rigorous empirical validation on out-of-sample data.

This chapter establishes the mathematical rationale for the specific error metrics and tracking algorithms implemented in the framework's evaluation engine.

---

## 1. The Mathematics of Forecasting Metrics

The evaluation engine computes a diverse suite of metrics because no single mathematical norm captures all dimensions of predictive failure {cite:p}`hyndman2006another`. 



### $L_1$ and $L_2$ Loss Metrics
The fundamental scale-dependent metrics measure the absolute magnitude of the error:

* **Mean Absolute Error (MAE):**

  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Y_i - \hat{Y}_i|$$

  MAE penalizes errors linearly, making it mathematically robust to massive, isolated target outliers. 

* **Root Mean Square Error (RMSE):**

  $$\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^N (Y_i - \hat{Y}_i)^2 }$$

  Because RMSE is based on the $L_2$ norm, it penalizes the variance of the errors {cite:p}`chai2014root`. A model with a low MAE but a high RMSE indicates that while it is generally accurate, it occasionally makes catastrophic forecasting errors.

### Relative Percentage Errors (The SMAPE Advantage)
In industrial datasets (e.g., varying smart meters), targets exist on vastly different scales. Scale-independent metrics are required to average performance across heterogeneous topologies.

* **Mean Absolute Percentage Error (MAPE):**

  $$\text{MAPE} = \frac{100}{N} \sum_{i=1}^N \left| \frac{Y_i - \hat{Y}_i}{Y_i} \right|$$

  While highly interpretable, MAPE possesses a severe mathematical asymmetry: it penalizes over-forecasting ($\hat{Y}_i > Y_i$) exponentially more heavily than under-forecasting. Furthermore, if the true target $Y_i = 0$, the metric explodes to infinity, instantly crashing automated evolutionary pipelines (AutoTAM).

* **Symmetric Mean Absolute Percentage Error (SMAPE):**

  $$\text{SMAPE} = \frac{100}{N} \sum_{i=1}^N \frac{|Y_i - \hat{Y}_i|}{(|Y_i| + |\hat{Y}_i|)/2}$$

  To construct a mathematically safe environment for the `AutoTAM` orchestrator, the framework relies heavily on SMAPE {cite:p}`hyndman2006another`. By dividing by the average of the true and predicted values, SMAPE strictly bounds the maximum error for any single observation to exactly $200\%$. This guarantees that a single zero-target anomaly cannot destabilize the global fitness function during evolutionary hyperparameter search.

---

## 2. Temporal Degradation Tracking

A core tenet of time-series forecasting is that the assumption of exchangeability (i.i.d.) is false. Data generating processes undergo Concept Drift over time.

To quantify this, the `detect_temporal_degradation` algorithm splits a contiguous, out-of-sample test array into two chronological halves: $\mathcal{H}_1$ and $\mathcal{H}_2$. 
It computes the performance ratio:

$$\text{Degradation} (\%) = \left( \frac{\text{RMSE}_{\mathcal{H}_2} - \text{RMSE}_{\mathcal{H}_1}}{\text{RMSE}_{\mathcal{H}_1}} \right) \times 100$$

If this metric yields $+20\%$, it mathematically proves that the model's structural physics are actively decaying, signaling the operational need to trigger the `AdaptiveTAM` or `KalmanTAM` meta-learners.

---

## 3. Residual Autocorrelation (The Durbin-Watson Proxy)

If a Generalized Additive Model perfectly captures the conditional expectation $\mu(X)$, the residuals $\epsilon_t = Y_t - \hat{Y}_t$ must be pure White Noise ($\mathbb{E}[\epsilon_t \epsilon_{t-k}] = 0$ for all $k > 0$).



If the residuals exhibit serial correlation, it proves the current topology is missing a critical time-dependent feature (e.g., an unmodeled daily seasonality or an auto-regressive tensor product). 

The `analyze_residuals` module computes the **Lag-1 Autocorrelation** ($\rho_1$) as a highly efficient computational proxy for the canonical Durbin-Watson ($DW$) statistic {cite:p}`durbin1950testing`. 

The classical $DW$ test evaluates:

$$DW = \frac{\sum_{t=2}^N (\epsilon_t - \epsilon_{t-1})^2}{\sum_{t=1}^N \epsilon_t^2}$$

Algebraically, this expands to:

$$DW \approx 2(1 - \rho_1)$$

By computing the simple Pearson correlation between $\epsilon_t$ and $\epsilon_{t-1}$:

$$\rho_1 = \frac{\text{Cov}(\epsilon_t, \epsilon_{t-1})}{\text{Var}(\epsilon_t)}$$

The orchestrator instantly evaluates the structural integrity of the time-domain. If $\rho_1 \gg 0$, the $DW$ statistic approaches $0$, signaling severe positive autocorrelation and triggering the `AutoTAM` knowledge graph to propose deeper temporal spline expansions.