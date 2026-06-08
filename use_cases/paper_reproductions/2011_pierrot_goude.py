"""
Pierrot & Goude (2011) - Short-Term Electricity Load Forecasting Benchmark
==========================================================================
This script benchmarks `StaticTAM` against `pygam` using the core methodologies 
proposed in Pierrot & Goude (2011).

Architectural Highlights Demonstrated:
--------------------------------------
1. Native Grouping vs. Manual Looping:
   The original paper fits 48 independent models (one for each half-hour of the day). 
   While PyGAM requires a manual Python `for` loop to achieve this, StaticTAM 
   handles it natively via `group_col='tod'`, computing the 48 models simultaneously 
   in a single, highly optimized PyTorch tensor batch.

2. The "Varying-Coefficient" Simplification:
   The original paper uses varying-coefficient interactions (e.g., Load_d1 interacting 
   with the day of the week). In PyGAM, the `by=` operator handles this via 1D scalar 
   multiplication. However, in advanced GAM theory (and StaticTAM), a true interaction 
   requires a Tensor Product `te()`, which applies a strict 2D Kronecker-sum penalty.
   
   Because these two regularization philosophies are mathematically distinct, comparing 
   PyGAM's `by=` directly to StaticTAM's `te()` results in a biased benchmark. To ensure 
   a mathematically fair 1:1 comparison of the core primal solvers and spline engines, 
   this script uses a purely decoupled, additive formulation for both models.
"""

#: <imports>
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tam as ta
from pathlib import Path

# Dynamically route the Python path to the project root for seamless execution
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

try:
    from pygam import LinearGAM, s, l, f
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False
    print("Warning: 'pygam' is not installed. PyGAM baselines skipped.")

from tam.common.utils import load_national_dataset
from utils_cases import track_and_evaluate, show_trackers_plot

trackers = {}
#: </imports>

#: <configuration>
CASE_NAME = "pierrot_goude_2011"
TARGET_COL = "Load"
EVAL_METRIC = "RMSE"

# Toggle models to run
make_these_models = ["pygam", "pygam_grid", "default", "grid", "auto_fit_1", "auto_fit_2"]
#: </configuration>

#: <data_preparation>
print("Loading and preparing dataset...")

df = load_national_dataset()
df = df.sort_values('date').reset_index(drop=True)

# Feature Engineering: Aligning with the 2011 paper requirements
df['day_type_week_sat_sun'] = df['day_type_week'].isin([5, 6]).astype(float)
df['trend'] = np.arange(len(df))

features = [
    'Load_d1', 'day_type_week', 'temperature_smooth_950', 
    'temperature_max_smooth_990', 'temperature_min_smooth_950', 
    'toy', 'day_type_week_sat_sun', 'trend'
]

df = df.dropna(subset=[TARGET_COL, 'tod'] + features).copy()

# Chronological Time-Series Split 
# Configured here to test generalization during standard regimes (pre-2020)
mask_train = (df['year'] < 2017)
mask_dev = (df['year'] == 2017) 
mask_val = (df['year'] == 2018)
mask_test = (df['year'] == 2019)

train_df = df[mask_train | mask_dev | mask_val].copy()
df_train_sub = df[mask_train].copy()
df_val_sub = df[mask_dev | mask_val].copy()

data_dict = {
    'fit': df[mask_train].copy(),
    'dev': df[mask_dev].copy(),
    'val': df[mask_val].copy(),
    'test': df[mask_test].copy()
}

df_stage1 = pd.concat([data_dict['fit'], data_dict['dev'], data_dict['val']])
cols_ws = ["date", "tod", TARGET_COL] + features

print(f"Data Split Total: {len(df)} records")
#: </data_preparation>

#: <models>

# Dictionary mapping for clean and safe PyGAM syntax
dico = {feat: i for i, feat in enumerate(features)}

# =====================================================================
# 1. PyGAM Baseline (Default/Fixed Lam)
# =====================================================================
if "pygam" in make_these_models and HAS_PYGAM:
    print("\n[1/6] Training PyGAM Baseline (Fixed Lam = 1e-5 to match TAM Default)...")
        
    # We explicitly set lam=1e-5 (and ~0 for the lag) to match StaticTAM's ap=-5 default
    gam_terms_default = (
        l(dico['Load_d1'], lam=1e-10) + 
        s(dico['temperature_smooth_950'], n_splines=20, lam=1e-5) + 
        s(dico['temperature_max_smooth_990'], n_splines=20, lam=1e-5) + 
        s(dico['temperature_min_smooth_950'], n_splines=20, lam=1e-5) + 
        s(dico['toy'], n_splines=20, lam=1e-5) + 
        f(dico['day_type_week_sat_sun'], lam=1e-5) + 
        f(dico['day_type_week'], lam=1e-5) + 
        l(dico['trend'], lam=1e-5)
    )
    
    gam_preds_s1 = pd.Series(index=df_stage1.index, dtype=float)
    gam_preds_te = pd.Series(index=data_dict['test'].index, dtype=float)

    try:
        fit_time = 0
        predict_time = 0
        for tod_val in df['tod'].unique():
            m_s1 = df_stage1['tod'] == tod_val
            m_tr = train_df['tod'] == tod_val
            m_te = data_dict['test']['tod'] == tod_val

            start = time.time()
            gam_fit = LinearGAM(gam_terms_default).fit(df_stage1.loc[m_s1, features], df_stage1.loc[m_s1, TARGET_COL])
            fit_time += time.time() - start
            
            start = time.time()
            gam_preds_s1.loc[m_s1] = gam_fit.predict(df_stage1.loc[m_s1, features])
            predict_time += time.time() - start
            
            start = time.time()
            gam_train = LinearGAM(gam_terms_default).fit(train_df.loc[m_tr, features], train_df.loc[m_tr, TARGET_COL])
            fit_time += time.time() - start
            
            start = time.time()
            if m_te.sum() > 0:
                gam_preds_te.loc[m_te] = gam_train.predict(data_dict['test'].loc[m_te, features])
            predict_time += time.time() - start

        gam_full_preds = np.concatenate([gam_preds_s1.values, gam_preds_te.values])
        track_and_evaluate("PyGAM Baseline (Fixed Lam)", gam_full_preds, fit_time, predict_time, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
    except Exception as e:
        print(f"PyGAM baseline failed: {e}")

# =====================================================================
# 1b. PyGAM Gridsearch (GCV Auto-Fit)
# =====================================================================
if "pygam_grid" in make_these_models and HAS_PYGAM:
    print("\n[2/6] Training PyGAM Gridsearch (GCV Optimization on 11 lam values)...")
        
    # Leave lam to standard so gridsearch can evaluate the 11 logspace steps
    gam_terms_search = (
        l(dico['Load_d1']) + 
        s(dico['temperature_smooth_950'], n_splines=20) + 
        s(dico['temperature_max_smooth_990'], n_splines=20) + 
        s(dico['temperature_min_smooth_950'], n_splines=20) + 
        s(dico['toy'], n_splines=20) + 
        f(dico['day_type_week_sat_sun']) + 
        f(dico['day_type_week']) + 
        l(dico['trend'])
    )
    
    gam_grid_preds_s1 = pd.Series(index=df_stage1.index, dtype=float)
    gam_grid_preds_te = pd.Series(index=data_dict['test'].index, dtype=float)

    try:
        fit_time_grid = 0
        predict_time_grid = 0
        for tod_val in df['tod'].unique():
            m_s1 = df_stage1['tod'] == tod_val
            m_tr = train_df['tod'] == tod_val
            m_te = data_dict['test']['tod'] == tod_val

            start = time.time()
            # Note: PyGAM's .gridsearch() evaluates 11 values by default
            gam_fit = LinearGAM(gam_terms_search).gridsearch(df_stage1.loc[m_s1, features].values, df_stage1.loc[m_s1, TARGET_COL].values, progress=False)
            fit_time_grid += time.time() - start
            
            start = time.time()
            gam_grid_preds_s1.loc[m_s1] = gam_fit.predict(df_stage1.loc[m_s1, features].values)
            predict_time_grid += time.time() - start
            
            start = time.time()
            gam_train = LinearGAM(gam_terms_search).gridsearch(train_df.loc[m_tr, features].values, train_df.loc[m_tr, TARGET_COL].values, progress=False)
            fit_time_grid += time.time() - start
            
            start = time.time()
            if m_te.sum() > 0:
                gam_grid_preds_te.loc[m_te] = gam_train.predict(data_dict['test'].loc[m_te, features].values)
            predict_time_grid += time.time() - start

        gam_grid_full_preds = np.concatenate([gam_grid_preds_s1.values, gam_grid_preds_te.values])
        track_and_evaluate("PyGAM Gridsearch (GCV AutoFit)", gam_grid_full_preds, fit_time_grid, predict_time_grid, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
    except Exception as e:
        print(f"PyGAM Gridsearch failed: {e}")

# =====================================================================
# 2. StaticTAM (Default Parameters - Native Grouping)
# =====================================================================
if "default" in make_these_models:
    print("\n[3/6] Training StaticTAM Default (Native group_col='tod')...")
    start = time.time()
    
    # Note on `ap` scaling: StaticTAM normalizes target data to Variance=1 internally.
    # To match standard unscaled regularization flexibility, penalties are set to 
    # lambda_p = 10^-5 (ap=-5). The linear lag term is left largely unpenalized (ap=-30).
    formula_tam = (
        f"{TARGET_COL} ~ "
        "l(Load_d1, ap=-30) +  "          
        "s(temperature_smooth_950, k=20, deg=3, p=2, ap=-5) + "               
        "s(temperature_max_smooth_990, k=20, deg=3, p=2, ap=-5) + "
        "s(temperature_min_smooth_950, k=20, deg=3, p=2, ap=-5) + "                     
        "s(toy, k=20, deg=3, p=2, ap=-5) + "
        "c(day_type_week_sat_sun, topo='nominal', ap=-5) + "      
        "c(day_type_week, topo='nominal', ap=-5) + "
        "l(trend, ap=-5)"                                     
    )
    
    model_ws_fit = ta.StaticTAM(formula=formula_tam, date_col='date', group_col='tod')
    model_ws_fit.fit(df_stage1[cols_ws])

    model_ws_train = ta.StaticTAM(formula=formula_tam, date_col='date', group_col='tod')
    model_ws_train.fit(train_df[cols_ws])
    ws_time_fit = time.time() - start

    start = time.time()
    stage1_preds_ws = pd.Series(index=df_stage1.index, dtype=float)
    stage1_preds_ws.loc[df_stage1.index] = model_ws_fit.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values

    stage2_preds_ws = pd.Series(index=data_dict['test'].index, dtype=float)
    stage2_preds_ws.loc[data_dict['test'].index] = model_ws_train.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    ws_full_preds = np.concatenate([stage1_preds_ws.values, stage2_preds_ws.values])
    track_and_evaluate("StaticTAM (Default)", ws_full_preds, ws_time_fit, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 3. StaticTAM (Grid Search Optimization - Native Grouping)
# =====================================================================
if "grid" in make_these_models:
    print("\n[4/6] Training StaticTAM Grid Search (Native group_col='tod')...")
    start = time.time()
    
    # Hyperparameters are exposed to the Grid Search via string tokens
    formula_grid = (
        f"{TARGET_COL} ~ "
        "l(Load_d1, ap='ap_load') +  "          
        "s(temperature_smooth_950, k='k_temp', deg=3, p=2, ap='ap_temp') + "               
        "s(temperature_max_smooth_990, k='k_temp', deg=3, p=2, ap='ap_temp') + "
        "s(temperature_min_smooth_950, k='k_temp', deg=3, p=2, ap='ap_temp') + "                     
        "s(toy, k='k_toy', deg=3, p=2, ap='ap_toy') + "
        "c(day_type_week_sat_sun, topo='nominal', ap='ap_cat') + "      
        "c(day_type_week, topo='nominal', ap='ap_cat') + "
        "l(trend, ap='ap_trend')"                                     
    )
    
    AP_LIST = [-10.0, -5.0, -1.0]
    K_LIST = [10, 20] 

    WS_GRID_CONFIG = {
        'ap_load': [-30.0, -10.0],
        'k_temp': K_LIST, 'ap_temp': AP_LIST,
        'k_toy': K_LIST,  'ap_toy': AP_LIST,
        'ap_cat': AP_LIST,
        'ap_trend': [-5.0, 0.0]
    }

    base_model_grid = ta.StaticTAM(formula=formula_grid, group_col="tod", date_col="date")
    
    # Leverages Multi-Start Coordinate Descent across validation axes
    best_base_model = base_model_grid.grid_search_fit(
        data_train=df_train_sub[cols_ws],
        data_val=df_val_sub[cols_ws],
        grid_search_config=WS_GRID_CONFIG
    )

    best_base_model.fit(train_df[cols_ws])
    ws_grid_time_fit = time.time() - start

    start = time.time()
    stage1_preds_grid = pd.Series(index=df_stage1.index, dtype=float)
    stage1_preds_grid.loc[df_stage1.index] = best_base_model.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values

    stage2_preds_grid = pd.Series(index=data_dict['test'].index, dtype=float)
    stage2_preds_grid.loc[data_dict['test'].index] = best_base_model.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    grid_full_preds = np.concatenate([stage1_preds_grid.values, stage2_preds_grid.values])
    track_and_evaluate("StaticTAM (Grid Optimized)", grid_full_preds, ws_grid_time_fit, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 4. StaticTAM (Auto Fit Continuous Bounds)
# =====================================================================
if "auto_fit_1" in make_these_models:
    print("\n[5/6] Training StaticTAM Auto Fit Bounds (Native group_col='tod')...")
    start = time.time()
    
    # Utilizes Generalized Cross Validation (MSP-GCV) to automatically find 
    # the optimal lambda_p penalization surface within bounded constraints.
    model_auto_1 = ta.StaticTAM(formula=formula_tam, group_col="tod", date_col="date")
    model_auto_1.auto_fit(train_df[cols_ws], alpha_p_bounds=(-30.0, 6.0), number_of_steps=13, gamma=2.0)
    auto_1_time_fit = time.time() - start

    start = time.time()
    stage1_preds_a1 = pd.Series(index=df_stage1.index, dtype=float)
    stage1_preds_a1.loc[df_stage1.index] = model_auto_1.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values

    stage2_preds_a1 = pd.Series(index=data_dict['test'].index, dtype=float)
    stage2_preds_a1.loc[data_dict['test'].index] = model_auto_1.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    a1_full_preds = np.concatenate([stage1_preds_a1.values, stage2_preds_a1.values])
    track_and_evaluate("StaticTAM (AutoFit Bounds)", a1_full_preds, auto_1_time_fit, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 5. StaticTAM (Auto Fit Discrete List)
# =====================================================================
if "auto_fit_2" in make_these_models:
    print("\n[6/6] Training StaticTAM Auto Fit List (Native group_col='tod')...")
    start = time.time()
    
    model_auto_2 = ta.StaticTAM(formula=formula_tam, group_col="tod", date_col="date")
    model_auto_2.auto_fit(train_df[cols_ws], alpha_p_list=[p for p in range(-30, 6, 3)], gamma=2.0)
    auto_2_time_fit = time.time() - start

    start = time.time()
    stage1_preds_a2 = pd.Series(index=df_stage1.index, dtype=float)
    stage1_preds_a2.loc[df_stage1.index] = model_auto_2.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values

    stage2_preds_a2 = pd.Series(index=data_dict['test'].index, dtype=float)
    stage2_preds_a2.loc[data_dict['test'].index] = model_auto_2.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    a2_full_preds = np.concatenate([stage1_preds_a2.values, stage2_preds_a2.values])
    track_and_evaluate("StaticTAM (AutoFit List)", a2_full_preds, auto_2_time_fit, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

#: </models>

#: <evaluation>
print("\nGenerating MLOps Dashboard...")
audit_models = list(trackers.keys())

show_trackers_plot(
    data_dict, 
    trackers, 
    case_name=CASE_NAME,
    models_to_audit=audit_models, 
    metric=EVAL_METRIC, 
    target_col=TARGET_COL, 
    date_col="date",
    is_timeseries=True,
    forecast_smoothing='ME',
    title=f"Pierrot & Goude (2011) Optimization Comparison",
    export_filename="mlops_dashboard.png"
)
print("Pipeline Complete.")
#: </evaluation>

"""

################ 32Go CPU ######################

================================================================================
 GLOBAL SPLIT PERFORMANCE (RMSE)
================================================================================
                            Model   Fit RMSE   Dev RMSE   Val RMSE  Train RMSE  Test RMSE  Fit Time (s)  Predict Time (s)  Test Drift (%)
0        StaticTAM (AutoFit List)  1548.7060  1542.4287  1444.2119   1530.2202  1582.6777         2.600             0.835          -12.09
1      StaticTAM (AutoFit Bounds)  1551.9726  1543.6174  1447.6214   1533.1501  1586.8492         1.306             0.839          -12.27
2      StaticTAM (Grid Optimized)  1558.6762  1548.3599  1465.4490   1541.3200  1597.8371        34.576             0.795          -13.15
3  PyGAM Gridsearch (GCV AutoFit)  1573.7996  1552.3650  1466.1536   1552.1879  1617.7197       183.728             0.967          -11.63
4             StaticTAM (Default)  1537.9896  1528.4297  1431.9910   1518.6975  1705.6027         1.245             0.753            1.09
5      PyGAM Baseline (Fixed Lam)  1561.0740  1549.1772  1451.8355   1540.8480  4118.9592        13.261             0.785          221.30

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                                Mean Error (Bias)  Std Error  Skewness   Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                                     
PyGAM Gridsearch (GCV AutoFit)          -183.3817  1607.2922   -2.6943    17.0449          0.9865                 -11.6313
StaticTAM (Default)                     -200.2063  1693.8117   -2.6858    22.7208          0.9787                   1.0877
StaticTAM (Grid Optimized)              -215.4070  1583.2508   -2.5272    16.1244          0.9881                 -13.1473
StaticTAM (AutoFit List)                -216.6747  1567.7758   -2.6430    16.9690          0.9877                 -12.0909
StaticTAM (AutoFit Bounds)              -228.3481  1570.3336   -2.6375    16.9339          0.9878                 -12.2705
PyGAM Baseline (Fixed Lam)              -247.3665  4111.5247  -34.8516  1611.9112          0.8299                 221.2971

################ 4Go GPU ######################
PyGAM does not support GPU, it runs on 32Go CPU

================================================================================
 GLOBAL SPLIT PERFORMANCE (RMSE)
================================================================================
                            Model   Fit RMSE   Dev RMSE   Val RMSE  Train RMSE  Test RMSE  Fit Time (s)  Predict Time (s)  Test Drift (%)
0        StaticTAM (AutoFit List)  1548.7060  1542.4287  1444.2119   1530.2202  1582.6777         3.720             0.610          -12.09
1      StaticTAM (AutoFit Bounds)  1551.9726  1543.6174  1447.6214   1533.1501  1586.8492         2.010             0.563          -12.27
2      StaticTAM (Grid Optimized)  1558.6762  1548.3599  1465.4490   1541.3200  1597.8371        19.823             1.082          -13.15
3  PyGAM Gridsearch (GCV AutoFit)  1573.7996  1552.3650  1466.1536   1552.1879  1617.7197       139.637             0.682          -11.63
4             StaticTAM (Default)  1537.9896  1528.4297  1431.9910   1518.6975  1705.6027         1.958             0.563            1.09
5      PyGAM Baseline (Fixed Lam)  1561.0740  1549.1772  1451.8355   1540.8480  4118.9592        12.637             0.667          221.30

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                                Mean Error (Bias)  Std Error  Skewness   Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                                     
PyGAM Gridsearch (GCV AutoFit)          -183.3817  1607.2922   -2.6943    17.0449          0.9865                 -11.6313
StaticTAM (Default)                     -200.2063  1693.8117   -2.6858    22.7208          0.9787                   1.0877
StaticTAM (Grid Optimized)              -215.4070  1583.2508   -2.5272    16.1244          0.9881                 -13.1473
StaticTAM (AutoFit List)                -216.6747  1567.7758   -2.6430    16.9690          0.9877                 -12.0909
StaticTAM (AutoFit Bounds)              -228.3481  1570.3336   -2.6375    16.9339          0.9878                 -12.2705
PyGAM Baseline (Fixed Lam)              -247.3665  4111.5247  -34.8516  1611.9112          0.8299                 221.2971

"""