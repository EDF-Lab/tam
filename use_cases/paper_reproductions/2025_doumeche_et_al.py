"""
Electricity Load Forecasting Benchmark: 2011 vs 2025 Architectures
==================================================================
This script benchmarks the evolution of Additive Models for time-series 
forecasting using the `StaticTAM` framework during the 2022 Energy Crisis.

Architectures Compared:
-----------------------
1. PyGAM Baseline: Standard local B-splines, looped sequentially over 48 half-hours.
2. Pierrot & Goude (2011): Local B-Splines with finite-difference penalties natively 
   grouped in StaticTAM.
3. Doumèche et al. (2025): Global Fourier bases with Sobolev (spectral) regularization 
   and continuous Fourier topologies for categorical transitions.
"""

#: <imports>
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tam as ta
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.filterwarnings("ignore")

try:
    from pygam import LinearGAM, s, l, f
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False
    print("Warning: 'pygam' is not installed. PyGAM baseline skipped.")

from tam.common.utils import load_national_dataset
from utils_cases import track_and_evaluate, show_trackers_plot

trackers = {}
#: </imports>

#: <configuration>
CASE_NAME = "doumeche_2025"
TARGET_COL = "Load"
EVAL_METRIC = "RMSE"

make_these_models = ["pygam", "tam_spline_2011", "tam_fourier_2025", "fourier_grid", "fourier_auto"]
#: </configuration>

#: <data_preparation>
print("Loading and preparing dataset...")

df = load_national_dataset()
df = df.sort_values('date').reset_index(drop=True)

df['day_type_week_sat_sun'] = df['day_type_week'].isin([5, 6]).astype(float)
df['trend'] = np.arange(len(df))

features = [
    'Load_d1', 'day_type_week', 'temperature_smooth_950', 
    'temperature_max_smooth_990', 'temperature_min_smooth_950', 
    'toy', 'day_type_week_sat_sun', 'trend'
]

df = df.dropna(subset=[TARGET_COL, 'tod'] + features).copy()

year_split = 2022
mask_train = (df['year'] < year_split)
mask_dev = (df['year'] == year_split) 
mask_val = (df['year'] == year_split + 1)
mask_test = (df['year'] == year_split + 2)

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

# =====================================================================
# 1. PyGAM Baseline (Splines)
# =====================================================================
if "pygam" in make_these_models and HAS_PYGAM:
    print("\n[1/5] Training PyGAM Baseline...")
    
    dico = {feat: i for i, feat in enumerate(features)}
    gam_terms = (
        l(dico['Load_d1']) + 
        s(dico['temperature_smooth_950'], n_splines=20) + 
        s(dico['temperature_max_smooth_990'], n_splines=20) + 
        s(dico['temperature_min_smooth_950'], n_splines=20) + 
        s(dico['toy'], n_splines=20, basis='cp') + 
        f(dico['day_type_week_sat_sun']) + 
        f(dico['day_type_week']) + 
        l(dico['trend'])
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
            gam_fit = LinearGAM(gam_terms).fit(df_stage1.loc[m_s1, features], df_stage1.loc[m_s1, TARGET_COL])
            fit_time = fit_time + time.time() - start
            start = time.time()
            gam_preds_s1.loc[m_s1] = gam_fit.predict(df_stage1.loc[m_s1, features])
            predict_time = predict_time + time.time() - start
            gam_train = LinearGAM(gam_terms).fit(train_df.loc[m_tr, features], train_df.loc[m_tr, TARGET_COL])
            fit_time = fit_time + time.time() - start
            start = time.time()
            if m_te.sum() > 0:
                gam_preds_te.loc[m_te] = gam_train.predict(data_dict['test'].loc[m_te, features])
            predict_time = predict_time + time.time() - start

        gam_full_preds = np.concatenate([gam_preds_s1.values, gam_preds_te.values])
        track_and_evaluate("PyGAM (Baseline)", gam_full_preds, fit_time, predict_time, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
    except Exception as e:
        print(f"PyGAM baseline failed: {e}")

# =====================================================================
# 2. StaticTAM (Pierrot & Goude 2011 - Local B-Splines)
# =====================================================================
if "tam_spline_2011" in make_these_models:
    print("\n[2/5] Training StaticTAM (2011 Spline Architecture)...")
    start = time.time()
    
    formula_spline_raw = f"""
        {TARGET_COL} ~ 
        l(Load_d1, ap=-30) +  
        s(temperature_smooth_950, k=20, deg=3, p=2, ap=-5) +                
        s(temperature_max_smooth_990, k=20, deg=3, p=2, ap=-5) + 
        s(temperature_min_smooth_950, k=20, deg=3, p=2, ap=-5) +                     
        s(toy, k=20, deg=3, p=2, ap=-5) + 
        c(day_type_week_sat_sun, topo='nominal', ap=-5) +      
        c(day_type_week, topo='nominal', ap=-5) + 
        l(trend, ap=-5)
    """
    formula_spline = " ".join(formula_spline_raw.split())
    
    model_spline = ta.StaticTAM(formula=formula_spline, date_col='date', group_col='tod')
    model_spline.fit(train_df[cols_ws])
    time_fit_spline = time.time() - start

    start = time.time()
    preds_s1 = model_spline.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values
    preds_te = model_spline.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values
    
    track_and_evaluate("StaticTAM (2011 Splines)", np.concatenate([preds_s1, preds_te]), time_fit_spline, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 3. StaticTAM (Doumèche 2025 - Global Fourier/Sobolev)
# =====================================================================
if "tam_fourier_2025" in make_these_models:
    print("\n[3/5] Training StaticTAM (2025 Fourier Architecture)...")
    start = time.time()
    
    formula_fourier_raw = f"""
        {TARGET_COL} ~ 
        l(Load_d1, ap=-30) +
        f(temperature_smooth_950, m=6, s=2, ap=-3) +
        f(temperature_max_smooth_990, m=6, s=2, ap=-3) +
        f(temperature_min_smooth_950, m=6, s=2, ap=-3) +
        f(toy, m=6, s=2, ap=-3) +
        c(day_type_week_sat_sun, topo='fourier', ap=-3) +
        c(day_type_week, topo='fourier', ap=-3) +
        l(trend, ap=-5)
    """
    formula_fourier = " ".join(formula_fourier_raw.split())
    
    model_fourier = ta.StaticTAM(formula=formula_fourier, date_col='date', group_col='tod')
    model_fourier.fit(train_df[cols_ws])
    time_fit_fourier = time.time() - start

    start = time.time()
    preds_s1 = model_fourier.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values
    preds_te = model_fourier.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values
    
    track_and_evaluate("StaticTAM (2025 Fourier)", np.concatenate([preds_s1, preds_te]), time_fit_fourier, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 4. StaticTAM (Grid Search Optimization over Fourier Params)
# =====================================================================
if "fourier_grid" in make_these_models:
    print("\n[4/5] Training StaticTAM (Fourier Grid Search)...")
    start = time.time()
    
    formula_grid_raw = f"""
        {TARGET_COL} ~ 
        l(Load_d1, ap='ap_load') +
        f(temperature_smooth_950, m='m_temp', s='s_temp', ap='ap_temp') +
        f(temperature_max_smooth_990, m='m_temp', s='s_temp', ap='ap_temp') +
        f(temperature_min_smooth_950, m='m_temp', s='s_temp', ap='ap_temp') +
        f(toy, m='m_toy', s='s_toy', ap='ap_toy') +
        c(day_type_week_sat_sun, topo='fourier', ap='ap_cat') +
        c(day_type_week, topo='fourier', ap='ap_cat') +
        l(trend, ap='ap_trend')
    """
    formula_grid = " ".join(formula_grid_raw.split())
    
    WS_GRID_CONFIG = {
        'ap_load': [-30.0, -10.0],
        'm_temp': [4, 6], 's_temp': [1, 2], 'ap_temp': [-6.0, -3.0],
        'm_toy': [6],  's_toy': [2], 'ap_toy': [-6.0, -3.0],
        'ap_cat': [-6.0, -3.0],
        'ap_trend': [-5.0, 0.0]
    }

    base_model_grid = ta.StaticTAM(formula=formula_grid, group_col="tod", date_col="date")
    best_base_model = base_model_grid.grid_search_fit(
        data_train=df_train_sub[cols_ws],
        data_val=df_val_sub[cols_ws],
        grid_search_config=WS_GRID_CONFIG
    )

    best_base_model.fit(train_df[cols_ws])
    time_fit_grid = time.time() - start

    start = time.time()
    preds_s1 = best_base_model.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values
    preds_te = best_base_model.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    track_and_evaluate("StaticTAM (Fourier Grid)", np.concatenate([preds_s1, preds_te]), time_fit_grid, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

# =====================================================================
# 5. StaticTAM (Auto Fit Discrete List on Fourier)
# =====================================================================
if "fourier_auto" in make_these_models:
    print("\n[5/5] Training StaticTAM (Fourier AutoFit)...")
    start = time.time()
    
    model_auto = ta.StaticTAM(formula=formula_fourier, group_col="tod", date_col="date")
    model_auto.auto_fit(train_df[cols_ws], alpha_p_list=[p for p in range(-15, 2, 2)], gamma=2.0)
    time_fit_auto = time.time() - start

    start = time.time()
    preds_s1 = model_auto.predict(df_stage1[cols_ws])[f"Estimated{TARGET_COL}"].values
    preds_te = model_auto.predict(data_dict['test'][cols_ws])[f"Estimated{TARGET_COL}"].values

    track_and_evaluate("StaticTAM (Fourier AutoFit)", np.concatenate([preds_s1, preds_te]), time_fit_auto, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

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
    forecast_smoothing='W', 
    title=f"Evolution of Additive Load Forecasting (2011 vs 2025)",
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
0     StaticTAM (2011 Splines)  1589.2545  1427.8147  1542.4993   1570.7578  1565.4677         1.154             1.131          -11.01
1  StaticTAM (Fourier AutoFit)  1627.4630  1418.3702  1553.2478   1602.5282  1573.9243         1.012             0.622          -10.94
2             PyGAM (Baseline)  1622.9534  1434.9088  1528.9668   1597.9963  1582.7181        19.116             1.244           -9.91
3     StaticTAM (Fourier Grid)  1677.1419  1388.6861  1550.0660   1641.1498  1619.6353         4.524             0.449           -8.71
4     StaticTAM (2025 Fourier)  1715.5644  1447.9565  1587.6533   1681.0914  1635.6038         0.479             0.582          -10.25

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                             Mean Error (Bias)  Std Error  Skewness  Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                                 
StaticTAM (2025 Fourier)             -119.2708  1631.2493   -2.0624   15.4512          0.9876                 -10.2468
StaticTAM (Fourier Grid)             -137.1722  1613.8161   -2.2160   16.6868          0.9871                  -8.7133
StaticTAM (Fourier AutoFit)          -153.2417  1566.4465   -1.9277   13.6543          0.9869                 -10.9447
PyGAM (Baseline)                     -156.5984  1574.9519   -1.9038   13.6774          0.9870                  -9.9107
StaticTAM (2011 Splines)             -166.4243  1556.5963   -1.6227   11.3713          0.9868                 -11.0116

################ 4Go GPU ######################
PyGAM does not support GPU, it runs on 32Go CPU

================================================================================
 GLOBAL SPLIT PERFORMANCE (RMSE)
================================================================================
                         Model   Fit RMSE   Dev RMSE   Val RMSE  Train RMSE  Test RMSE  Fit Time (s)  Predict Time (s)  Test Drift (%)
0     StaticTAM (2011 Splines)  1589.2545  1427.8147  1542.4993   1570.7578  1565.4677         2.507             1.406          -11.01
1  StaticTAM (Fourier AutoFit)  1627.4630  1418.3702  1553.2478   1602.5282  1573.9243         2.718             0.999          -10.94
2             PyGAM (Baseline)  1622.9534  1434.9088  1528.9668   1597.9963  1582.7181        44.436             3.305           -9.91
3     StaticTAM (Fourier Grid)  1677.1419  1388.6861  1550.0660   1641.1498  1619.6353         3.973             0.925           -8.71
4     StaticTAM (2025 Fourier)  1715.5644  1447.9565  1587.6533   1681.0914  1635.6038         0.739             0.995          -10.25

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                             Mean Error (Bias)  Std Error  Skewness  Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                                 
StaticTAM (2025 Fourier)             -119.2708  1631.2493   -2.0624   15.4512          0.9876                 -10.2468
StaticTAM (Fourier Grid)             -137.1722  1613.8161   -2.2160   16.6868          0.9871                  -8.7133
StaticTAM (Fourier AutoFit)          -153.2417  1566.4465   -1.9277   13.6543          0.9869                 -10.9447
PyGAM (Baseline)                     -156.5984  1574.9519   -1.9038   13.6774          0.9870                  -9.9107
StaticTAM (2011 Splines)             -166.4243  1556.5963   -1.6227   11.3713          0.9868                 -11.0116

"""