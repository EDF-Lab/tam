#: <imports>
import time
import copy
import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from tabicl import TabICLRegressor
import tam as ta
from tam.common.plotting import plot_effect_with_model_and_data
from tam.evaluation.metrics import calculate_regression_metrics

from utils_cases import (
    load_data_from_disk, 
    save_data_to_disk, 
    intercept_and_save_plot, 
    intercept_and_save_text, 
    track_and_evaluate, 
    show_trackers_plot,
    setup_tabicl_environment
)

warnings.filterwarnings("ignore")
trackers = {}
trackers_AutoTAM = {}
tabicl_status = setup_tabicl_environment()
#: </imports>

#: <configuration>
CASE_NAME = "air_passengers"
FILE_NAME = "airpassengers.csv"
TARGET_COL = "value"
EVAL_METRIC = "MAPE"
#: </configuration>

#: <local_helpers>
def tune_sarima(ts_fit_log, val_actuals, dev_len, val_len, metric=EVAL_METRIC):
    pdq = list(itertools.product([0, 1], [0, 1], [0, 1]))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    best_score, best_params, best_model = float('inf'), {}, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(ts_fit_log, order=param, seasonal_order=param_seasonal).fit(disp=False)
                val_preds = np.exp(model.forecast(steps=dev_len + val_len).values[dev_len:])
                metrics = calculate_regression_metrics(val_actuals, val_preds)
                score = metrics.get(metric, float('inf'))
                if score < best_score:
                    best_score, best_params, best_model = score, {'order': param, 'seasonal_order': param_seasonal}, model
            except Exception:
                continue
    return best_params, best_model, best_score

def prepare_residuals(df_subset, base_model, lag=12):
    df_clean = df_subset.dropna(subset=['lag_log_passengers', 'time_of_year']).copy()
    df_clean['Estimated_log_passengers'] = base_model.predict(df_clean)["Estimatedlog_passengers"]
    df_clean['Residual_log_passengers'] = df_clean['log_passengers'] - df_clean['Estimated_log_passengers']
    df_clean[f'Residual_lag_{lag}'] = df_clean['Residual_log_passengers'].shift(lag)
    return df_clean.dropna(subset=[f'Residual_lag_{lag}']).copy()

def build_honest_timeline(model_key, oof_preds, warmed_preds, test_index):
    stage1_preds = oof_preds[model_key]
    stage1_vals = stage1_preds.values if hasattr(stage1_preds, 'values') else stage1_preds
    stage2_vals = warmed_preds[model_key].iloc[-len(test_index):].values
    return np.exp(np.concatenate([stage1_vals, stage2_vals]))
#: </local_helpers>

#: <data_preparation>
df = load_data_from_disk(CASE_NAME, FILE_NAME)

if df is not None:
    if df[TARGET_COL].min() <= 0:
        print(f"Warning: Cached data in {FILE_NAME} is malformed. Fetching fresh dataset.")
        df = None

if df is None:
    df = sm.datasets.get_rdataset("AirPassengers").data
    df['date'] = pd.date_range(start='1949-01-01', periods=len(df), freq='MS')
    df['log_passengers'] = np.log(df[TARGET_COL])
    df['lag_log_passengers'] = df['log_passengers'].shift(12)
    df["month"] = df['date'].dt.month
    df["year"] = df['date'].dt.year
    df['time_of_year'] = (df["month"] - 1) / 12.
    save_data_to_disk(df, CASE_NAME, FILE_NAME)

if 'date' in df.columns:
    df.set_index('date', drop=False, inplace=True)
    df.index.name = None

test_size = 24
train_df = df.iloc[:-test_size].copy()

data_dict = {
    'fit': train_df.iloc[:-36].copy(),
    'dev': train_df.iloc[-36:-24].copy(),
    'val': train_df.iloc[-24:].copy(),
    'test': df.iloc[-test_size:].copy()
}

df_stage1 = pd.concat([data_dict['fit'], data_dict['dev'], data_dict['val']])
df_full = pd.concat(data_dict.values())
#: </data_preparation>

#: <statistical_baselines_manual>
ts_fit = data_dict['fit'].set_index('date')[TARGET_COL]
ts_fit.index.freq = 'MS'

ts_train = train_df.set_index('date')[TARGET_COL]
ts_train.index.freq = 'MS'

val_actuals = data_dict['val'][TARGET_COL].values
dev_len, val_len = len(data_dict['dev']), len(data_dict['val'])

start_fit = time.time()
hw_model_fit = ExponentialSmoothing(ts_fit, seasonal_periods=12, trend='add', seasonal='mul').fit()
hw_model_train = ExponentialSmoothing(ts_train, seasonal_periods=12, trend='add', seasonal='mul').fit()
end_fit_start_pred = time.time()
hw_full_preds = np.concatenate([
    hw_model_fit.fittedvalues.values,
    hw_model_fit.forecast(dev_len + val_len).values,
    hw_model_train.forecast(len(data_dict['test'])).values
])
track_and_evaluate("Holt-Winters", hw_full_preds, end_fit_start_pred - start_fit, time.time() - end_fit_start_pred, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

start_fit = time.time()
sarima_model_fit = SARIMAX(np.log(ts_fit), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
sarima_model_train = SARIMAX(np.log(ts_train), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
end_fit_start_pred = time.time()
sarima_full_preds = np.exp(np.concatenate([
    sarima_model_fit.fittedvalues.values,
    sarima_model_fit.forecast(steps=dev_len + val_len).values,
    sarima_model_train.forecast(steps=len(data_dict['test'])).values
]))
track_and_evaluate("SARIMA", sarima_full_preds, end_fit_start_pred - start_fit, time.time() - end_fit_start_pred, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
#: </statistical_baselines_manual>

#: <statistical_baselines_tuned>
start_fit = time.time()
best_sarima_params, best_sarima_fit, _ = tune_sarima(np.log(ts_fit), val_actuals, dev_len, val_len, metric=EVAL_METRIC)
sarima_model_train = SARIMAX(np.log(ts_train), **best_sarima_params).fit(disp=False)
end_fit_start_pred = time.time()
sarima_tuned_preds = np.exp(np.concatenate([
    best_sarima_fit.fittedvalues.values,
    best_sarima_fit.forecast(steps=dev_len + val_len).values,
    sarima_model_train.forecast(steps=len(data_dict['test'])).values
]))
track_and_evaluate("SARIMA (Tuned)", sarima_tuned_preds, end_fit_start_pred - start_fit, time.time() - end_fit_start_pred, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
#: </statistical_baselines_tuned>

#: <foundation_model_tabicl_regressor>
start = time.time()
if tabicl_status is not False:
    import torch.nn.functional as F
    if not hasattr(F, '_patched_for_dtype'):
        original_linear = F.linear
        def patched_linear(input, weight, bias=None):
            if input.dtype != weight.dtype:
                input = input.to(weight.dtype)
            return original_linear(input, weight, bias)
        F.linear = patched_linear
        F._patched_for_dtype = True

    features = ["month", "lag_log_passengers"]
    target = "log_passengers"

    df_fit_clean = data_dict['fit'].dropna(subset=features + [target])
    df_train_clean = train_df.dropna(subset=features + [target])

    X_fit = df_fit_clean[features].copy()
    y_fit = df_fit_clean[[target]].copy()
    
    X_train_full = df_train_clean[features].copy()
    y_train_full = df_train_clean[[target]].copy()
    
    X_stage1 = pd.concat([data_dict['fit'], data_dict['dev'], data_dict['val']])[features].copy()
    X_stage2 = data_dict['test'][features].copy()

    # Force 'month' to string so TabICL knows it is categorical.
    for df_x in [X_fit, X_train_full, X_stage1, X_stage2]:
        df_x['month'] = df_x['month'].astype(str)

    local_path = tabicl_status if isinstance(tabicl_status, str) else None

    random_state = 42
    n_estimators = 32

    tabicl_model_stage1 = TabICLRegressor(
        verbose=False, 
        random_state=random_state, 
        n_estimators=n_estimators,
        model_path=local_path, 
        allow_auto_download=(local_path is None)
    )

    tabicl_model_stage2 = TabICLRegressor(
        verbose=False, 
        random_state=random_state, 
        n_estimators=n_estimators,
        model_path=local_path, 
        allow_auto_download=(local_path is None)
    )
    tabicl_model_stage1.fit(X_fit, y_fit.values.ravel())
    tabicl_model_stage2.fit(X_train_full, y_train_full.values.ravel())

    stage1_preds_tabicl = tabicl_model_stage1.predict(X_stage1)
    stage2_preds_tabicl = tabicl_model_stage2.predict(X_stage2)

    tabicl_full_preds = np.exp(np.concatenate([
        stage1_preds_tabicl,
        stage2_preds_tabicl
    ]))

    track_and_evaluate("TabICL_Regressor", tabicl_full_preds, 0, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
else:
    print("SKIPPING TABICL (Model not available)")
#: </foundation_model_tabicl_regressor>

#: <foundation_model_tabicl_forecaster>
start = time.time()
if tabicl_status is not False:
    import torch.nn.functional as F
    try:
        from tabicl import TabICLForecaster
    except ImportError:
        print("Warning: 'tabicl[forecast]' is not installed. TabICL forecaster skipped.")
        TabICLForecaster = None

    if TabICLForecaster is not None:
        if not hasattr(F, '_patched_for_dtype'):
            original_linear = F.linear
            def patched_linear(input, weight, bias=None):
                if input.dtype != weight.dtype:
                    input = input.to(weight.dtype)
                return original_linear(input, weight, bias)
            F.linear = patched_linear
            F._patched_for_dtype = True

        actual_target_col = "log_passengers"
        
        df_fit_context = data_dict['fit'][['date', actual_target_col]].rename(
            columns={'date': 'timestamp', actual_target_col: 'target'}
        ).copy()
        
        df_train_full_context = train_df[['date', actual_target_col]].rename(
            columns={'date': 'timestamp', actual_target_col: 'target'}
        ).copy()
        
        df_stage1_future = pd.concat([data_dict['dev'], data_dict['val']])[['date', actual_target_col]].rename(
            columns={'date': 'timestamp', actual_target_col: 'target'}
        ).copy()
        
        df_stage2_future = data_dict['test'][['date', actual_target_col]].rename(
            columns={'date': 'timestamp', actual_target_col: 'target'}
        ).copy()

        tabicl_model = TabICLForecaster()

        def predict_rolling_12_months(forecaster, context_df, future_df, horizon=12):
            """
            Keeps a rolling context, forecasting a full 12 months at every single step.
            Returns a 1D array for standard evaluation AND a dictionary containing 
            the full 12-month forecast trajectories at each step.
            """
            timeline_eval_preds = []
            rolling_trajectories = {}
            current_context = context_df.copy()

            for i in range(len(future_df)):
                # 1. Predict exactly the next 12 months based on current context
                forecast = forecaster.predict_df(current_context, prediction_length=horizon)
                forecast_vals = forecast.values.ravel()
                
                # 2. Store the full 12-month trajectory for this specific date
                current_timestamp = future_df.iloc[i]['timestamp']
                rolling_trajectories[current_timestamp] = forecast_vals.copy()
                
                # 3. Build the 1D array for the MLOps dashboard
                if i == 0:
                    # Pad the start of the evaluation array
                    timeline_eval_preds.extend(forecast_vals[:horizon - 1])
                
                # Append the horizon-th step to track long-term accuracy
                timeline_eval_preds.append(forecast_vals[horizon - 1])
                
                # 4. Roll the context forward by exactly 1 actual month
                true_actual_step = future_df.iloc[[i]]
                current_context = pd.concat([current_context, true_actual_step])
                
            return np.array(timeline_eval_preds)[:len(future_df)], rolling_trajectories

        # Generate strictly rolling 12-month forecasts
        stage1_forecast_vals, stage1_trajectories = predict_rolling_12_months(
            tabicl_model, df_fit_context, df_stage1_future, horizon=12
        )
        
        stage2_forecast_vals, stage2_trajectories = predict_rolling_12_months(
            tabicl_model, df_train_full_context, df_stage2_future, horizon=12
        )

        # Concatenate for tracking
        stage1_preds_tabicl = np.concatenate([
            df_fit_context['target'].values, 
            stage1_forecast_vals
        ])
        
        stage2_preds_tabicl = stage2_forecast_vals

        tabicl_full_preds = np.exp(np.concatenate([
            stage1_preds_tabicl,
            stage2_preds_tabicl
        ]))

        track_and_evaluate(
            "TabICL_Forecaster", 
            tabicl_full_preds, 
            0, 
            time.time() - start, 
            data_dict, 
            trackers, 
            target_col=TARGET_COL, 
            metric=EVAL_METRIC
        )

else:
    print("SKIPPING TABICL (Model not available)")
#: </foundation_model_tabicl_forecaster>

#: <additive_tam>
start = time.time()
formula = "log_passengers ~ c(time_of_year, n_cat=12, topo='fourier', ap=-8.0) + l(lag_log_passengers, ap=-30.0)"
cols_tam = ["date", "log_passengers", "time_of_year", "lag_log_passengers"]

model_tam_fit = ta.StaticTAM(formula=formula, date_col='date')
model_tam_fit.fit(data_dict['fit'][cols_tam].dropna())

model_tam_train = ta.StaticTAM(formula=formula, date_col='date')
model_tam_train.fit(train_df[cols_tam].dropna())
ws_time_fit = time.time() - start

start = time.time()
stage1_preds = pd.Series(index=df_stage1.index, dtype=float)
stage1_preds.loc[df_stage1.dropna(subset=cols_tam).index] = model_tam_fit.predict(df_stage1.dropna())["Estimatedlog_passengers"].values

stage2_preds = pd.Series(index=data_dict['test'].index, dtype=float)
stage2_preds.loc[data_dict['test'].dropna(subset=cols_tam).index] = model_tam_train.predict(data_dict['test'].dropna())["Estimatedlog_passengers"].values

ws_full_preds = np.exp(np.concatenate([stage1_preds.values, stage2_preds.values]))
track_and_evaluate("StaticTAM", ws_full_preds, ws_time_fit, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

df_train_clean = train_df.dropna(subset=['lag_log_passengers', 'time_of_year']).copy()

with intercept_and_save_plot(CASE_NAME, "effect_lag_log_passengers.png"):
    plot_effect_with_model_and_data(model=model_tam_train, data=df_train_clean, effect='lag_log_passengers')

with intercept_and_save_plot(CASE_NAME, "effect_time_of_year.png"):
    plot_effect_with_model_and_data(model=model_tam_train, data=df_train_clean, effect='time_of_year')
#: </additive_tam>

#: <adaptive_tam>
start = time.time()
d_adapt_1 = prepare_residuals(df_stage1, model_tam_fit)
model_adapt_1 = ta.AdaptiveTAM(base_model=model_tam_fit, adaptive_formula="Residual_log_passengers ~ l(Residual_lag_12)", update_interval_periods=1, training_window_periods=50, steps_per_period=1, horizon_steps=12, default_alpha_p=0)
model_adapt_1.predict_online(data=d_adapt_1)

d_adapt_2 = prepare_residuals(df_full, model_tam_train)
model_adapt_2 = ta.AdaptiveTAM(base_model=model_tam_train, adaptive_formula="Residual_log_passengers ~ l(Residual_lag_12)", update_interval_periods=1, training_window_periods=50, steps_per_period=1, horizon_steps=12, default_alpha_p=0)
model_adapt_2.predict_online(data=d_adapt_2)

series_stage1 = pd.Series(index=df_stage1.index, dtype=float)
series_stage1.loc[d_adapt_1.index] = model_adapt_1.predictions_["AdaptedEstimatedlog_passengers"].values

series_stage2 = pd.Series(index=data_dict['test'].index, dtype=float)
valid_test_idx = d_adapt_2.index.intersection(data_dict['test'].index)
series_stage2.loc[valid_test_idx] = model_adapt_2.predictions_["AdaptedEstimatedlog_passengers"].loc[valid_test_idx]

adapt_full_preds = np.exp(np.concatenate([series_stage1.values, series_stage2.values]))
track_and_evaluate("AdaptiveTAM_Lag12", adapt_full_preds, 0, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
#: </adaptive_tam>

#: <kalman_tam>
start = time.time()
model_kalman_1 = ta.KalmanTAM(base_model=model_tam_fit, kalman_formula="log_passengers ~ l(Residual_lag_12)", date_col='date', horizon_steps=12, process_noise_var=0, observation_noise_var=1)
p_all_1 = model_kalman_1.predict_online(d_adapt_1.copy())

model_kalman_2 = ta.KalmanTAM(base_model=model_tam_train, kalman_formula="log_passengers ~ l(Residual_lag_12)", date_col='date', horizon_steps=12, process_noise_var=0, observation_noise_var=1)
p_all_2 = model_kalman_2.predict_online(d_adapt_2.copy())

series_stage1.loc[d_adapt_1.index] = p_all_1["KalmanAdapted_log_passengers"].values
valid_test_idx = d_adapt_2.index.intersection(data_dict['test'].index)
series_stage2.loc[valid_test_idx] = p_all_2["KalmanAdapted_log_passengers"].loc[valid_test_idx]

kalman_full_preds = np.exp(np.concatenate([series_stage1.values, series_stage2.values]))
track_and_evaluate("KalmanTAM_Residual", kalman_full_preds, 0, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)
#: </kalman_tam>

#: <opera_tam>
start = time.time()
experts = ["AdaptiveTAM_Lag12", "KalmanTAM_Residual"]
opera_formula = "log_passengers ~ l(AdaptiveTAM_Lag12) + l(KalmanTAM_Residual)"

df_p_log_1 = df_stage1[['date', 'log_passengers', 'month']].copy()
for expert in experts:
    df_p_log_1[expert] = np.log(trackers[expert].y_pred_full[:len(df_stage1)])

opera_1 = ta.OperaTAM(formula=opera_formula, algorithm='EWA', date_col='date', group_col='month', loss_type='absolute', horizon_steps=12, eta=50)
res_1 = opera_1.predict_online(df_p_log_1.dropna().copy())

df_p_log_2 = df_full[['date', 'log_passengers', 'month']].copy()
for expert in experts:
    df_p_log_2[expert] = np.log(trackers[expert].y_pred_full)

opera_2 = ta.OperaTAM(formula=opera_formula, algorithm='EWA', date_col='date', group_col='month', loss_type='absolute', horizon_steps=12, eta=50)
res_2 = opera_2.predict_online(df_p_log_2.dropna().copy())

series_stage1.loc[res_1.index] = res_1['prediction_opera']
valid_test_idx = res_2.index.intersection(data_dict['test'].index)
series_stage2.loc[valid_test_idx] = res_2.loc[valid_test_idx, 'prediction_opera']

opera_full_preds = np.exp(np.concatenate([series_stage1.values, series_stage2.values]))
track_and_evaluate("OperaTAM", opera_full_preds, 0, time.time() - start, data_dict, trackers, target_col=TARGET_COL, metric=EVAL_METRIC)

with intercept_and_save_plot(CASE_NAME, "OperaTAM_weights_group_1.png"):
    opera_2.plot_weights(df=df_p_log_2.dropna().copy(), group_name=1)
#: </opera_tam>

#: <evaluation>
show_trackers_plot(
    data_dict, 
    trackers, 
    case_name=CASE_NAME,
    models_to_audit=["OperaTAM", "Holt-Winters"], 
    metric=EVAL_METRIC, 
    target_col=TARGET_COL, 
    title=f"{CASE_NAME.capitalize()}: MLOps Evaluation Dashboard",
    export_filename="mlops_dashboard.png"
)

show_trackers_plot(
    data_dict, 
    trackers_AutoTAM, 
    case_name=CASE_NAME,
    metric=EVAL_METRIC, 
    target_col=TARGET_COL, 
    title=f"{CASE_NAME.capitalize()}: AutoTAM Search Dashboard",
    export_filename="AutoTAM_dashboard.png"
)
#: </evaluation>

"""

################ 32Go CPU ######################

================================================================================
 GLOBAL SPLIT PERFORMANCE (MAPE)
================================================================================
                Model  Fit MAPE  Dev MAPE  Val MAPE  Train MAPE  Test MAPE  Fit Time (s)  Predict Time (s)  Test Drift (%)
0            OperaTAM    5.2501    5.3824    4.8305      5.1618     3.1429         0.000             0.118           55.72
1  KalmanTAM_Residual    5.7616    5.9486    5.1733      5.6379     3.1633         0.000             0.217           81.72
2   AdaptiveTAM_Lag12    4.9096    4.8958    4.4889      4.8027     3.2668         0.000             0.191           25.37
3           StaticTAM    5.3198    3.8221    4.0966      4.8815     3.8269         0.051             0.017            8.31
4        Holt-Winters    2.9340    2.8212    5.0503      3.3459     6.3910         0.174             0.010           79.22
5    TabICL_Regressor   12.0201    3.2742   10.2795     10.7974     6.4837         0.000             3.177           63.08
6              SARIMA   16.9755    1.7272    9.7022     13.9960     8.5154         1.162             0.011           65.59
7      SARIMA (Tuned)   17.3406    1.3077    3.1889     12.9070    13.1344        13.833             0.006          100.64
8   TabICL_Forecaster    0.0000   13.9772   13.4584      4.0894    16.5111         0.000           166.501            7.77

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                    Mean Error (Bias)  Std Error  Skewness  Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                        
KalmanTAM_Residual             7.8690    15.2537    0.0873   -0.5157          0.2566                  81.7152
OperaTAM                       8.9323    14.0707   -0.0265   -0.6277          0.1223                  55.7167
AdaptiveTAM_Lag12              9.9599    13.8404   -0.1615   -0.8017          0.0550                  25.3680
StaticTAM                     14.3020    14.2320   -0.2839   -0.7953          0.0611                   8.3111
Holt-Winters                  28.6582    15.3042   -0.2499   -0.6491          0.3519                  79.2179
TabICL_Regressor              30.1126    20.9820    0.6943    0.3562          0.1878                  63.0824
SARIMA                        39.4431    17.5699    0.1071   -0.3947          0.4930                  65.5923
SARIMA (Tuned)                60.5343    26.4372    0.2207   -0.9172          0.7573                 100.6412
TabICL_Forecaster             75.3624    72.6140    0.5284   -0.5925          0.6271                   7.7719

################ 4Go GPU ######################
PyGAM does not support GPU, it runs on 32Go CPU

================================================================================
 GLOBAL SPLIT PERFORMANCE (MAPE)
================================================================================
                Model  Fit MAPE  Dev MAPE  Val MAPE  Train MAPE  Test MAPE  Fit Time (s)  Predict Time (s)  Test Drift (%)
0            OperaTAM    5.2501    5.3824    4.8305      5.1618     3.1429         0.000             0.735           55.72
1  KalmanTAM_Residual    5.7616    5.9486    5.1733      5.6379     3.1633         0.000             1.400           81.72
2   AdaptiveTAM_Lag12    4.9096    4.8958    4.4889      4.8027     3.2668         0.000             0.083           25.37
3           StaticTAM    5.3198    3.8221    4.0966      4.8815     3.8269         0.022             0.021            8.31
4        Holt-Winters    2.9340    2.8212    5.0503      3.3459     6.3910         0.219             0.017           79.22
5    TabICL_Regressor   12.0201    3.2742   10.2795     10.7974     6.4837         0.000             3.645           63.08
6              SARIMA   16.9755    1.7272    9.7022     13.9960     8.5154         1.249             0.019           65.59
7      SARIMA (Tuned)   17.3406    1.3077    3.1889     12.9070    13.1344        15.895             0.010          100.64
8   TabICL_Forecaster    0.0000   13.9772   13.4584      4.0894    16.5111         0.000           187.372            7.77

================================================================================
 RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)
================================================================================
                    Mean Error (Bias)  Std Error  Skewness  Kurtosis  Lag-1 AutoCorr  RMSE_Drift_H2_vs_H1 (%)
Model                                                                                                        
KalmanTAM_Residual             7.8690    15.2537    0.0873   -0.5157          0.2566                  81.7152
OperaTAM                       8.9323    14.0707   -0.0265   -0.6277          0.1223                  55.7167
AdaptiveTAM_Lag12              9.9599    13.8404   -0.1615   -0.8017          0.0550                  25.3680
StaticTAM                     14.3020    14.2320   -0.2839   -0.7953          0.0611                   8.3111
Holt-Winters                  28.6582    15.3042   -0.2499   -0.6491          0.3519                  79.2179
TabICL_Regressor              30.1126    20.9820    0.6943    0.3562          0.1878                  63.0824
SARIMA                        39.4431    17.5699    0.1071   -0.3947          0.4930                  65.5923
SARIMA (Tuned)                60.5343    26.4372    0.2207   -0.9172          0.7573                 100.6412
TabICL_Forecaster             75.3624    72.6140    0.5284   -0.5925          0.6271                   7.7719

"""