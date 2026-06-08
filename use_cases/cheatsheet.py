"""
TAM (Time series Additive Model) Framework - Ultimate Cheatsheet
"""

import pandas as pd
import numpy as np
import tam as ta
import torch
import gc

from utils_cases import show_trackers_plot, seed_everything, intercept_and_save_text, intercept_and_save_plot
from tam.common.plotting import plot_effect_with_model_and_data

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CASE_NAME="cheatsheet"
seed_everything(seed=42)

# ==============================================================================
# 1. Data Generation
# ==============================================================================
dates = pd.date_range('2012-01-01', '2026-12-31', freq='D')
N, t = len(dates), np.arange(len(dates))

df = pd.DataFrame({
    'date': dates, 't': t, 
    'x1': np.random.uniform(0, 10, N), 'x2': np.random.normal(0, 1, N),
    'x3': dates.dayofweek, 'x4': dates.month, 'x10': dates.day,
    'x5': t / N, 'x6': np.random.randn(N) * 2,
    'x7': np.random.binomial(1, 0.5, N), 'x8': np.sin(t / 30.0)
})

dy = df['date'].dt.year
y_comps = [
    3*df['x1'] + 5*df['x2'] + 2*df['x3'] + 4*df['x4'] + 5*df['x7'] + 2*df['x10'],
    8 * np.exp(-0.5 * df['x5']) * np.sin(1.0 * df['x5'] * 50), 
    10 * np.sin(df['x1'] / 2.0) * np.cos(df['x8'] * 3.0) + 5 * np.sin(df['x1'] / 2.0) * np.cos(df['x2'] * 10.0), 
    10 * np.sin(df['x6']) + np.random.normal(0, 1.0, N)         
]

for i, y_val in enumerate(y_comps, 1): df[f'y{i}'] = y_val
df['yA'], df['yB'] = df['y1'] + df['y2'], df['y3'] + df['y4']
df['y'], df['Lag_y'] = df['yA'] + df['yB'], (df['yA'] + df['yB']).shift(1).bfill()

d_dict = {
    'train': df[dy <= 2022].copy(), 'dev': df[dy == 2023].copy(), 
    'val': df[dy == 2024].copy(), 'test': df[dy >= 2025].copy()
}

# ==============================================================================
# 2. Formulas Definition
# ==============================================================================
base_cats = "+ c(x3, n_cat=7, topo='nominal') + c(x4, n_cat=12, topo='fourier') + c(x7, n_cat=2, topo='ordinal') + c(x10, n_cat=31, topo='nominal')"
base_lin = "l(x1) + l(x2) + l(x5) + l(x6) + l(x8)"

neurons = 30
hidden_layers = 1

formulas = {
    "Linear": f"{base_lin} + l(Lag_y) {base_cats}",
    "Linear_PID": f"{base_lin} + pid(Lag_y, w=7, d_pen=10.0) {base_cats}",
    "Fourier": f"f(x8, m=10, s=1, cyclic=True) + l(x5) + l(x1) + l(x2) + f(x6, m=10, s=1, cyclic=False) + l(Lag_y) {base_cats}",
    "Spline": (
        f"s(x1, k=15, extrapolate='continue') + s(x2, k=15, extrapolate='continue') + "
        f"s(x5, k=10, extrapolate='continue') + s(x6, k=15, extrapolate='continue') + "
        f"s(x8, k=10, extrapolate='continue') + s(Lag_y, k=20, extrapolate='continue') {base_cats}"
    ),
    "Chebyshev": f"p(x1, deg=5) + p(x2, deg=5) + p(x5, deg=5) + p(x6, deg=5) + p(x8, deg=5) + p(Lag_y, deg=5) {base_cats}",
    "Wavelet": f"w(x6, n_scales=5, n_locations=20) + w(x1, n_scales=4, n_locations=10) + l(x2) + l(x5) + w(x8, n_scales=4, n_locations=10) + l(Lag_y) {base_cats}",
    "Tensor_interaction": f"te(l(x1), f(x8, m=10, s=1, cyclic=True)) + te(l(x1), l(x2)) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "Physics": (
        f"phys(x5, basis='spline', k=20, D1=0.5, D2=1.0) + "
        f"s(x1, k=10, extrapolate='continue') + s(x2, k=10, extrapolate='continue') + "
        f"s(x6, k=10, extrapolate='continue') + s(x8, k=10, extrapolate='continue') + "
        f"s(Lag_y, k=15, extrapolate='continue') {base_cats}"
    ),
    "RBF_interaction": f"rbf(x1, others='x8|x2', n_centers=50, gamma=0.1) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "Tree": f"t(x1, n_trees=10, max_depth=3, seed=42) + t(x8, n_trees=10, max_depth=3, seed=42) + t(x2, n_trees=10, max_depth=3, seed=42) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "LinearTree": f"lt(x1, max_leaves=8) + lt(x2, max_leaves=8) + lt(x5, max_leaves=8) + lt(x6, max_leaves=8) + lt(x8, max_leaves=8) + l(Lag_y) {base_cats}",
    "StaticTAM_neural_ReLU": f"n(x1, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + n(x2, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + n(x8, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "StaticTAM_neural_Cos": f"n(x1, n_neurons={neurons}, act='cos', n_hidden_layers={hidden_layers}) + n(x2, n_neurons={neurons}, act='cos', n_hidden_layers={hidden_layers}) + n(x8, n_neurons={neurons}, act='cos', n_hidden_layers={hidden_layers}) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "StaticTAM_neural_Tanh": f"n(x1, n_neurons={neurons}, act='tanh', n_hidden_layers={hidden_layers}) + n(x2, n_neurons={neurons}, act='tanh', n_hidden_layers={hidden_layers}) + n(x8, n_neurons={neurons}, act='tanh', n_hidden_layers={hidden_layers}) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "StaticTAM_neural_interaction": f"n(x1, others='x2|x8', n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
}

neural_formulas = {
    "NeuralTAM_ReLU": formulas["StaticTAM_neural_ReLU"],
    "NeuralTAM_Cos": formulas["StaticTAM_neural_Cos"],
    "NeuralTAM_Tanh": formulas["StaticTAM_neural_Tanh"],
    "NeuralTAM_interaction": formulas["StaticTAM_neural_interaction"]
}

# ==============================================================================
# 3. Base Models Fitting
# ==============================================================================
static_models, neural_models = {}, {}

for name, form in formulas.items():
    print(f"Fitting StaticTAM: {name}...")
    static_models[name] = ta.StaticTAM(formula=f"y ~ {form}", date_col="date").fit(d_dict['train'])

for name, form in neural_formulas.items():
    print(f"Fitting NeuralTAM: {name}...")
    neural_models[name] = ta.NeuralTAM(
        formula=f"y ~ {form}", date_col="date", epochs=100, lr=0.001, batch_size=128, weight_decay=1e-4, patience=30
    ).fit(d_dict['train'])

for name, m in {**static_models, **neural_models}.items():
    df[f'E_{name}'] = m.predict(df)["Estimatedy"].values

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ==============================================================================
# 4. HierarchicalTAM
# ==============================================================================
print("\n" + "="*50 + "\n FITTING HIERARCHICAL TAM \n" + "="*50)
df_long = pd.melt(
    df, id_vars=['date','x1','x2','x3','x4','x5','x6','x7','x8','x10','Lag_y'], 
    value_vars=['y','yA','yB'], var_name='Node', value_name='Target'
)

node_formulas = {
    "y": f"Target ~ n(x1, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + n(x2, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + n(x8, n_neurons={neurons}, act='relu', n_hidden_layers={hidden_layers}) + f(x6, m=10, s=1, cyclic=False) + l(x5) + l(Lag_y) {base_cats}",
    "yA": f"Target ~ l(x1) + l(x2) + l(x5) + l(Lag_y) + c(x3, n_cat=7, topo='nominal') + c(x4, n_cat=12, topo='fourier') + c(x7, n_cat=2, topo='ordinal') + c(x10, n_cat=31, topo='nominal')", 
    "yB": f"Target ~ l(x1) + l(x2) + f(x6, m=10, s=1, cyclic=False) + f(x8, m=10, s=1, cyclic=True) + l(Lag_y)"
}

m_hier = ta.HierarchicalTAM(
    structure={"y": ['yA','yB']}, formulas=node_formulas, node_col="Node", date_col="date"
).fit(df_long[df_long['date'].dt.year <= 2022])

reconciled_df = m_hier.predict(df_long)
df['E_Hier'] = reconciled_df[reconciled_df['Node'] == 'y']['EstimatedTarget'].values

# ==============================================================================
# 5. AdaptiveTAM & KalmanTAM
# ==============================================================================
print("\n" + "="*50 + "\n APPLYING ONLINE DRIFT CORRECTION \n" + "="*50)
for name, m in {**static_models, **neural_models}.items():
    print(f"Adapting {name}...")
    df[f'L_Res_{name}'] = (df['y'] - df[f'E_{name}']).shift(1).bfill()
    
    df[f'E_Adapt_{name}'] = ta.AdaptiveTAM(
        base_model=m, adaptive_formula=f"Residualy ~ l(L_Res_{name})", 
        update_interval_periods=1, training_window_periods=28, steps_per_period=1, horizon_steps=1
    ).predict_online(df)["AdaptedEstimatedy"].values
    
    df[f'E_Kalman_{name}'] = ta.KalmanTAM(
        base_model=m, kalman_formula=f"y ~ l(L_Res_{name})", 
        date_col="date", horizon_steps=1, process_noise_var=0.5, observation_noise_var=1.0, 
        use_decomposition=True, block_size=60
    ).predict_online(df)["KalmanAdapted_y"].values

df_y, m_h = df_long[df_long['Node'] == 'y'].copy(), m_hier.sub_models['y']
df_y['L_Res_Hier'] = (df_y['Target'] - df['E_Hier']).shift(1).bfill()

df['E_Adapt_Hier'] = ta.AdaptiveTAM(
    base_model=m_h, adaptive_formula="ResidualTarget ~ l(L_Res_Hier)", 
    update_interval_periods=1, training_window_periods=28, steps_per_period=1, horizon_steps=1
).predict_online(df_y)["AdaptedEstimatedTarget"].values

df['E_Kalman_Hier'] = ta.KalmanTAM(
    base_model=m_h, kalman_formula="Target ~ l(L_Res_Hier)", 
    date_col="date", horizon_steps=1, process_noise_var=0.5, observation_noise_var=1.0, 
    use_decomposition=True, block_size=60
).predict_online(df_y)["KalmanAdapted_Target"].values

# ==============================================================================
# 6. OperaTAM
# ==============================================================================
def run_opera(cols, output_col):
    formula = "y ~ " + "+".join(f"l({c})" for c in cols)
    opera = ta.OperaTAM(
        formula=formula, algorithm='MLpol', date_col='date', loss_type='square', horizon_steps=1
    )
    df[output_col] = opera.predict_online(df)['prediction_opera'].values

    with intercept_and_save_plot(CASE_NAME, f"opera_weights_{output_col}.png"):
        opera.plot_weights(df=df)

static_non_phys = [n for n in static_models if n != "Physics"]

run_opera([f"E_{n}" for n in static_non_phys], 'OE_StaticTAM')
run_opera([f"E_Adapt_{n}" for n in static_non_phys], 'OE_AdaptiveTAM')
run_opera([f"E_Kalman_{n}" for n in static_non_phys], 'OE_KalmanTAM')

run_opera(['E_Physics', 'E_Adapt_Physics', 'E_Kalman_Physics'], 'OE_PhysicsTAM')

run_opera([f"E_{n}" for n in neural_models] + [f"E_Adapt_{n}" for n in neural_models] + [f"E_Kalman_{n}" for n in neural_models], 'OE_NeuralTAM')
run_opera(['E_Hier', 'E_Adapt_Hier', 'E_Kalman_Hier'], 'OE_HierarchicalTAM')

level_1_operas = ['OE_StaticTAM', 'OE_AdaptiveTAM', 'OE_KalmanTAM', 'OE_PhysicsTAM', 'OE_NeuralTAM', 'OE_HierarchicalTAM']
run_opera(level_1_operas, 'OOE_GlobalTAM')

# ==============================================================================
# 7. Feature Effects Visualization
# ==============================================================================
print("\n" + "="*50 + "\n FEATURE EFFECT VISUALIZATION \n" + "="*50)
with intercept_and_save_plot(CASE_NAME, "effect_physics.png"):
    plot_effect_with_model_and_data(model=static_models["Physics"], data=d_dict['test'], effect='x5')

with intercept_and_save_plot(CASE_NAME, "effect_tensor.png"):
    plot_effect_with_model_and_data(model=static_models["Tensor_interaction"], data=d_dict['test'], effect='x1', color_by='x8')

# ==============================================================================
# 8. MLOps Evaluation
# ==============================================================================
trackers = {}

groups_to_eval = {
    "1. PhysicsTAM Family (Explicit PDEs)": ['E_Physics', 'E_Adapt_Physics', 'E_Kalman_Physics']+['OE_PhysicsTAM'],
    "2. StaticTAM Models (Non-Physics)": [f"E_{n}" for n in static_non_phys]+['OE_StaticTAM'],
    "3. NeuralTAM Models": [f"E_{n}" for n in neural_models.keys()] + [f"E_Adapt_{n}" for n in neural_models.keys()] + [f"E_Kalman_{n}" for n in neural_models.keys()]+['OE_NeuralTAM'],
    "4. HierarchicalTAM Base": ['E_Hier', 'E_Adapt_Hier', 'E_Kalman_Hier']+['OE_HierarchicalTAM'],
    "5. AdaptiveTAM Models (Non-Physics)": [f"E_Adapt_{n}" for n in static_non_phys]+['OE_AdaptiveTAM'],
    "6. KalmanTAM Models (Non-Physics)": [f"E_Kalman_{n}" for n in static_non_phys]+['OE_KalmanTAM'],
    "7. OperaTAM Meta-Learners": [op for op in level_1_operas] + ['OOE_GlobalTAM']
}

for group_name, cols in groups_to_eval.items():
    trackers[group_name] = {}
    print(f"\n--- {group_name} ---")
    for col in cols:
        tr = ta.BenchmarkTracker(col)
        tr.y_pred_full = df[col].values
        tr.slice_and_evaluate(d_dict, target_col='y')
        
        trackers[group_name][col] = tr

    show_trackers_plot(
        data_dict=d_dict, 
        tck=trackers[group_name], 
        case_name=CASE_NAME, 
        target_col='y', 
        date_col='date', 
        is_timeseries=True, 
        metric='RMSE',
        heatmap_col="x4", 
        title=f"TAM Evaluation for {group_name}",
        export_filename=f"dashboard_{group_name}.png",
        forecast_smoothing="ME"
    )

# ==============================================================================
# 9. SafetyTAM (ACI)
# ==============================================================================
print("\n" + "="*50 + "\n ADAPTIVE CONFORMAL INFERENCE (SAFETY TAM) \n" + "="*50)
safety = ta.SafetyTAM(alpha=0.1) 

val_idx = d_dict['val'].index
safety.calibrate(
    y_true=df.loc[val_idx, 'y'].values, 
    y_pred=df.loc[val_idx, 'OOE_GlobalTAM'].values
)

test_idx = d_dict['test'].index
df_safety = safety.predict_intervals(
    y_pred=df.loc[test_idx[-30:], 'OOE_GlobalTAM'].values, 
    y_true_online=df.loc[test_idx[-30:], 'y'].values, 
    method='aci', 
    gamma=0.05
)

coverage = df_safety['Covered'].mean() * 100
print(f"SafetyTAM ACI Coverage achieved: {coverage:.2f}% (Target: {100 * (1 - safety.alpha_target):.2f}%)")

print("Generating SafetyTAM Confidence Intervals Plot...")

with intercept_and_save_plot(CASE_NAME, "aci_safetytam.png"):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    x_dates = df.loc[test_idx[-30:], 'date']

    ax.plot(x_dates, df_safety['Actual'], label='Actual Values', color='#333333', linewidth=1.5, alpha=0.8)
    ax.plot(x_dates, df_safety['Predicted'], label='Predicted (OOE_GlobalTAM)', color='#C44E52', linewidth=2)

    ax.fill_between(
        x_dates, 
        df_safety['Lower'], 
        df_safety['Upper'], 
        color='#C44E52', 
        alpha=0.25, 
        label=f"ACI Confidence Interval ({100 * (1 - safety.alpha_target):.0f}% Target)"
    )

    ax.set_title("SafetyTAM: Adaptive Conformal Inference on Test Set", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Target Value", fontsize=12, fontweight='bold')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 10. AutoTAM
# ==============================================================================
print("\n" + "="*50 + "\n EVOLUTIONARY AUTO-ML (AUTOTAM) \n" + "="*50)
from tam.model.autotam.auto_tam import AutoTAM

cols_to_keep = ["date", "y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x10", "Lag_y"]

auto_model = AutoTAM(
    formula="y ~ AutoPipe(x1, x2, x3, x4, x5, x6, x7, x8, x10, Lag_y)", 
    n_experts=1, # put 5 to achieve better performance
    pop_size=2, # put 20 to achieve better performance
    use_opera=True, 
    eta=0.1, 
    export_dir=f"use_cases/temp/autotam_exports/{CASE_NAME}"
)

with intercept_and_save_text(CASE_NAME, "AutoTAM_training_log.txt"):
    auto_model.fit(
        df_fit=d_dict['train'][cols_to_keep], 
        df_dev=d_dict['dev'][cols_to_keep], 
        df_val=d_dict['val'][cols_to_keep], 
        date_col='date', 
        expansions={"prior": True, "autofit": True, "kalman": True, "adaptive": True, "grid": False}, 
        validation_strategy='expanding_window',
        refit_on_full_train=True
    )

print("\nGenerating AutoTAM Evolutionary Summary Graph...")
with intercept_and_save_plot(CASE_NAME, "AutoTAM_summary_report.png"):
    auto_model.summary_report()  

with intercept_and_save_text(CASE_NAME, "AutoTAM_performance_board.txt"):
    auto_model.print_performance_board() 

df_auto_preds = auto_model.predict(df[cols_to_keep], date_col='date')

auto_pred_col = [c for c in df_auto_preds.columns if c != 'date'][0]
df['E_AutoTAM_Champion'] = df_auto_preds[auto_pred_col].values

tr_auto = ta.BenchmarkTracker('E_AutoTAM_Champion')
tr_auto.y_pred_full = df['E_AutoTAM_Champion'].values
tr_auto.slice_and_evaluate(d_dict, target_col='y')

print(f"\n--- 8. AutoTAM Champion ---")
print(f"{'E_AutoTAM_Champion':30s} : {tr_auto.get_metric('test', 'RMSE'):.2f} (RMSE)")

trackers["7. OperaTAM Meta-Learners"]['E_AutoTAM_Champion'] = tr_auto

show_trackers_plot(
    data_dict=d_dict, 
    tck=trackers["7. OperaTAM Meta-Learners"], 
    case_name=CASE_NAME, 
    target_col='y', 
    date_col='date', 
    is_timeseries=True, 
    metric='RMSE', 
    heatmap_col="x4",
    title="TAM Evaluation: Complete Stack including AutoTAM",
    export_filename=f"dashboard_automl.png",
    forecast_smoothing="ME"
)