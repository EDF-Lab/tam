import pandas as pd, numpy as np, tam as ta
from importlib import resources

df = pd.read_csv(resources.files('tam.data').joinpath('airpassengers.csv')).dropna().copy()
df['date'] = pd.to_datetime(df['date'])
d_dict = {'train': df.iloc[:-24], 'test': df.iloc[-24:]}

f1, f2 = "log_passengers ~ c(month, topo='fourier', ap=-8.0) + l(lag_log_passengers, ap=-30.0)", "log_passengers ~ n(month) + l(lag_log_passengers, ap=-30.0)"
df['E1'] = ta.StaticTAM(formula=f1, date_col="date").fit(d_dict['train']).predict(df)["Estimatedlog_passengers"].values
df['E2'] = ta.StaticTAM(formula=f2, date_col="date").fit(d_dict['train']).predict(df)["Estimatedlog_passengers"].values

opera = ta.OperaTAM("log_passengers ~ l(E1) + l(E2)", algorithm='MLPOL', date_col='date', horizon_steps=12) # or expert_cols=["E1", "E2"] / target_col="log_passengers"
df['OPERA'] = opera.predict_online(df)['prediction_opera'].values

for name, col in [("Expert 1", "E1"), ("Expert 2", "E2"), ("OPERA", "OPERA")]:
    tr = ta.BenchmarkTracker(name)
    tr.y_pred_full = np.exp(df[col].values)
    tr.slice_and_evaluate(d_dict, target_col='value')
    print(f"[{name}] Test MAPE: {tr.get_metric('test', 'MAPE'):.2f}%")

opera.plot_weights(df=df)