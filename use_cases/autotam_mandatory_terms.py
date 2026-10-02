"""
AutoTAM with mandatory terms and mandatory variables.

Same synthetic data and split as cheatsheet.py, same small AutoTAM settings as its section 10. Three models:
  - AutoTAM_Free          : y ~ AutoPipe(...)                      (reference, the cheatsheet champion)
  - AutoTAM_MandatoryTerm : y ~ s(x1, k=10) + AutoPipe(...)        (the term is kept by every static model)
  - AutoTAM_MandatoryVar  : free formula, mandatory_variables=["x10"] (x10 appears in every static model)

The script fails if a static model of the search lacks the mandatory term or variable, so a regression of the feature is caught
by the regression check (pre_push.py, own use cases), not only by the unit tests.
"""

import random

import numpy as np
import pandas as pd
import torch
import tam as ta

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.parser import parse_formula_to_terms, term_subsumes

CASE_NAME = "autotam_mandatory_terms"
SEED = 42


def seed_everything(seed=42):
    """Same seeding as utils_cases.seed_everything (kept here so the script runs alone, also as a frozen copy)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. Data (same as cheatsheet.py)
# ==============================================================================
seed_everything(seed=SEED)

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

for i, y_val in enumerate(y_comps, 1):
    df[f'y{i}'] = y_val
df['yA'], df['yB'] = df['y1'] + df['y2'], df['y3'] + df['y4']
df['y'], df['Lag_y'] = df['yA'] + df['yB'], (df['yA'] + df['yB']).shift(1).bfill()

d_dict = {
    'train': df[dy <= 2022].copy(), 'dev': df[dy == 2023].copy(),
    'val': df[dy == 2024].copy(), 'test': df[dy >= 2025].copy()
}

cols_to_keep = ["date", "y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x10", "Lag_y"]
PIPE = "AutoPipe(x1, x2, x3, x4, x5, x6, x7, x8, x10, Lag_y)"
MANDATORY_TERM = "s(x1, k=10)"
MANDATORY_VAR = "x10"

# ==============================================================================
# 2. The three models
# ==============================================================================
models = {
    "AutoTAM_Free": dict(formula=f"y ~ {PIPE}"),
    "AutoTAM_MandatoryTerm": dict(formula=f"y ~ {MANDATORY_TERM} + {PIPE}"),
    "AutoTAM_MandatoryVar": dict(formula=f"y ~ {PIPE}", mandatory_variables=[MANDATORY_VAR]),
}


def static_formulas(model):
    """Formulas of the static (StaticTAM) experts evaluated during the search."""
    return [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("StaticTAM")]


rmse = {}
for name, kwargs in models.items():
    print("\n" + "=" * 50 + f"\n {name} \n" + "=" * 50)
    seed_everything(seed=SEED)
    auto_model = AutoTAM(
        n_experts=1,
        pop_size=2,
        use_opera=True,
        eta=0.1,
        export_dir=f"use_cases/temp/autotam_exports/{CASE_NAME}/{name}",
        **kwargs,
    )
    auto_model.fit(
        df_fit=d_dict['train'][cols_to_keep],
        df_dev=d_dict['dev'][cols_to_keep],
        df_val=d_dict['val'][cols_to_keep],
        date_col='date',
        expansions={"prior": True, "autofit": True, "kalman": True, "adaptive": True, "grid": False},
        validation_strategy='expanding_window',
        refit_on_full_train=True,
    )

    # The mandatory constraints hold in every static model the search evaluated.
    formulas = static_formulas(auto_model)
    assert formulas, f"{name}: no static model in the search log"
    for f in formulas:
        assert f.count("(") == f.count(")"), f"{name}: unbalanced parentheses in {f}"
        _, terms = parse_formula_to_terms(f)
        if name == "AutoTAM_MandatoryTerm":
            assert any(term_subsumes(tm, MANDATORY_TERM) for tm in terms), f"{name}: mandatory term missing in {f}"
        if name == "AutoTAM_MandatoryVar":
            assert any(tm.get("feature") == MANDATORY_VAR for tm in terms), f"{name}: mandatory variable missing in {f}"

    df_preds = auto_model.predict(df[cols_to_keep], date_col='date')
    pred_col = [c for c in df_preds.columns if c != 'date'][0]
    df[name] = df_preds[pred_col].values

    tr = ta.BenchmarkTracker(name)
    tr.y_pred_full = df[name].values
    tr.slice_and_evaluate(d_dict, target_col='y')
    rmse[name] = tr.get_metric('test', 'RMSE')
    print(f"\n{name:30s} : {rmse[name]:.5f} (test RMSE), {len(formulas)} static models checked")

print("\n" + "=" * 50 + "\n Summary (test RMSE) \n" + "=" * 50)
for name, value in rmse.items():
    print(f"{name:30s} : {value:.5f}")
