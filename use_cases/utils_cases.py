#: <utils_imports>
import sys
import os
import shutil
import requests
import logging
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from IPython.display import display
from importlib import resources

from tam.evaluation.tracker import BenchmarkTracker
from tam.evaluation.eval_plotting import plot_benchmark_dashboard, generate_summary_table
#: </utils_imports>

def seed_everything(seed=42):
    """Locks all stochastic operations for NeuralTAM reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you are using a GPU, enforce deterministic CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Forces cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

#: <offline_utils>

def setup_tabicl_environment(base_dir="use_cases/temp/temp_input"):
    """
    Intelligently handles TabICL model loading.
    Returns the file path if found locally, True if network is open, or False if blocked.
    """
    target_file = "tabicl-regressor-v2-20260212.ckpt"
    input_dir = Path(base_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = input_dir / target_file
    
    # 1. If the file exists, return the exact path to pass to the model natively
    if ckpt_path.exists():
        print(f"\n✅ Found local TabICL checkpoint at: {ckpt_path}")
        return str(ckpt_path)

    # 2. If missing, check network connection
    try:
        requests.head("https://huggingface.co", timeout=3)
        return True
    except requests.RequestException:
        # 3. If blocked, fail gracefully
        print(f"\n⚠️ Skipping TabICL (Network Blocked). To include it, please download the model from https://huggingface.co/jingang/TabICL/resolve/main/{target_file} and save it to {input_dir}\n")
        return False
#: </offline_utils>

#: <io_utils>
def save_data_to_disk(df: pd.DataFrame, case_name: str, file_name: str, base_dir: str = "use_cases/temp/temp_input"):
    """
    Saves a DataFrame to disk using cross-platform absolute paths.
    Defaults to CSV to guarantee environment compatibility 
    across different Pandas versions.
    """
    save_path = Path(base_dir).resolve() / case_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    full_file_path = save_path / file_name
    
    if file_name.endswith('.csv'):
        df.to_csv(full_file_path, index=False)
    else:
        df.to_pickle(full_file_path, protocol=4)
        
    print(f"Data saved for {case_name} at: {full_file_path}")

def load_data_from_disk(case_name: str, file_name: str, base_dir: str = "use_cases/temp/temp_input"):
    """Loads a DataFrame from disk if the path exists."""

    full_file_path = resources.files('tam.data').joinpath(file_name)
    if not full_file_path.exists():
        full_file_path = Path(base_dir).resolve() / case_name / file_name
    
    if full_file_path.exists():
        print(f"Loading data for {case_name} from: {full_file_path}")
        if file_name.endswith('.csv'):
            df = pd.read_csv(full_file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return pd.read_pickle(full_file_path)
    return None

@contextmanager
def intercept_and_save_plot(case_name: str, file_name: str, base_dir: str = "use_cases/temp/graph"):
    """
    Context manager to safely intercept plotting from internal framework components.
    Captures the active figure to disk and suppresses GUI popups.
    """
    original_show = plt.show
    
    def mock_show(*args, **kwargs):
        pass

    plt.show = mock_show
    try:
        yield
    finally:
        plt.show = original_show
        
        fig = plt.gcf()
        if fig.get_axes():
            save_path = Path(base_dir).resolve() / case_name
            save_path.mkdir(parents=True, exist_ok=True)
            full_file_path = save_path / file_name
            
            fig.savefig(full_file_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Graph intercepted and exported for {case_name} at: {file_name}")
            
        plt.close('all')

@contextmanager
def intercept_and_save_text(case_name: str, file_name: str, base_dir: str = "use_cases/temp/graph"):
    """
    Context manager to redirect all standard output, standard error, 
    and Python logging to a text file. Suppresses terminal output 
    and streams it directly to disk.
    """
    save_path = Path(base_dir).resolve() / case_name
    save_path.mkdir(parents=True, exist_ok=True)
    full_file_path = save_path / file_name

    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]

    with open(full_file_path, 'w', encoding='utf-8') as f:
        file_handler = logging.StreamHandler(f)
        root_logger.handlers = [file_handler]
        
        with redirect_stdout(f), redirect_stderr(f):
            try:
                yield
            finally:
                root_logger.handlers = original_handlers
                
    print(f"Text report intercepted and exported for {case_name} at: {full_file_path}")
#: </io_utils>

#: <evaluation_utils>
def build_honest_predictions(model_key, oof_preds, warmed_preds, test_index):
    """
    Constructs the full array of predictions combining OOF (stage 1) 
    and test inference (stage 2) for cross-sectional tracking.
    Uses .iloc to avoid KeyError if the prediction dataframe resets its index.
    """
    stage1_preds = oof_preds[model_key]
    stage1_vals = stage1_preds.values if hasattr(stage1_preds, 'values') else stage1_preds
    stage2_vals = warmed_preds[model_key].iloc[-len(test_index):].values
    
    return np.concatenate([stage1_vals, stage2_vals])

def track_and_evaluate(model_name, full_predictions, time_fit, time_predict, data_dict, trackers_dict, target_col='value', metric='MAPE', verbose=True):
    """Generic wrapper to register predictions and evaluate models dynamically."""
    tracker = BenchmarkTracker(model_name)
    tracker.time_fit = time_fit
    tracker.time_predict = time_predict
    tracker.y_pred_full = full_predictions
    
    tracker.slice_and_evaluate(data_dict, target_col=target_col)
    trackers_dict[model_name] = tracker
    
    if verbose:
        val_score = tracker.get_metric('val', metric)
        test_score = tracker.get_metric('test', metric)
        print(f"DONE: {model_name:<20} | Val {metric}: {val_score:.2f} | Test {metric}: {test_score:.2f}")
        
    return tracker

def show_trackers_plot(data_dict, tck, case_name, models_to_audit=None, metric="MAPE", target_col="value", date_col="date", is_timeseries=True, group_col="year", title="Benchmark Evaluation", heatmap_col="month", export_filename="dashboard.png",forecast_smoothing=None):
    """Generates the dashboard and performance tables using configurable targets and metrics."""
    if models_to_audit is None:
        models_to_audit = []
        
    print("\n" + "="*80 + "\n VISUAL BENCHMARK DASHBOARD\n" + "="*80)
    with intercept_and_save_plot(case_name, export_filename):
        plot_benchmark_dashboard(
            data_dict=data_dict, 
            trackers_dict=tck, 
            target_col=target_col, 
            date_col=date_col, 
            is_timeseries=is_timeseries, 
            primary_metric=metric,
            title=title,
            heatmap_col=heatmap_col,
            forecast_smoothing=forecast_smoothing
        )

    print("\n" + "="*80 + f"\n GLOBAL SPLIT PERFORMANCE ({metric})\n" + "="*80)
    summary_df = generate_summary_table(tck, primary_metric=metric, splits_to_show=['fit', 'dev', 'val', 'train', 'test'])
    display(summary_df)

    print("\n" + "="*80 + "\n RESIDUAL & DRIFT DIAGNOSTICS (OUT-OF-SAMPLE TEST SET)\n" + "="*80)
    diag_records = [{'Model': name, **t.diagnostics['test']} for name, t in tck.items() if 'test' in t.diagnostics and t.diagnostics['test']]
    if diag_records:
        display(pd.DataFrame(diag_records).set_index('Model').sort_values(by='Mean Error (Bias)', key=abs).round(4))

    if models_to_audit:
        print("\n" + "="*80 + "\n GRANULAR VULNERABILITY ANALYSIS\n" + "="*80)
        for model_name in models_to_audit:
            if model_name in tck:
                grouped_report = tck[model_name].report_grouped_metrics(
                    data_dict=data_dict, group_col=group_col, target_col=target_col, metrics=[metric], splits=['val', 'test']
                )
                if not grouped_report.empty:
                    print(f"\n--- Model: {model_name} ---")
                    display(grouped_report.round(2))
#: </evaluation_utils>