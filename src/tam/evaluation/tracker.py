# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Tracking and Telemetry Module for AutoTAM.

Provides a universal benchmark tracking object to log predictions, 
compute standard regression metrics, and execute MLOps diagnostics 
(like temporal degradation) across multiple dataset splits.
"""

#: <tracker_imports>
import numpy as np
import pandas as pd
from .metrics import calculate_regression_metrics
from .performance_analyzer import analyze_residuals, detect_temporal_degradation
#: </tracker_imports>

#: <tracker_class>
class BenchmarkTracker:
    """Universal TAM Tracker for experiments."""
    
#: <tracker_init>
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {}      
        self.predictions = {}  
        self.diagnostics = {}
        self.y_pred_full = None 
        self.time_fit = 0.0
        self.time_predict = 0.0
#: </tracker_init>

#: <tracker_slice_and_evaluate>
    def slice_and_evaluate(self, data_dict: dict, target_col: str = 'value'):
        """
        Slices continuous predictions into respective folds and computes 
        standard metrics and MLOps diagnostics.
        """
        if self.y_pred_full is None:
            print(f"Warning: '{self.model_name}' - 'y_pred_full' not found. Cannot evaluate.")
            return

        current_idx = 0
        for split_name, df in data_dict.items():
            length = len(df)
            y_true = df[target_col].values
            y_pred = self.y_pred_full[current_idx : current_idx + length]
            
            self.predictions[split_name] = y_pred
            self.metrics[split_name] = calculate_regression_metrics(y_true, y_pred)
            self.diagnostics[split_name] = analyze_residuals(y_true, y_pred)
            
            if split_name == 'test':
                drift = detect_temporal_degradation(y_true, y_pred, metric='RMSE')
                self.diagnostics['test']['RMSE_Drift_H2_vs_H1 (%)'] = drift
                
            current_idx += length
            
        components = ['fit', 'dev', 'val']
        if all(k in data_dict for k in components) and 'train' not in data_dict:
            yt_train = np.concatenate([data_dict[k][target_col].values for k in components])
            yp_train = np.concatenate([self.predictions[k] for k in components])
            self.predictions['train'] = yp_train
            self.metrics['train'] = calculate_regression_metrics(yt_train, yp_train)
#: </tracker_slice_and_evaluate>

#: <tracker_get_metric>
    def get_metric(self, split: str, metric: str) -> float:
        """Safely retrieves a specific metric from a specific data split."""
        return self.metrics.get(split, {}).get(metric, np.nan)
#: </tracker_get_metric>

#: <tracker_report_grouped>
    def report_grouped_metrics(self, data_dict: dict, group_col: str, target_col: str = 'value', metrics: list = ['RMSE', 'SMAPE'], splits: list = ['test', 'val']) -> pd.DataFrame:
        """
        Generates a summary report of model performance segmented by a specified categorical group.
        """
        report_records = []
        for split in splits:
            if split not in data_dict or split not in self.predictions: 
                continue
                
            df_split = data_dict[split].copy()
            if group_col not in df_split.columns: 
                continue
                
            df_split['__y_pred'] = self.predictions[split]
            df_split['__y_true'] = df_split[target_col]
            df_clean = df_split.dropna(subset=['__y_true', '__y_pred']).copy()
            
            for group_val, group_data in df_clean.groupby(group_col):
                yt, yp = group_data['__y_true'].values, group_data['__y_pred'].values
                g_metrics = calculate_regression_metrics(yt, yp)
                
                record = {'Split': split.upper(), group_col.capitalize(): group_val, 'Samples': len(yt)}
                for m in metrics: 
                    record[m] = g_metrics.get(m, np.nan)
                report_records.append(record)
                
        if not report_records: 
            return pd.DataFrame()
            
        df_report = pd.DataFrame(report_records)
        pivot_report = df_report.pivot(index=group_col.capitalize(), columns='Split', values=metrics)
        return pivot_report.swaplevel(0, 1, axis=1).sort_index(axis=1)
#: </tracker_report_grouped>
#: </tracker_class>