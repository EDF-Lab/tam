r"""
Implements the Aggregated TAM (OperaTAM) model.

This module provides GPU-accelerated Online Prediction by Expert Aggregation.
It implements both fully vectorized Exponentially Weighted Average (EWA) and 
temporally optimized Regret-based algorithms (MLpol) natively in PyTorch.

By utilizing 3D tensor batching (Groups, Time, Experts) combined with 
TorchScript compilation, this module ensures strict mathematical equivalence 
with the original Cesa-Bianchi frameworks while entirely avoiding the severe 
CPU-GPU kernel launch bottlenecks typical of sequential online learning loops.

Acknowledgments:
This implementation is heavily inspired by the original OPERA framework:
Gaillard, P., & Goude, Y. (2016). OPERA: a R package for online aggregation 
of experts.
"""

import re
import textwrap
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE, _balance_groups, _ensure_dummies, _cleanup_dummies


#: <torch_jit_mlpol>
@torch.jit.script
def _mlpol_loop_optimized_3d(
    experts_tensor: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compiled TorchScript loop for MLpol (Polynomial Minimax Strategy).
    Processes all groups and time steps simultaneously in a 3D tensor format 
    (Batch, Time, Experts) to saturate GPU cores and eliminate Python overhead.
    
    Args:
        experts_tensor: Predictions from experts, shape (B, T, K).
        y_true: Ground truth targets, shape (B, T, 1).
        loss_type: The loss function to use ('square' or 'absolute').
        
    Returns:
        Tuple containing the mixed predictions (B, T) and weights history (B, T, K).
    """
    B, T, K = experts_tensor.shape
    dtype = experts_tensor.dtype
    device = experts_tensor.device
    
    # Per-Group Scaling to prevent numerical instability
    scale_factor = torch.max(torch.abs(y_true), dim=1, keepdim=True)[0]
    # Prevent division by zero if a group's target is perfectly zero
    scale_factor = torch.where(
        scale_factor == 0.0, 
        torch.tensor(1.0, dtype=dtype, device=device), 
        scale_factor
    )
        
    X_scaled = experts_tensor / scale_factor
    Y_scaled = y_true / scale_factor
    
    # Pre-allocate outputs to avoid repetitive memory allocation
    weights_history = torch.zeros((B, T, K), dtype=dtype, device=device)
    predictions = torch.zeros((B, T), dtype=dtype, device=device)
    
    # Initialize state variables for all groups
    w = torch.ones((B, K), dtype=dtype, device=device) / float(K)
    cum_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    max_sq_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    learning_rates = torch.ones((B, K), dtype=dtype, device=device) / (2.0**20)

    # Compiled sequential time loop
    for t in range(T):
        xt_scaled = X_scaled[:, t, :]
        yt_scaled = Y_scaled[:, t, 0]
        
        weights_history[:, t, :] = w
        
        # Original scale for actual prediction tracking
        predictions[:, t] = torch.sum(w * experts_tensor[:, t, :], dim=1)
        
        # Scaled space for gradient regret calculation
        y_hat_scaled = torch.sum(w * xt_scaled, dim=1)
        
        if loss_type == 'square':
            r = 2.0 * (y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
        else:
            r = torch.sign(y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
            
        r_square = r ** 2
        cum_regrets += r
        
        # Adaptive learning rate adjustments
        max_r_square = torch.max(r_square, dim=1, keepdim=True)[0]
        max_sq_regret_diff = torch.clamp(max_r_square - max_sq_regrets, min=0.0)
        
        learning_rates = 1.0 / (1.0 / learning_rates + r_square + max_sq_regret_diff)
        max_sq_regrets += max_sq_regret_diff
        
        # Polynomial weight update: w is proportional to max(0, R)^2 (linearized)
        relu_regrets = torch.clamp(cum_regrets, min=0.0)
        w_next = learning_rates * relu_regrets
        w_sum = torch.sum(w_next, dim=1, keepdim=True)
        
        # Normalize weights, falling back to uniform if all regrets are <= 0
        mask = w_sum > 0.0
        w = torch.where(
            mask, 
            w_next / w_sum, 
            torch.ones((B, K), dtype=dtype, device=device) / float(K)
        )

    return predictions, weights_history
#: </torch_jit_mlpol>

#: <torch_jit_ewa>
@torch.jit.script
def _ewa_loop_optimized_3d(
    experts_tensor: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str,
    eta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compiled TorchScript loop for EWA (Exponentially Weighted Aggregation).
    Processes all groups simultaneously and prevents exponential underflow/overflow 
    using the stable log-sum-exp (safe softmax) numerical trick.
    """
    B, T, K = experts_tensor.shape
    dtype = experts_tensor.dtype
    device = experts_tensor.device
    
    # Pre-allocate outputs
    weights_history = torch.zeros((B, T, K), dtype=dtype, device=device)
    predictions = torch.zeros((B, T), dtype=dtype, device=device)
    
    # State variables
    w = torch.ones((B, K), dtype=dtype, device=device) / float(K)
    cum_losses = torch.zeros((B, K), dtype=dtype, device=device)

    for t in range(T):
        xt = experts_tensor[:, t, :]
        yt = y_true[:, t, 0]
        
        weights_history[:, t, :] = w
        predictions[:, t] = torch.sum(w * xt, dim=1)
        
        # Compute expert losses
        if loss_type == 'square':
            losses = (xt - yt.unsqueeze(1)) ** 2
        else:
            losses = torch.abs(xt - yt.unsqueeze(1))
            
        cum_losses += losses
        
        # Stable Exponential Update (Safe Softmax)
        scaled_losses = -eta * cum_losses
        max_val = torch.max(scaled_losses, dim=1, keepdim=True)[0]
        w_unnorm = torch.exp(scaled_losses - max_val)
        
        w_sum = torch.sum(w_unnorm, dim=1, keepdim=True)
        
        mask = w_sum > 0.0
        w = torch.where(
            mask,
            w_unnorm / w_sum,
            torch.ones((B, K), dtype=dtype, device=device) / float(K)
        )

    return predictions, weights_history
#: </torch_jit_ewa>

class OperaTAM:
    r"""
    Meta-model performing Online Aggregation of external experts natively on GPU.

    Supported Algorithms:
      - 'EWA': Exponentially Weighted Average.
      - 'MLPOL': Polynomial Minimax Strategy based on linearized regret.

    Attributes:
        formula (str): Aggregation topology (e.g., 'Target ~ l(Expert1) + l(Expert2)').
        algorithm (str): The aggregation strategy to use ('EWA' or 'MLPOL').
        eta (float): Learning rate parameter (primarily for EWA).
        loss_type (str): The loss function to evaluate experts ('square' or 'absolute').
        horizon_steps (int): Number of steps to shift the learned weights forward to enforce causal boundaries in multi-step forecasting.
        group_col (str, optional): Column name for independent group processing.
        date_col (str): Column name for time indexing to ensure chronological evaluation.
        target_col (str): Name of the target variable extracted from the formula.
        expert_cols (list): Extracted names of the expert prediction columns.
        weights_history_ (Dict[str, np.ndarray]): Dynamic weights per group.
    """

    def __init__(
        self, 
        formula: Optional[str] = None,
        target_col: Optional[str] = None,
        expert_cols: Optional[list] = None,  
        algorithm: str = 'MLPOL',
        eta: float = 1.0, 
        loss_type: str = 'square',
        horizon_steps: int = 1,
        group_col: Optional[str] = None,
        date_col: Optional[str] = None
    ):
        r"""
        Initializes the Native GPU Aggregator.

        Args:
            formula (str, optional): Formula defining the aggregation topology.
            target_col (str, optional): Target column name (used if formula is not provided).
            expert_cols (list, optional): List of expert column names (used if formula is not provided).
            algorithm (str): 'EWA' or 'MLPOL'.
            eta (float): Learning rate parameter (used by EWA).
            loss_type (str): Loss formulation ('square' or 'absolute').
            horizon_steps (int): Number of steps to shift the learned weights forward to enforce causal boundaries in multi-step forecasting. Default is 1.
            group_col (str, optional): If provided, runs aggregation independently per group.
            date_col (str, optional): Column name utilized for temporal sorting and padding.
        """

        if formula is None:
            if not target_col or not expert_cols:
                raise ValueError("You must provide either a 'formula' OR both 'target_col' and 'expert_cols'.")
            
            expert_terms = " + ".join([f"l({col})" for col in expert_cols])
            self.formula = f"{target_col} ~ {expert_terms}"
        else:
            self.formula = formula

        self.algorithm = algorithm.upper()
        self.eta = eta
        self.group_col = group_col or "__dummy_group__"
        self.date_col = date_col or "__dummy_date__"
        self.loss_type = loss_type
        self.horizon_steps = horizon_steps

        if self.algorithm not in ['EWA', 'MLPOL']:
            raise ValueError("Algorithm must be 'EWA' or 'MLPOL'.")
            
        if self.loss_type not in ['square', 'absolute']:
            raise ValueError("Unsupported loss_type. Use 'square' or 'absolute'.")
            
        if '~' not in self.formula:
            raise ValueError("Formula must contain '~' separating target and experts.")
            
        target_str, terms_str = self.formula.split('~')
        self.target_col = target_str.strip()
        
        self.expert_cols = re.findall(r'l\s*\(\s*([a-zA-Z0-9_]+)\s*\)', terms_str)
        
        if not self.expert_cols:
            raise ValueError("No valid experts found. Ensure syntax uses 'l(expert_name)'.")
            
        self.weights_history_: Dict[str, np.ndarray] = {}

    def predict_online(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Executes the online aggregation routing logic using 3D Tensor Batching.
        Processes the entire historical dataset for all groups simultaneously.
        Automatically balances the temporal groups to ensure contiguous 
        3D tensor stacking on the GPU, preventing dimension mismatches.

        Args:
            df (pd.DataFrame): Time-series dataframe containing targets and experts.

        Returns:
            pd.DataFrame: Augmented dataframe with aggregated predictions and weights.
        """
        # Create a temporary group identifier if none exists to maintain 3D logic
        temp_group_col = self.group_col if self.group_col else "__dummy_group__"
        df_to_process = _ensure_dummies(df, self.group_col, self.date_col)
        
        if not self.group_col:
            df_to_process[temp_group_col] = "all_data"
            
        # Padding Phase: Balance groups to ensure uniform tensor dimensions
        mask, balanced_df = _balance_groups(
            dataset=df_to_process, 
            group_col=temp_group_col, 
            date_col=self.date_col, 
            method="fill"
        )
        
        x_list = []
        y_list = []
        group_names = []
        
        # Preparation Phase: Extract sorted tensors using the balanced dataframe
        for group_name, df_group in balanced_df.groupby(temp_group_col):
            # Guarantee chronological order for sequential online evaluation
            df_g = df_group.sort_values(by=self.date_col) if self.date_col in df_group.columns else df_group.sort_index()
            group_names.append(group_name)
            
            x_list.append(torch.tensor(df_g[self.expert_cols].values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE))
            y_list.append(torch.tensor(df_g[self.target_col].values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE).unsqueeze(1))
            
        # Create contiguous 3D tensors: (Groups, Time, Experts)
        X_tensor_3d = torch.stack(x_list).contiguous()
        Y_tensor_3d = torch.stack(y_list).contiguous()
        
        # Computation Phase: Execute the entire historical simulation in one GPU pass
        if self.algorithm == 'MLPOL':
            preds_3d, weights_3d = _mlpol_loop_optimized_3d(X_tensor_3d, Y_tensor_3d, self.loss_type)
        else:
            preds_3d, weights_3d = _ewa_loop_optimized_3d(X_tensor_3d, Y_tensor_3d, self.loss_type, self.eta)
            
        # Reassembly Phase
        preds_np = preds_3d.cpu().numpy()
        weights_np = weights_3d.cpu().numpy()
        
        if self.horizon_steps > 1:
            shift = self.horizon_steps - 1
            shifted_weights = np.zeros_like(weights_np)
            shifted_weights[:, :shift, :] = 1.0 / len(self.expert_cols)
            shifted_weights[:, shift:, :] = weights_np[:, :-shift, :]
            experts_np = X_tensor_3d.cpu().numpy()
            preds_np = np.sum(shifted_weights * experts_np, axis=2)
            weights_np = shifted_weights

        # Initialize columns
        balanced_df['prediction_opera'] = 0.0
        for col in self.expert_cols:
            balanced_df[f'weight_{col}'] = 0.0
            
        # Accurately map tensor outputs back to their respective chronological DataFrame indices
        for i, g_name in enumerate(group_names):
            group_idx = balanced_df[balanced_df[temp_group_col] == g_name].index
            
            if self.date_col in balanced_df.columns:
                group_idx = balanced_df.loc[group_idx].sort_values(by=self.date_col).index
            else:
                group_idx = group_idx.sort_values()

            balanced_df.loc[group_idx, 'prediction_opera'] = preds_np[i]
            for j, col in enumerate(self.expert_cols):
                balanced_df.loc[group_idx, f'weight_{col}'] = weights_np[i, :, j]
                
            valid_mask = mask.loc[group_idx].values
            self.weights_history_[g_name] = weights_np[i][valid_mask]

        if not self.group_col:
            balanced_df = balanced_df.drop(columns=[temp_group_col])

        # Apply the boolean mask to strip away the padding injected by _balance_groups
        return _cleanup_dummies(balanced_df[mask], self.group_col, self.date_col)

    def plot_weights(
        self, 
        df: Optional[pd.DataFrame] = None, 
        group_name: Optional[str] = None
    ) -> None:
        r"""
        Plots the temporal evolution of dynamic expert weights using stacked area charts.
        Automatically handles datetime indices/columns and multi-group subplots.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.dates as mdates
        groups_to_plot = list(self.weights_history_.keys()) if group_name is None else [group_name]
        n_groups = len(groups_to_plot)
        
        if n_groups == 0:
            raise ValueError("No prediction history available to plot. Call predict_online() first.")

        modern_colors = [
            '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
            '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'
        ]
        n_experts = len(self.expert_cols)
        if n_experts <= len(modern_colors):
            plot_colors = modern_colors[:n_experts]
        else:
            cmap = plt.get_cmap('tab20')
            plot_colors = [cmap(i % 20) for i in range(n_experts)]

        fig, axes = plt.subplots(n_groups, 1, figsize=(12, max(4, 3 * n_groups)), squeeze=False)
        
        wrapped_formula = textwrap.fill(self.formula, width=80)
        group_text = f" - Group: {self.group_col} = {group_name}" if (self.group_col != "__dummy_group__" and group_name) else ""

        # Use native suptitle instead of absolute fig.text to prevent overlapping
        fig.suptitle(f"Dynamic Expert Weights Allocation ({self.algorithm}){group_text}\n\n"
                     f"{wrapped_formula}", 
                     fontsize=12, fontweight='bold', color='#333333')

        is_datetime_global = False

        for i, g_name in enumerate(groups_to_plot):
            if g_name not in self.weights_history_:
                continue
                
            ax = axes[i, 0]
            weights = self.weights_history_[g_name]
            
            if df is not None:
                if self.group_col and self.group_col != "__dummy_group__" and self.group_col in df.columns:
                    df_g = df[df[self.group_col] == g_name]
                else:
                    df_g = df
                
                if 'date' in df_g.columns:
                    x_axis = pd.to_datetime(df_g['date']).values
                    is_datetime = True
                elif isinstance(df_g.index, pd.DatetimeIndex):
                    x_axis = df_g.index
                    is_datetime = True
                else:
                    x_axis = np.arange(weights.shape[0])
                    is_datetime = False
            else:
                x_axis = np.arange(weights.shape[0])
                is_datetime = False

            is_datetime_global = is_datetime

            ax.stackplot(
                x_axis,
                weights.T,
                labels=self.expert_cols,
                colors=plot_colors,
                alpha=0.9,
                edgecolor='white',
                linewidth=1.2
            )
                        
            ax.set_xlim(x_axis[0], x_axis[-1])
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            
            if is_datetime:
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, axis='y', color='white', linewidth=1.5, alpha=1.0, zorder=0)
            ax.grid(False, axis='x')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#cccccc')
            
            ax.tick_params(axis='both', which='major', colors='#555555', length=0, pad=8)
            
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[::-1], 
                    labels[::-1], 
                    loc='center left', 
                    bbox_to_anchor=(1.02, 0.5),
                    ncol=1,
                    fontsize=10,
                    frameon=False,
                    title="Experts",
                    title_fontproperties={'weight':'bold', 'size': 11}
                )

            if i == n_groups - 1:
                ax.set_xlabel("Time" if is_datetime_global else "Steps", 
                              fontsize=11, fontweight='bold', color='#555555')
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) 
        
        plt.show()