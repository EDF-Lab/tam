"""
Implements NeuralTAM: A Deep-GAM Hybrid with Group-wise Orthogonal Backfitting.

Provides a hybrid architecture that fits structured parametric effects using a 
closed-form solver, and fits deep neural effects iteratively via Coordinate Descent,
maintaining independent neural networks for distinct data groups.
"""

import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from tam.model.additive import StaticTAM
from tam.model.spectrum import NeuralEffect
from tam.common.utils import TORCH_DEVICE, _ensure_dummies, _cleanup_dummies

#: <deep_neural_component>
class DeepNeuralComponent(nn.Module):
    """
    Standard PyTorch MLP architecture dynamically built from formula parameters.
    
    Attributes:
        network (nn.Sequential): The constructed sequential neural network layers.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        n_neurons: int, 
        n_hidden_layers: int, 
        activation_name: str
    ):
        """
        Initializes the neural network component.

        Args:
            input_dim (int): Dimensionality of the input features.
            n_neurons (int): Number of neurons per hidden layer.
            n_hidden_layers (int): Total number of hidden layers.
            activation_name (str): Activation identifier ('relu', 'tanh', 'cos').
        """
        super().__init__()
        layers = []
        
        if activation_name == 'relu':
            act = nn.ReLU()
        elif activation_name == 'tanh':
            act = nn.Tanh()
        elif activation_name == 'cos':
            class CosineActivation(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor: 
                    return torch.cos(x)
            act = CosineActivation()
        else:
            act = nn.ReLU()

        curr_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(curr_dim, n_neurons))
            layers.append(act)
            curr_dim = n_neurons
        
        final_layer = nn.Linear(curr_dim, 1)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1).
        """
        return self.network(x)
#: </deep_neural_component>

#: <neural_tam_init>
class NeuralTAM:
    """
    Hybrid Deep-GAM Model managing two-stage structured and neural training.

    Maintains independent MLP ensembles per data group to ensure mathematical 
    consistency with the base StaticTAM solver.

    Attributes:
        formula_ (str): R-style model formula.
        group_col_ (str): Column name for data segmentation.
        date_col_ (str): Column name for chronological tracking.
        epochs (int): Max training epochs for backfitting MLPs.
        lr (float): Initial learning rate for Adam optimizer.
        batch_size (int): Training batch size.
        val_split (float): Fraction of data for internal early stopping validation.
        shuffle_split (bool): If True, shuffles data before validation splitting.
        patience (int): Epochs to wait for validation loss improvement.
        weight_decay (float): L2 regularization penalty for network weights.
        backfit_cycles (int): Number of Coordinate Descent cycles per group.
        gam_shrinkage (float): Relaxation parameter (0 to 1) for residual updates.
        base_additive_model (StaticTAM): The underlying linear GAM solver.
        mlps_ (Dict[str, Dict[str, nn.Module]]): Trained networks mapped by group and feature.
        x_scalers_ (Dict[str, Dict[str, tuple]]): Z-score scalers for input features.
        y_scalers_ (Dict[str, Dict[str, tuple]]): Z-score scalers for target residuals.
        coefficients_ (torch.Tensor): Fitted coefficients from the base model.
        target_col_ (str): Name of the target variable.
    """
    
    def __init__(
        self, 
        formula: str, 
        group_col: Optional[str] = None, 
        date_col: Optional[str] = None, 
        epochs: int = 500, 
        lr: float = 0.01, 
        batch_size: int = 1024, 
        val_split: float = 0.2, 
        shuffle_split: bool = False,
        patience: int = 25,
        weight_decay: float = 1e-4,
        backfit_cycles: int = 3,
        gam_shrinkage: float = 1.0,
        **kwargs: Any
    ):
        """Initializes the NeuralTAM model configuration."""
        self.formula_ = formula
        self.group_col_ = group_col or "__dummy_group__"
        self.date_col_ = date_col or "__dummy_date__"
        
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle_split = shuffle_split
        self.patience = patience
        self.weight_decay = weight_decay
        self.backfit_cycles = backfit_cycles
        self.gam_shrinkage = gam_shrinkage
        
        self.base_additive_model = StaticTAM(formula, self.group_col_, self.date_col_, **kwargs)
        
        self.mlps_: Dict[str, Dict[str, nn.Module]] = {} 
        self.x_scalers_: Dict[str, Dict[str, tuple]] = {}
        self.y_scalers_: Dict[str, Dict[str, tuple]] = {}
        
        self.coefficients_ = None 
        self.target_col_ = None
    @property
    def effects_list_(self) -> list:
        """Passthrough to the base additive model's effects."""
        return getattr(self.base_additive_model, 'effects_list_', [])

    @property
    def unique_groups_(self) -> list:
        return getattr(self.base_additive_model, 'unique_groups_', [])

    @property
    def features_config_(self) -> dict:
        return getattr(self.base_additive_model, 'features_config_', {})
#: </neural_tam_init>

#: <neural_tam_fit>
    def fit(self, data_train: pd.DataFrame, data_val: Optional[pd.DataFrame] = None) -> 'NeuralTAM':
        """
        Fits the base additive model and backfits neural components.

        Args:
            data_train (pd.DataFrame): Primary training dataset.
            data_val (Optional[pd.DataFrame]): Explicit validation dataset for early stopping.

        Returns:
            NeuralTAM: The fitted model instance.
        """
        self.base_additive_model.fit(data_train)
        self.coefficients_ = self.base_additive_model.coefficients_
        self.target_col_ = self.base_additive_model.target_col_
        
        self._backfit_mlps(data_train, data_val)
        return self
#: </neural_tam_fit>

    def grid_search_fit(
        self, 
        data_train: pd.DataFrame, 
        data_val: pd.DataFrame, 
        grid_search_config: dict
    ) -> 'NeuralTAM':
        """
        Performs coordinate descent grid search on the base model before backfitting.

        Args:
            data_train (pd.DataFrame): Training data.
            data_val (pd.DataFrame): Validation data used for grid scoring.
            grid_search_config (dict): Token mapping for hyperparameter search axes.

        Returns:
            NeuralTAM: The fitted model instance.
        """
        self.base_additive_model.grid_search_fit(data_train, data_val, grid_search_config)
        self.coefficients_ = self.base_additive_model.coefficients_
        self.target_col_ = self.base_additive_model.target_col_
        
        self._backfit_mlps(data_train, data_val)
        return self

#: <orthogonal_backfitting>
    def _backfit_mlps(self, data_train: pd.DataFrame, data_val: Optional[pd.DataFrame] = None) -> None:
        """
        Executes cyclic orthogonal backfitting (Coordinate Descent) independently per group.

        Args:
            data_train (pd.DataFrame): Training dataset containing full features and target.
            data_val (Optional[pd.DataFrame]): Validation dataset.
        """
        # Ensure dummy columns exist for local grouping logic
        data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
        if data_val is not None:
            data_val = _ensure_dummies(data_val, self.group_col_, self.date_col_)

        decomposed_train = self.base_additive_model.decompose_prediction(data_train)
        effect_cols = [c for c in decomposed_train.columns if c.startswith('effect_')]
        est_col = f'Estimated{self.target_col_}'
        decomposed_train[est_col] = decomposed_train[effect_cols].sum(axis=1)
        
        neural_effects = [e for e in self.base_additive_model.effects_list_ if isinstance(e, NeuralEffect)]
        if not neural_effects: 
            return
            
        global_res_train_full = data_train[self.target_col_].values - decomposed_train[est_col].values
        
        global_res_val_full = None
        if data_val is not None:
            decomposed_val = self.base_additive_model.decompose_prediction(data_val)
            decomposed_val[est_col] = decomposed_val[effect_cols].sum(axis=1)
            global_res_val_full = data_val[self.target_col_].values - decomposed_val[est_col].values

        unique_groups = self.base_additive_model.unique_groups_

        for group_name in unique_groups:
            self.mlps_[group_name] = {}
            self.x_scalers_[group_name] = {}
            self.y_scalers_[group_name] = {}
            
            mask_train = (data_train[self.group_col_] == group_name).values
            group_data_train = data_train.iloc[mask_train]
            group_res_train = global_res_train_full[mask_train]
            
            if len(group_data_train) == 0:
                continue
                
            group_data_val = None
            group_res_val = None
            mask_val = None
            if data_val is not None:
                mask_val = (data_val[self.group_col_] == group_name).values
                group_data_val = data_val.iloc[mask_val]
                group_res_val = global_res_val_full[mask_val]
                if len(group_data_val) == 0:
                    group_data_val = None

            current_preds_train = {}
            for ne in neural_effects:
                col_name = f'effect_{ne.feature_name}'
                if col_name in decomposed_train.columns:
                    current_preds_train[ne.feature_name] = decomposed_train.loc[mask_train, col_name].values.copy()
                else:
                    current_preds_train[ne.feature_name] = np.zeros_like(group_res_train)

            current_preds_val = {}
            if group_data_val is not None:
                for ne in neural_effects:
                    col_name = f'effect_{ne.feature_name}'
                    if col_name in decomposed_val.columns:
                        current_preds_val[ne.feature_name] = decomposed_val.loc[mask_val, col_name].values.copy()
                    else:
                        current_preds_val[ne.feature_name] = np.zeros_like(group_res_val)

            X_train_dict, X_val_dict = {}, {}
            for ne in neural_effects:
                X_tensor = torch.tensor(group_data_train[ne.input_features].values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
                X_mean = X_tensor.mean(dim=0, keepdim=True)
                X_std = X_tensor.std(dim=0, keepdim=True)
                X_std[X_std < 1e-6] = 1.0 
                
                self.x_scalers_[group_name][ne.feature_name] = (X_mean, X_std)
                X_train_dict[ne.feature_name] = (X_tensor - X_mean) / X_std
                
                if group_data_val is not None:
                    X_val_raw = torch.tensor(group_data_val[ne.input_features].values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
                    X_val_dict[ne.feature_name] = (X_val_raw - X_mean) / X_std

            for _ in range(self.backfit_cycles):
                for ne in neural_effects:
                    f_name = ne.feature_name
                    
                    partial_res_train = group_res_train + current_preds_train[f_name]
                    Y_tensor_raw = torch.tensor(partial_res_train, dtype=torch.get_default_dtype(), device=TORCH_DEVICE).unsqueeze(1)
                    
                    Y_mean = Y_tensor_raw.mean()
                    Y_std = Y_tensor_raw.std()
                    if Y_std < 1e-6: 
                        Y_std = torch.tensor(1.0, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
                    
                    self.y_scalers_[group_name][f_name] = (Y_mean.item(), Y_std.item())
                    Y_scaled_train = (Y_tensor_raw - Y_mean) / Y_std
                    X_scaled_train = X_train_dict[f_name]
                    
                    if group_data_val is not None:
                        partial_res_val = group_res_val + current_preds_val[f_name]
                        Y_val_raw = torch.tensor(partial_res_val, dtype=torch.get_default_dtype(), device=TORCH_DEVICE).unsqueeze(1)
                        Y_scaled_val = (Y_val_raw - Y_mean) / Y_std
                        X_scaled_val = X_val_dict[f_name]
                    else:
                        num_samples = X_scaled_train.shape[0]
                        val_size = max(1, int(num_samples * self.val_split))
                        train_size = num_samples - val_size
                        
                        if self.shuffle_split: 
                            indices = torch.randperm(num_samples, device=TORCH_DEVICE)
                        else: 
                            indices = torch.arange(num_samples, device=TORCH_DEVICE)
                            
                        train_idx, val_idx = indices[:train_size], indices[train_size:]
                        X_scaled_val, Y_scaled_val = X_scaled_train[val_idx], Y_scaled_train[val_idx]
                        X_scaled_train, Y_scaled_train = X_scaled_train[train_idx], Y_scaled_train[train_idx]
                    
                    mlp = DeepNeuralComponent(
                        input_dim=len(ne.input_features),
                        n_neurons=ne.n_neurons,
                        n_hidden_layers=getattr(ne, 'n_hidden_layers', 1),
                        activation_name=ne.activation
                    ).to(TORCH_DEVICE)
                    
                    optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
                    )
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    best_model_state = None
                    epochs_no_improve = 0
                    train_size_curr = X_scaled_train.shape[0]
                    
                    for _ in range(self.epochs):
                        mlp.train()
                        epoch_indices = torch.randperm(train_size_curr, device=TORCH_DEVICE)
                        
                        for start_idx in range(0, train_size_curr, self.batch_size):
                            batch_idx = epoch_indices[start_idx:start_idx + self.batch_size]
                            optimizer.zero_grad()
                            loss = criterion(mlp(X_scaled_train[batch_idx]), Y_scaled_train[batch_idx])
                            loss.backward()
                            optimizer.step()
                        
                        mlp.eval()
                        with torch.no_grad():
                            val_loss = criterion(mlp(X_scaled_val), Y_scaled_val).item()
                        
                        scheduler.step(val_loss)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = copy.deepcopy(mlp.state_dict())
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        
                        if epochs_no_improve >= self.patience:
                            break
                    
                    if best_model_state is not None:
                        mlp.load_state_dict(best_model_state)
                    
                    mlp.eval()
                    self.mlps_[group_name][f_name] = mlp
                    
                    with torch.no_grad():
                        preds_scaled_train = mlp(X_train_dict[f_name]).cpu().numpy().flatten()
                        preds_raw_train = (preds_scaled_train * self.y_scalers_[group_name][f_name][1]) + self.y_scalers_[group_name][f_name][0]
                        
                        updated_pred_train = current_preds_train[f_name] + self.gam_shrinkage * (preds_raw_train - current_preds_train[f_name])
                        group_res_train = partial_res_train - updated_pred_train
                        current_preds_train[f_name] = updated_pred_train
                        
                        if group_data_val is not None:
                            preds_scaled_val = mlp(X_val_dict[f_name]).cpu().numpy().flatten()
                            preds_raw_val = (preds_scaled_val * self.y_scalers_[group_name][f_name][1]) + self.y_scalers_[group_name][f_name][0]
                            
                            updated_pred_val = current_preds_val[f_name] + self.gam_shrinkage * (preds_raw_val - current_preds_val[f_name])
                            group_res_val = partial_res_val - updated_pred_val
                            current_preds_val[f_name] = updated_pred_val
#: </orthogonal_backfitting>

#: <neural_decompose>
    def decompose_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes the prediction by evaluating parametric and trained neural effects.

        Args:
            data (pd.DataFrame): Dataset to generate predictions for.

        Returns:
            pd.DataFrame: A DataFrame containing individual effect columns and the total estimation.
        """

        data = _ensure_dummies(data, self.group_col_, self.date_col_)
        decomposed = self.base_additive_model.decompose_prediction(data)
        unique_groups = self.base_additive_model.unique_groups_
        
        neural_effects = [e for e in self.base_additive_model.effects_list_ if isinstance(e, NeuralEffect)]
        
        for ne in neural_effects:
            feature_name = ne.feature_name
            col_name = f'effect_{feature_name}'
            
            if col_name not in decomposed.columns:
                decomposed[col_name] = 0.0

            for group_name in unique_groups:
                if group_name not in self.mlps_ or feature_name not in self.mlps_[group_name]:
                    continue
                    
                mlp = self.mlps_[group_name][feature_name]
                mask = (data[self.group_col_] == group_name).values
                
                if not mask.any():
                    continue
                    
                group_data = data.iloc[mask]
                X_raw = torch.tensor(group_data[ne.input_features].values, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
                
                X_mean, X_std = self.x_scalers_[group_name][feature_name]
                Y_mean, Y_std = self.y_scalers_[group_name][feature_name]
                
                X_scaled = (X_raw - X_mean) / X_std
                
                with torch.no_grad():
                    mlp_preds_scaled = mlp(X_scaled).cpu().numpy().flatten()
                    mlp_preds_raw = (mlp_preds_scaled * Y_std) + Y_mean
                
                decomposed.loc[mask, col_name] = mlp_preds_raw
        
        effect_cols = [c for c in decomposed.columns if c.startswith('effect_')]
        decomposed[f'Estimated{self.target_col_}'] = decomposed[effect_cols].sum(axis=1)
        
        return decomposed
#: </neural_decompose>

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates final target predictions.

        Args:
            data (pd.DataFrame): The dataset containing input features.

        Returns:
            pd.DataFrame: DataFrame containing the predictions.
        """
        return self.decompose_prediction(data)