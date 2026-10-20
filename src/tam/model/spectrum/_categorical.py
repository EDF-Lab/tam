# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements categorical effects via One-Hot Encoding."""

from typing import List, Optional

import torch
import numpy as np
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_categorical>
class CategoricalEffect(BaseEffect):
    r"""
    Categorical effect supporting Nominal (Ridge), Ordinal (Difference), Fourier (cycle) and Graph (Markov random field) topologies.

    ``topology='graph'`` links each level to its ``k_neighbors`` nearest levels, the distance being that of the centroids (mean of the
    ``coords`` of the training rows of each level), and penalises the graph Laplacian ``D - A``: a level with few rows borrows strength
    from the levels around it. A level with no training row has no centroid and keeps the zero effect.

    A code that is none of the levels (outside ``0 .. n_categories - 1`` once normalised) takes, with ``unseen='edge'`` (the default), the effect of the
    nearest edge level; with ``unseen='mean'`` the
    mean of the effects of the levels weighted by their number of training rows (a proxy for an unknown level: the average level), and with
    ``unseen='zero'`` no effect at all (the prediction falls back on the intercept and the other terms), and the behaviour of ``'edge'`` is the one before this option existed. A term with a single level (a flag that is constant in the training
    rows) keeps that level for every row: its column is then the constant of the intercept, and dropping it would remove part of the intercept. ``vocabulary`` is set when the levels are the codes of the training
    rows (``n_cat`` omitted in the formula); the model then maps the codes to level indices before the effect sees them.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_categories: int, 
        topology: str, 
        lambda_p: float,
        penalty_order: int,
        extrapolate: str,
        coords: Optional[List[str]] = None,
        k_neighbors: int = 6,
        unseen: str = 'edge'
    ):
        if str(unseen).lower() not in ('mean', 'zero', 'edge'):
            raise ValueError(f"unseen must be 'mean', 'zero' or 'edge'; got {unseen!r}")
        self.unseen = str(unseen).lower()
        self._level_weights = None
        clean_topo = topology.lower()
        if clean_topo not in ['nominal', 'ordinal', 'fourier', 'graph']:
            raise ValueError(f"Topology '{topology}' invalid. Use 'nominal', 'ordinal', 'fourier' or 'graph'.")
        if clean_topo == 'graph' and not coords:
            raise ValueError(f"c({feature_name}, topo='graph') needs the coordinates of the levels: coords='x:y'.")

        super().__init__(feature_name, f"categorical_{clean_topo}", lambda_p, extrapolate)
        self.n_categories = int(n_categories)
        self.vocabulary = False
        self.topology = clean_topo
        self.input_features = [feature_name] + (list(coords) if clean_topo == 'graph' else [])
        self.k_neighbors = int(k_neighbors)
        self._laplacian = None
        self.penalty_order = penalty_order

        if self.topology == 'fourier':
            self.m = self.n_categories // 2 + self.n_categories % 2
            self.s = 0

    def summary_details(self) -> str:
        return f"Cats={self.n_categories}, Topo={self.topology}, Unseen={self.unseen}"

    def get_n_coeffs(self) -> int:
        if self.topology == 'fourier':
            return 2 * self.m
        return self.n_categories

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Share of the training rows of each level (the weights of the mean effect given to an unseen code); graph topology: also the
        centroid of each level from the training tensor (level, coordinates...) and the Laplacian of the k nearest."""
        if self.topology == 'fourier':
            return
        n = self.n_categories
        flat = x_cols.reshape(-1, x_cols.shape[-1]).detach().to(torch.get_default_dtype()).cpu()
        index = torch.round((flat[:, 0] + 1.0) / 2.0 * (n - 1)).long() if n > 1 else torch.zeros(flat.shape[0], dtype=torch.long)
        valid = (index >= 0) & (index < n)
        if self._level_weights is None and bool(valid.any()):
            shares = torch.bincount(index[valid], minlength=n).to(torch.get_default_dtype())
            self._level_weights = shares / shares.sum()
        if self.topology != 'graph' or self._laplacian is not None:
            return
        coordinates = flat[valid, 1:]
        index = index[valid]
        sums = torch.zeros(n, coordinates.shape[1], dtype=coordinates.dtype)
        counts = torch.zeros(n, dtype=coordinates.dtype)
        sums.index_add_(0, index, coordinates)
        counts.index_add_(0, index, torch.ones(index.shape[0], dtype=coordinates.dtype))
        present = torch.nonzero(counts > 0).flatten()
        centroids = sums[present] / counts[present].unsqueeze(-1)
        adjacency = torch.zeros(n, n, dtype=coordinates.dtype)
        k = min(self.k_neighbors, present.numel() - 1)
        if k >= 1:
            distances = torch.cdist(centroids, centroids)
            distances.fill_diagonal_(float('inf'))
            neighbours = torch.topk(distances, k, largest=False).indices
            for row in range(present.numel()):
                adjacency[present[row], present[neighbours[row]]] = 1.0
            adjacency = torch.maximum(adjacency, adjacency.T)
        self._laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
    
#: </init_categorical>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Projects inputs based on topology:
        - Nominal/Ordinal: One-Hot basis.
        - Fourier: Sin/cos basis.
        """

        if self.topology == 'graph' and x_col.shape[-1] == len(self.input_features):
            x_col = x_col[..., 0]  # the other columns are the coordinates of the levels, read by `initialize`

        if self.topology in ['nominal', 'ordinal', 'graph']:
            x_indices = torch.round((x_col + 1.0) / 2.0 * (self.n_categories - 1)).long()
            if self.n_categories == 1:
                known = torch.ones_like(x_col, dtype=torch.bool)  # a single level (a feature constant in training) is every row, as before
            else:
                known = (x_indices >= 0) & (x_indices <= self.n_categories - 1)
            x_safe = torch.clamp(x_indices, 0, self.n_categories - 1)
            phi = torch.nn.functional.one_hot(x_safe, num_classes=self.n_categories)
            if self.unseen != 'edge':
                phi = phi * known.unsqueeze(-1).to(phi.dtype)  # a code outside the levels is not any level
            phi = phi.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            if self.unseen == 'mean' and self._level_weights is not None:
                # ... and takes the mean effect of the levels, weighted by their training rows (unseen='zero': the zero row)
                mean_row = self._level_weights.to(device=phi.device, dtype=phi.dtype)
                phi = phi + (~known).unsqueeze(-1).to(phi.dtype) * mean_row
            return phi
        
        elif self.topology == 'fourier':
            x_scaled = x_col * np.pi
            x_expanded = x_scaled.unsqueeze(-1)
            freqs = torch.arange(1, self.m + 1, device=x_col.device, dtype=torch.get_default_dtype())
            dims_to_add = x_expanded.dim() - 1
            freqs_expanded = freqs.view(*([1] * dims_to_add), -1)
            theta = x_expanded * freqs_expanded / 2
            return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)    
        
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds penalty: Identity (Nominal) or Difference (Ordinal)."""
        if self.topology == 'graph':
            eye = torch.eye(self.n_categories, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            if self._laplacian is None:
                return eye * self.lambda_p
            return self.lambda_p * (self._laplacian.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype()) + 1e-4 * eye)

        if self.topology == 'nominal':
            return torch.eye(
                self.n_categories, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
            ) * self.lambda_p
                        
        elif self.topology == 'ordinal':
            # Create the identity matrix directly on the GPU
            I = torch.eye(self.n_categories, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            # Compute differences natively on the GPU
            D = torch.diff(I, n=self.penalty_order, dim=0)
            P = D.T @ D
            return self.lambda_p * P
        
        elif self.topology == 'fourier':            
            freqs = torch.arange(1, self.m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            penalty_half = self.lambda_p * (1 + freqs ** (2 * self.s))
            diag_full = torch.cat([penalty_half, penalty_half])
            return torch.diag(diag_full)
        
        return torch.zeros(1)
#: </penalty_matrix>