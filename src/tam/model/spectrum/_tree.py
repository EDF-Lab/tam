r"""
Implements the GPU-Native Tree / Random Forest Effect.

This module abstracts regression trees as sparse Euclidean bit-strings. 
Depending on the parameters provided, this module acts as a dual-architecture:
1. Oblivious Random Trees (max_depth): Generates symmetric, multi-dimensional 
   binary splits (2^D leaves). Ideal for spatial interactions (e.g., Lat/Lon).
2. Random Histograms (max_leaves): Generates flat, 1-Dimensional N-ary splits. 
   Ideal for Piecewise Linear Regressions and avoiding Over-Complete Matrix Singularities.

By utilizing PyTorch vectorization, both architectures generate an unbiased 
estimator of a shift-invariant kernel (Random Binning Features) entirely on the GPU, 
embedding algorithmic jump logic into the continuous TAM covariance matrix.
"""

import torch
import numpy as np
from typing import List, Optional

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_tree>
class TreeEffect(BaseEffect):
    r"""
    Implements a Random Forest effect using tensorized Random Binning Features.

    The input space is recursively partitioned into a set of disjoint, 
    hyper-rectangular terminal regions (leaves). 

    The output is a highly sparse Euclidean bit-string of leaf assignments, 
    scaled by 1/sqrt{B} to bound the Reproducing Kernel Hilbert Space (RKHS) 
    norm as the ensemble grows.

    Attributes:
        n_trees (int): Number of independent, randomized trees in the ensemble (B).
        max_depth (int): Max depth of each tree (D). Generates 2^D leaves per tree.
        max_leaves (int, optional): Overrides max_depth to force flat N-ary splits.
        lambda_p (float): Isotropic Ridge regularization strength.
        input_features (List[str]): List of all input feature names consumed.
        seed (int): Random seed for deterministic partition generation.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_trees: int, 
        max_depth: int,
        max_leaves: Optional[int], 
        lambda_p: float,
        additional_features: Optional[List[str]],
        seed: int,
        extrapolate: str,
        sparsity_alpha: float = 0.0,
        split_strategy: str = "uniform"
    ):
        super().__init__(feature_name, "tree", lambda_p, extrapolate)
        self.n_trees = n_trees
        self.seed = seed
        self.sparsity_alpha = sparsity_alpha
        self.split_strategy = split_strategy
        
        # Scaling factor to bound the RKHS norm as the ensemble size approaches infinity
        self.scale = 1.0 / np.sqrt(self.n_trees)

        # Support for multivariate partitioning (e.g., Lat/Lon interactions)
        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)

        # Architectural Router: Oblivious Binary Tree vs. Flat N-ary Histogram
        if max_leaves is not None:
            # Bypass binary depth entirely. Use explicit N-ary splits on a single feature.
            # This prevents the matrix singularity (0-data leaves) in piecewise regressions.
            self.leaves_per_tree = max_leaves
            self.n_splits_per_tree = max_leaves - 1
            self.is_oblivious_binary = False
        else:
            # Standard Oblivious Binary Tree behavior for multi-dimensional spatial data
            self.max_depth = max_depth
            self.leaves_per_tree = 2 ** max_depth
            self.n_splits_per_tree = max_depth
            self.is_oblivious_binary = True

        self.total_leaves = self.n_trees * self.leaves_per_tree

        # Tensors holding the random partition geometry
        self.split_features = None
        self.split_thresholds = None
        
        # Pre-computed bit-shift multipliers for fast binary leaf indexing
        self.depth_multipliers = None

    def get_n_coeffs(self) -> int:
        r"""
        Returns the total dimension of the sparse Primal dictionary.
        Corresponds exactly to the total number of leaves across all trees.
        """
        return self.total_leaves

    def _init_forest(self, x_in: torch.Tensor, is_dummy: bool):
        r"""
        Initializes the random partition geometry (split features and thresholds) 
        directly on the designated compute device.
        """
        # Safeguard: Do not permanently initialize the geometry during memory footprint probing
        if is_dummy:
            return

        n_features = x_in.shape[-1]
        device = x_in.device
        
        rng = torch.Generator(device=device)
        rng.manual_seed(self.seed)

        # Pre-compute domain bounds for backward compatibility ("uniform" strategy)
        x_min = x_in.min(dim=0).values.min()
        x_max = x_in.max(dim=0).values.max()
        if x_max == x_min:
            x_max += 1e-3
            x_min -= 1e-3

        if self.is_oblivious_binary:
            # --- Architecture 1: Oblivious Binary Tree ---
            self.split_features = torch.randint(
                0, n_features, size=(self.n_trees, self.max_depth), 
                generator=rng, device=device, dtype=torch.long
            )
            
            if self.split_strategy == "quantile":
                self.split_thresholds = torch.empty(
                    self.n_trees, self.max_depth, 
                    device=device, dtype=x_in.dtype
                )
                for i in range(self.n_trees):
                    for d in range(self.max_depth):
                        feat_idx = self.split_features[i, d]
                        # Safely slice across all batch/time dimensions and flatten
                        feature_data = x_in[..., feat_idx].flatten().to(torch.float64)
                        rand_prob = torch.rand(1, generator=rng, device=device, dtype=torch.float64)
                        self.split_thresholds[i, d] = torch.quantile(feature_data, rand_prob).to(x_in.dtype).squeeze()
            else:
                # Default "uniform" fallback for backward compatibility
                self.split_thresholds = (x_max - x_min) * torch.rand(
                    self.n_trees, self.max_depth, 
                    generator=rng, device=device, dtype=x_in.dtype
                ) + x_min

            # Cache binary multipliers [2^0, 2^1, ..., 2^(D-1)] for O(1) leaf indexing
            self.depth_multipliers = torch.pow(
                2, torch.arange(self.max_depth, device=device)
            ).unsqueeze(0).expand(self.n_trees, -1)
            
        else:
            # --- Architecture 2: Flat N-ary Histogram ---
            # Each tree randomly picks exactly ONE feature to slice into max_leaves
            self.split_features = torch.randint(
                0, n_features, size=(self.n_trees, 1), 
                generator=rng, device=device, dtype=torch.long
            )
            
            if self.n_trees == 1:
                # [Anti-Starvation Protocol]
                if self.split_strategy == "quantile":
                    probabilities = torch.linspace(
                        0.0, 1.0, self.leaves_per_tree + 1, 
                        device=device, dtype=torch.float64
                    )[1:-1]
                    
                    feat_idx = self.split_features[0, 0]
                    # Safely slice and flatten
                    feature_data = x_in[..., feat_idx].flatten().to(torch.float64)
                    
                    even_splits = torch.quantile(feature_data, probabilities).to(x_in.dtype).squeeze()
                    
                    # Squeeze guard for 1-split edge case
                    if even_splits.dim() == 0:
                        even_splits = even_splits.unsqueeze(0)
                    self.split_thresholds = even_splits.unsqueeze(0)
                else:
                    # Default "uniform" deterministic domain spacing
                    even_splits = torch.linspace(
                        x_min, x_max, self.leaves_per_tree + 1, 
                        device=device, dtype=x_in.dtype
                    )[1:-1]
                    self.split_thresholds = even_splits.unsqueeze(0)
            else:
                # [Ensemble Protocol]
                if self.split_strategy == "quantile":
                    raw_thresholds = torch.empty(
                        self.n_trees, self.n_splits_per_tree, 
                        device=device, dtype=x_in.dtype
                    )
                    
                    for i in range(self.n_trees):
                        feat_idx = self.split_features[i, 0]
                        # Safely slice and flatten
                        feature_data = x_in[..., feat_idx].flatten().to(torch.float64)
                        
                        rand_probs = torch.rand(
                            self.n_splits_per_tree, 
                            generator=rng, device=device, dtype=torch.float64
                        )
                        raw_thresholds[i] = torch.quantile(feature_data, rand_probs).to(x_in.dtype)
                        
                    self.split_thresholds, _ = torch.sort(raw_thresholds, dim=-1)
                else:
                    # Default "uniform" domain sampling
                    raw_thresholds = (x_max - x_min) * torch.rand(
                        self.n_trees, self.n_splits_per_tree, 
                        generator=rng, device=device, dtype=x_in.dtype
                    ) + x_min
                    self.split_thresholds, _ = torch.sort(raw_thresholds, dim=-1)
                    
        # Evaluate the routing logic strictly on the initialization (training) data
        x_expanded = x_in.unsqueeze(-2).unsqueeze(-2)
        batch_shape = list(x_in.shape[:-1])
        
        if self.is_oblivious_binary:
            index_shape = batch_shape + [self.n_trees, self.max_depth, 1]
            split_feat_expanded = self.split_features.view(
                *([1] * len(batch_shape)), self.n_trees, self.max_depth, 1
            ).expand(*index_shape)

            input_expanded = x_expanded.expand(
                *batch_shape, self.n_trees, self.max_depth, x_in.shape[-1]
            )
            x_splits = torch.gather(input_expanded, dim=-1, index=split_feat_expanded).squeeze(-1)
            split_decisions = (x_splits > self.split_thresholds).to(torch.long)
            leaf_indices = torch.sum(split_decisions * self.depth_multipliers, dim=-1)
        else:
            index_shape = batch_shape + [self.n_trees, 1, 1]
            split_feat_expanded = self.split_features.view(
                *([1] * len(batch_shape)), self.n_trees, 1, 1
            ).expand(*index_shape)

            input_expanded = x_expanded.expand(
                *batch_shape, self.n_trees, 1, x_in.shape[-1]
            )
            x_splits = torch.gather(input_expanded, dim=-1, index=split_feat_expanded).squeeze(-1).squeeze(-1)
            
            x_splits_expand = x_splits.unsqueeze(-1)
            thresh_expand = self.split_thresholds.view(*([1] * len(batch_shape)), self.n_trees, self.n_splits_per_tree)
            leaf_indices = torch.sum((x_splits_expand > thresh_expand).to(torch.long), dim=-1)

        # Count how many data points land in each leaf
        one_hot_bins = torch.nn.functional.one_hot(leaf_indices, num_classes=self.leaves_per_tree)
        # Sum across the batch dimension (dim=0)
        self.empirical_counts = one_hot_bins.sum(dim=0).flatten()
#: </init_tree>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Projects inputs into the sparse bit-string Primal representation natively on GPU.
        Computes the mutually exclusive indicator functions for every tree simultaneously.
        """
        # 1. Architectural Shape Normalization (The Ultimate Router)
        if x_col.dim() == 1:
            # 1D: From OOD Wrapper (Univariate) -> [N_ood]
            batch_shape = list(x_col.shape)
            x_in = x_col.unsqueeze(-1)
        elif x_col.dim() == 2:
            if x_col.shape[-1] == len(self.input_features):
                # Ambiguity Resolution: Is it the te() dummy pass [1, 1], or OOD Wrapper [N_ood, Features]?
                if self.split_features is None and x_col.shape == (1, 1) and len(self.input_features) == 1:
                    batch_shape = list(x_col.shape)
                    x_in = x_col.unsqueeze(-1)
                else:
                    batch_shape = list(x_col.shape[:-1])
                    x_in = x_col
            else:
                # 2D: From te() regular pass [Batch, Time]
                batch_shape = list(x_col.shape)
                x_in = x_col.unsqueeze(-1)
        else:
            # 3D+: From _factory.py [Batch, Time, Features]
            batch_shape = list(x_col.shape[:-1])
            x_in = x_col

        # 2. Bulletproof Dummy Pass Detection (Memory Probe)
        is_dummy = False
        if self.split_features is None:
            if batch_shape == [1, 1]:
                is_dummy = True

        if self.split_features is None:
            self._init_forest(x_in, is_dummy)

        if is_dummy:
            return torch.zeros(
                *batch_shape, self.total_leaves, 
                device=x_in.device, dtype=torch.get_default_dtype()
            )

        # 3. Vectorized Evaluation
        x_expanded = x_in.unsqueeze(-2).unsqueeze(-2)

        if self.is_oblivious_binary:
            index_shape = batch_shape + [self.n_trees, self.max_depth, 1]
            split_feat_expanded = self.split_features.view(
                *([1] * len(batch_shape)), self.n_trees, self.max_depth, 1
            ).expand(*index_shape)

            input_expanded = x_expanded.expand(
                *batch_shape, self.n_trees, self.max_depth, x_in.shape[-1]
            )
            x_splits = torch.gather(input_expanded, dim=-1, index=split_feat_expanded).squeeze(-1)
            split_decisions = (x_splits > self.split_thresholds).to(torch.long)
            leaf_indices = torch.sum(split_decisions * self.depth_multipliers, dim=-1)

        else:
            index_shape = batch_shape + [self.n_trees, 1, 1]
            split_feat_expanded = self.split_features.view(
                *([1] * len(batch_shape)), self.n_trees, 1, 1
            ).expand(*index_shape)

            input_expanded = x_expanded.expand(
                *batch_shape, self.n_trees, 1, x_in.shape[-1]
            )
            x_splits = torch.gather(input_expanded, dim=-1, index=split_feat_expanded).squeeze(-1).squeeze(-1)
            
            x_splits_expand = x_splits.unsqueeze(-1)
            thresh_expand = self.split_thresholds.view(*([1] * len(batch_shape)), self.n_trees, self.n_splits_per_tree)
            leaf_indices = torch.sum((x_splits_expand > thresh_expand).to(torch.long), dim=-1)

        one_hot_bins = torch.nn.functional.one_hot(leaf_indices, num_classes=self.leaves_per_tree)

        # 4. Final Output Tensor
        # Output shape perfectly matches the required Basis Tensor format [*batch_shape, Leaves]
        phi_tensor = one_hot_bins.view(*batch_shape, self.total_leaves).to(torch.float64)

        # Apply the RKHS normalization bound IN-PLACE
        phi_tensor.mul_(self.scale)
        
        return phi_tensor
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Constructs the optimal structural penalty sub-matrix.
        
        If sparsity_alpha > 0, the penalty dynamically adapts to the empirical 
        data density of each specific leaf. Starved leaves receive massive penalties 
        to prevent overfitting, while dense leaves receive standard shrinkage.
        """
        if self.sparsity_alpha > 0.0 and hasattr(self, 'empirical_counts'):
            # Calculate Anisotropic Penalty based on empirical data counts
            C_i = self.empirical_counts.to(TORCH_DEVICE, dtype=torch.float64)
            C_bar = C_i.mean()
            epsilon = 1.0  # Smoothing constant to prevent division by zero
            
            # Starved leaves (< C_bar) will get factors > 1.0
            # Dense leaves (> C_bar) will get factors < 1.0
            penalty_scaling = ((C_i + epsilon) / C_bar) ** (-self.sparsity_alpha)
            diag_vals = self.lambda_p * penalty_scaling
        else:
            # Fallback to pure Isotropic Penalty (or for alpha = 0)
            diag_vals = torch.full(
                (self.total_leaves,), self.lambda_p, 
                device=TORCH_DEVICE, dtype=torch.get_default_dtype()
            )
        
        # Build as a sparse COO tensor directly to save VRAM
        indices = torch.arange(self.total_leaves, device=TORCH_DEVICE)
        indices = torch.stack([indices, indices], dim=0)
        
        if hasattr(torch.sparse, 'check_sparse_tensor_invariants'):
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    indices, 
                    diag_vals, 
                    size=(self.total_leaves, self.total_leaves), 
                    device=TORCH_DEVICE
                )
        else:
            return torch.sparse_coo_tensor(
                indices, 
                diag_vals, 
                size=(self.total_leaves, self.total_leaves), 
                device=TORCH_DEVICE
            )
#: </penalty_matrix>