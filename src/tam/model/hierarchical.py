# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Hierarchical Joint Solver Module.

This module implements the HierarchicalTAM class, which trains multiple 
StaticTAM models simultaneously. It links them through aggregation constraints 
(Parent = sum(Children)) enforced via regularization in the global loss function.
"""

import warnings

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, NamedTuple, Union, Optional, Any
from tam.common.utils import TORCH_DEVICE, _ensure_dummies, _cleanup_dummies
from tam.common.hardware import hw

from .additive import StaticTAM
from ._memory import can_fit_dense_matrix, get_safe_chunk_size
from ._math import solve_sparse_cg

#: Relative gap above which a parent is reported as not being the sum of its children: the data are not an exact hierarchy.
_ADDITIVITY_TOLERANCE = 1e-3

#: <hierarchical_system>
class _HierarchicalSystem(NamedTuple):
    r"""The assembled joint system: ``cov_X`` and ``P`` are ``None`` when only the matrix-free solve is possible."""
    node_data_dict: Dict[str, Any]
    node_to_idx: Dict[str, Any]
    penalty_blocks: List[torch.Tensor]
    total_coeffs: int
    n_samples: int
    run_device: Any
    direct: bool
    cov_X: Optional[torch.Tensor]
    cov_XY: torch.Tensor
    P: Optional[torch.Tensor]
#: </hierarchical_system>

#: <init_hierarchical>
class HierarchicalTAM:
    r"""
    Joint Hierarchical Solver.

    Trains multiple StaticTAM models simultaneously, linked by aggregation constraints.
    Instead of fitting each node independently, it solves a global linear system
    that penalizes deviations from the hierarchy (Parent = Sum of Children).

    Attributes:
        structure (Dict[str, List[str]]): Hierarchy definition {Parent: [Child1, Child2]}.
        sub_models (Dict[str, StaticTAM]): Dictionary of initialized sub-models per node.
        lambda_p_hier (float): Strength of the hierarchical constraint regularization.
        align (str): How the nodes are matched on ``(group, date)`` at fit, ``"strict"`` or ``"inner"``.
        additivity_gap_ (Dict[str, float]): After fit, per parent ``max |Y_p - sum Y_c| / mean |Y_p|``.
    """
    
    def __init__(
        self, 
        structure: Dict[str, List[str]], 
        formulas: Union[str, Dict[str, str]],
        node_col: str,
        group_col: Optional[str] = None,
        date_col: Optional[str] = None,
        lambda_p_hier: float = 1.0,
        align: str = "strict"
    ):
        r"""
        Initializes the HierarchicalTAM model.

        Args:
            structure: Dictionary defining the hierarchy trees (Parent -> Children).
            formulas: A single string formula applied to all nodes, or a dictionary 
                      mapping node names to specific formulas.
            node_col: Column name defining the hierarchy nodes (e.g., 'Region').
            group_col: Optional column name used for 3D tensor batching (e.g., 'TimeOfDay').
            date_col: Optional column name for time indexing.
            lambda_p_hier: Regularization strength for the aggregation constraint.
            align: ``"strict"`` raises when the nodes do not share the same ``(group, date)`` keys and lists the first missing keys
                per node; ``"inner"`` keeps the keys every node has and warns with the rows dropped per node. Without a
                ``date_col`` the nodes are matched by row position and must have the same number of rows.
        """
        if align not in ("strict", "inner"):
            raise ValueError(f"HierarchicalTAM: align must be 'strict' or 'inner', got {align!r}.")
        self.align = align
        self.structure = structure
        self.node_col = node_col
        self.group_col = group_col or "__dummy_group__"
        self.date_col = date_col or "__dummy_date__"
        self.lambda_p_hier = lambda_p_hier
        
        self.nodes = set(structure.keys())
        for children in structure.values():
            self.nodes.update(children)
        self.nodes = sorted(list(self.nodes))
        
        self.sub_models = {}
        for node in self.nodes:
            f = formulas if isinstance(formulas, str) else formulas[node]
            self.sub_models[node] = StaticTAM(
                formula=f, 
                group_col=self.group_col, 
                date_col=self.date_col
            )
            
        self.global_coeffs_ = None
        self.additivity_gap_ = {}
#: </init_hierarchical>

#: <fit_hierarchical>
    def fit(self, data: pd.DataFrame) -> 'HierarchicalTAM':
        r"""
        Fits the global hierarchical system.

        Constructs a block-diagonal matrix of all sub-models and adds off-diagonal
        terms representing the hierarchical constraints before solving. Uses
        memory-safe chunking to prevent VRAM exhaustion.

        Args:
            data: DataFrame containing data for all nodes (distinguished by node_col).
        """
        print("--- Fitting Hierarchical System ---")

        system = self._assemble_system(data)
        beta_global = self._solve(system)

        for node in self.nodes:
            start, end = system.node_to_idx[node]
            self.sub_models[node].coefficients_ = beta_global[start:end, :].unsqueeze(0)

        return self

    def _align_nodes(self, data: pd.DataFrame) -> pd.DataFrame:
        r"""
        Matches the nodes on ``(group, date)`` and sorts each node by ``(group, date)``.

        Raises:
            ValueError: ``"strict"``: the nodes do not hold the same keys (the first missing keys are listed per node).
                ``"inner"``: no key is common to every node. Either mode: a node repeats a key, or, without a ``date_col``,
                the nodes differ in row count.
        """
        frames = {node: data[data[self.node_col] == node] for node in self.nodes}

        if self.date_col == "__dummy_date__":
            counts = {node: len(frame) for node, frame in frames.items()}
            if len(set(counts.values())) > 1:
                raise ValueError(
                    "HierarchicalTAM: without a date_col the nodes are matched by row position and must have the same row count, "
                    f"got {counts}. Pass date_col to match them on (group, date)."
                )
            return pd.concat(list(frames.values()), ignore_index=True)

        keys = {node: set(zip(frame[self.group_col], frame[self.date_col])) for node, frame in frames.items()}
        union = set().union(*keys.values())
        common = set.intersection(*keys.values())
        label = (lambda k: str(pd.Timestamp(k[1]))) if self.group_col == "__dummy_group__" else (lambda k: f"({k[0]}, {pd.Timestamp(k[1])})")

        if self.align == "strict":
            missing = {node: sorted(union - keys[node]) for node in self.nodes if keys[node] != union}
            if missing:
                lines = [f"  {node}: {len(m)} missing, first: {[label(k) for k in m[:3]]}" for node, m in missing.items()]
                raise ValueError(
                    "HierarchicalTAM: the nodes do not share the same (group, date) keys (align='strict'). "
                    "Keys that other nodes have and this node lacks:\n" + "\n".join(lines)
                    + "\nUse align='inner' to keep the common keys."
                )
        else:
            if not common:
                raise ValueError("HierarchicalTAM: align='inner' found no common (group, date) key across the nodes.")
            dropped = {node: len(frames[node]) - sum(k in common for k in zip(frames[node][self.group_col], frames[node][self.date_col]))
                       for node in self.nodes}
            dropped = {node: n for node, n in dropped.items() if n}
            if dropped:
                warnings.warn(
                    f"HierarchicalTAM align='inner': kept {len(common)} common (group, date) keys; rows dropped per node: "
                    + ", ".join(f"{node}: {n}" for node, n in dropped.items()),
                    UserWarning, stacklevel=3,
                )

        aligned = []
        for node in self.nodes:
            frame = frames[node]
            if self.align == "inner":
                frame = frame[[k in common for k in zip(frame[self.group_col], frame[self.date_col])]]
            frame = frame.sort_values([self.group_col, self.date_col], kind="stable")
            if len(frame) != len(common if self.align == "inner" else union):
                raise ValueError(f"HierarchicalTAM: node {node!r} repeats a (group, date) key; each node needs one row per key.")
            aligned.append(frame)
        return pd.concat(aligned, ignore_index=True)

    def _check_additivity(self, data: pd.DataFrame) -> None:
        r"""
        Stores ``additivity_gap_`` per parent, :math:`\max |Y_p - \sum_c Y_c| / \operatorname{mean} |Y_p|`, and warns once when
        a parent is further than ``_ADDITIVITY_TOLERANCE`` from the sum of its children (``data`` is aligned and sorted).
        """
        target = lambda node: data.loc[data[self.node_col] == node, self.sub_models[node].target_col_].to_numpy(dtype=float)
        gaps = {}
        for parent, children in self.structure.items():
            y_parent = target(parent)
            difference = np.abs(y_parent - sum(target(child) for child in children)).max()
            scale = np.abs(y_parent).mean()
            gaps[parent] = float(difference / scale) if scale > 0 else (0.0 if difference == 0 else float("inf"))
        self.additivity_gap_ = gaps
        off = {parent: gap for parent, gap in gaps.items() if gap > _ADDITIVITY_TOLERANCE}
        if off:
            warnings.warn(
                "HierarchicalTAM: the target of a parent is not the sum of its children's "
                f"(max |Y_parent - sum Y_children| / mean |Y_parent|, tolerance {_ADDITIVITY_TOLERANCE:g}): "
                + ", ".join(f"{parent}: {gap:.4g}" for parent, gap in off.items()),
                UserWarning, stacklevel=3,
            )

    def _assemble_system(self, data: pd.DataFrame) -> '_HierarchicalSystem':
        r"""
        Assembles the global system :math:`(\Sigma_X, \Sigma_{XY}, P)` of the joint fit.

        The dense matrices ``cov_X`` (data terms plus the aggregation constraints) and ``P`` (block-diagonal node penalties)
        are built when they fit in memory. Otherwise only ``cov_XY`` is accumulated and :meth:`_solve` applies the system
        matrix-free from the node data.
        """
        data = _ensure_dummies(data, self.group_col, self.date_col)
        data = self._align_nodes(data)
        self._check_additivity(data)

        node_data_dict = {}
        node_to_idx = {}
        penalty_blocks = []
        current_idx = 0

        for node in self.nodes:
            # Isolate data based on the structural hierarchy node, not the tensor group
            node_data = data[data[self.node_col] == node].reset_index(drop=True)
            x, y, _ = self.sub_models[node]._prepare_data(
                node_data, target_col=self.sub_models[node].target_col_
            )
            node_data_dict[node] = (x.squeeze(0), y.squeeze(0))

            dummy_phi = self.sub_models[node]._build_design_matrix(x[:, 0:1, :]).squeeze(0)
            k = dummy_phi.shape[1]
            node_to_idx[node] = (current_idx, current_idx + k)
            current_idx += k
            penalty_blocks.append(self.sub_models[node]._build_penalty_matrix())
            del dummy_phi

        total_coeffs = current_idx
        n_samples = node_data_dict[self.nodes[0]][0].shape[0]
        run_device = TORCH_DEVICE

        is_safe_for_direct = can_fit_dense_matrix(total_coeffs, run_device, batch_size=1)
        cov_bytes = 1 * total_coeffs * total_coeffs * 8 # 8 bytes for float64
        available_bytes = hw.get_available_memory()
        is_safe_for_dense_cov = cov_bytes < (available_bytes * 0.4)

        if is_safe_for_direct or is_safe_for_dense_cov:
            cov_X_global = torch.zeros((total_coeffs, total_coeffs), device=run_device, dtype=torch.get_default_dtype())
            cov_XY_global = torch.zeros((total_coeffs, 1), device=run_device, dtype=torch.get_default_dtype())
            P_global = torch.zeros((total_coeffs, total_coeffs), device=run_device, dtype=torch.get_default_dtype())

            for i, node in enumerate(self.nodes):
                start, end = node_to_idx[node]
                p_block = penalty_blocks[i].to(run_device)
                if p_block.dim() == 3:
                    p_block = p_block.squeeze(0)
                P_global[start:end, start:end] = p_block

            safe_batch = max(256, get_safe_chunk_size(n_samples, total_coeffs, run_device))
            start_idx = 0

            print("Accumulating Block Matrices and Constraints...")
            while start_idx < n_samples:
                end_idx = min(start_idx + safe_batch, n_samples)

                try:
                    phi_chunks = {}
                    y_chunks = {}

                    for node in self.nodes:
                        x_full, y_full = node_data_dict[node]
                        x_chunk = x_full[start_idx:end_idx, :].to(run_device)
                        y_chunks[node] = y_full[start_idx:end_idx, :].to(run_device)
                        phi_chunks[node] = self.sub_models[node]._build_design_matrix(x_chunk.unsqueeze(0)).squeeze(0)

                    for node in self.nodes:
                        start, end = node_to_idx[node]
                        phi_c = phi_chunks[node]
                        cov_X_global[start:end, start:end] += phi_c.mT @ phi_c
                        cov_XY_global[start:end, :] += phi_c.mT @ y_chunks[node]

                    for parent, children in self.structure.items():
                        p_start, p_end = node_to_idx[parent]
                        phi_p = phi_chunks[parent]

                        cov_X_global[p_start:p_end, p_start:p_end] += self.lambda_p_hier * (phi_p.mT @ phi_p)

                        for child in children:
                            c_start, c_end = node_to_idx[child]
                            phi_c = phi_chunks[child]

                            cov_X_global[c_start:c_end, c_start:c_end] += self.lambda_p_hier * (phi_c.mT @ phi_c)

                            cross = -self.lambda_p_hier * (phi_p.mT @ phi_c)
                            cov_X_global[p_start:p_end, c_start:c_end] += cross
                            cov_X_global[c_start:c_end, p_start:p_end] += cross.mT

                            for sibling in children:
                                if sibling != child:
                                    s_start, s_end = node_to_idx[sibling]
                                    phi_s = phi_chunks[sibling]
                                    cov_X_global[c_start:c_end, s_start:s_end] += self.lambda_p_hier * (phi_c.mT @ phi_s)

                    del phi_chunks, y_chunks
                    start_idx += safe_batch

                except (torch.OutOfMemoryError, MemoryError):
                    safe_batch, run_device = hw.handle_oom(
                        current_batch=safe_batch,
                        device=run_device,
                        context="Hierarchical dense matrix accumulation",
                        allow_cpu_fallback=False
                    )
                    safe_batch = max(256, safe_batch)
                    continue

            return _HierarchicalSystem(
                node_data_dict, node_to_idx, penalty_blocks, total_coeffs, n_samples, run_device,
                is_safe_for_direct, cov_X_global, cov_XY_global, P_global,
            )

        print(f"Notice: Hierarchical D={total_coeffs} is massive. Routing to CG Solver.")

        cov_XY_global = torch.zeros((total_coeffs, 1), device=run_device, dtype=torch.get_default_dtype())
        safe_batch = max(256, get_safe_chunk_size(n_samples, total_coeffs, run_device))

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + safe_batch, n_samples)
            try:
                for node in self.nodes:
                    start, end = node_to_idx[node]
                    x_chunk = node_data_dict[node][0][start_idx:end_idx, :].to(run_device)
                    y_chunk = node_data_dict[node][1][start_idx:end_idx, :].to(run_device)
                    phi_chunk = self.sub_models[node]._build_design_matrix(x_chunk.unsqueeze(0)).squeeze(0)
                    cov_XY_global[start:end, :] += phi_chunk.mT @ y_chunk
                start_idx += safe_batch
            except (torch.OutOfMemoryError, MemoryError):
                safe_batch, run_device = hw.handle_oom(
                    current_batch=safe_batch,
                    device=run_device,
                    context="Hierarchical CG RHS computation",
                    allow_cpu_fallback=False
                )
                continue

        return _HierarchicalSystem(
            node_data_dict, node_to_idx, penalty_blocks, total_coeffs, n_samples, run_device,
            False, None, cov_XY_global, None,
        )

    def _solve(self, system: '_HierarchicalSystem') -> torch.Tensor:
        r"""
        Solves the assembled global system: direct inversion, dense CG, or matrix-free CG, as chosen when it was assembled.
        """
        node_data_dict, node_to_idx, penalty_blocks = system.node_data_dict, system.node_to_idx, system.penalty_blocks
        total_coeffs, n_samples, run_device = system.total_coeffs, system.n_samples, system.run_device
        cov_X_global, cov_XY_global, P_global = system.cov_X, system.cov_XY, system.P

        if cov_X_global is not None:
            if system.direct:
                print("Solving Global System via Direct Inversion...")
                jitter_scale = 1e-6 * n_samples
                A = cov_X_global + n_samples * P_global + jitter_scale * torch.eye(total_coeffs, device=run_device, dtype=torch.get_default_dtype())
                return hw.safe_solve(A, cov_XY_global)

            print("Notice: LAPACK workspace exceeded. Routing to Fast Dense CG solver...")
            def compute_dense_hierarchical_Av(v: torch.Tensor) -> torch.Tensor:
                nonlocal run_device
                v = v.to(run_device)
                jitter_scale = 1e-6 * n_samples
                return (cov_X_global @ v) + n_samples * (P_global @ v) + (jitter_scale * v)

            return solve_sparse_cg(compute_dense_hierarchical_Av, cov_XY_global, tol=1e-6, max_iter=5000)

        def compute_hierarchical_Av(v: torch.Tensor) -> torch.Tensor:
            nonlocal run_device
            Av = torch.zeros_like(v)
            local_start = 0
            local_safe_batch = get_safe_chunk_size(n_samples, total_coeffs, run_device)

            while local_start < n_samples:
                local_end = min(local_start + local_safe_batch, n_samples)

                try:
                    phi_chunks = {}
                    for node in self.nodes:
                        x_chunk = node_data_dict[node][0][local_start:local_end, :].to(run_device)
                        phi_chunks[node] = self.sub_models[node]._build_design_matrix(x_chunk.unsqueeze(0)).squeeze(0)

                    for i, node in enumerate(self.nodes):
                        start, end = node_to_idx[node]
                        v_local = v[start:end, :]
                        phi = phi_chunks[node]
                        Av[start:end, :] += (phi.mT @ (phi @ v_local))

                    for parent, children in self.structure.items():
                        p_start, p_end = node_to_idx[parent]
                        v_p = v[p_start:p_end, :]
                        phi_p = phi_chunks[parent]

                        constraint_diff = phi_p @ v_p
                        for child in children:
                            c_start, c_end = node_to_idx[child]
                            phi_c = phi_chunks[child]
                            constraint_diff -= (phi_c @ v[c_start:c_end, :])

                        Av[p_start:p_end, :] += self.lambda_p_hier * (phi_p.mT @ constraint_diff)

                        for child in children:
                            c_start, c_end = node_to_idx[child]
                            phi_c = phi_chunks[child]
                            Av[c_start:c_end, :] -= self.lambda_p_hier * (phi_c.mT @ constraint_diff)

                    local_start += local_safe_batch

                except (torch.OutOfMemoryError, MemoryError):
                    local_safe_batch, run_device = hw.handle_oom(
                        current_batch=local_safe_batch,
                        device=run_device,
                        context="Hierarchical Av computation",
                        allow_cpu_fallback=False
                    )
                    continue

            for i, node in enumerate(self.nodes):
                start, end = node_to_idx[node]
                v_local = v[start:end, :]
                P = penalty_blocks[i].to(run_device)
                if P.dim() == 3:
                    P = P.squeeze(0)
                Av[start:end, :] += n_samples * (P @ v_local)

            jitter_scale = 1e-6 * n_samples
            return Av + (jitter_scale * v)

        return solve_sparse_cg(compute_hierarchical_Av, cov_XY_global, tol=1e-6, max_iter=5000)
#: </fit_hierarchical>

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        r"""
        Generates predictions for all nodes in the hierarchy.
        
        Args:
            data: Input DataFrame containing data for all nodes.
            
        Returns:
            pd.DataFrame: Concatenated predictions for all nodes.
        """
        data = _ensure_dummies(data, self.group_col, self.date_col)
        
        results = []
        for node in self.nodes:
            node_data = data[data[self.node_col] == node].copy()
            if not node_data.empty:
                results.append(self.sub_models[node].predict(node_data))
                
        final_preds = pd.concat(results)
        return _cleanup_dummies(final_preds, self.group_col, self.date_col)

#: <grid_search_hierarchical>
    def grid_search_fit(
        self, 
        data_train: pd.DataFrame, 
        data_val: pd.DataFrame, 
        grid_config: Dict[str, List[Any]], 
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> 'HierarchicalTAM':
        r"""
        Joint optimization of hyperparameters and hierarchical constraint strength.
        
        Uses Coordinate Descent to find optimal formula tokens and lambda_p_hier.
        
        Args:
            data_train: Training DataFrame.
            data_val: Validation DataFrame.
            grid_config: Dictionary mapping parameter names (tokens) to value lists.
            scoring_weights: Optional weights for each node in the final score calculation.
        """
        print("--- Starting Hierarchical Grid Search ---")
        
        tokens = {}
        if 'lambda_p_hier' in grid_config:
            tokens['lambda_p_hier'] = grid_config['lambda_p_hier']
            
        for node, model in self.sub_models.items():
            parsed = model.parsed_terms_ 
            for term in parsed:
                for k, v in term['params'].items():
                    if isinstance(v, str) and v in grid_config and v not in tokens:
                        tokens[v] = grid_config[v]
                        
        token_names = list(tokens.keys())
        print(f"Optimizing axes: {token_names}")
        
        if not token_names:
            self.fit(data_train)
            return self
            
        if scoring_weights is None:
            scoring_weights = {n: 1.0 for n in self.nodes}

        def evaluate_config(current_tokens: Dict) -> float:
            if 'lambda_p_hier' in current_tokens:
                self.lambda_p_hier = float(current_tokens['lambda_p_hier'])
            
            for node, model in self.sub_models.items():
                from .spectrum import create_effects_from_parsed_terms
                new_effects = create_effects_from_parsed_terms(
                    model.parsed_terms_, current_tokens, model.default_alpha_p_
                )
                model.effects_list_ = new_effects
                model.is_grid_search_template_ = False
                
            self.fit(data_train)
            preds = self.predict(data_val)
            
            total_error = 0
            total_weight = 0
            
            for node in self.nodes:
                w = scoring_weights.get(node, 0.0)
                if w == 0: continue
                
                mask = (preds[self.node_col] == node)
                if not mask.any(): continue
                
                target_col = self.sub_models[node].target_col_
                y_hat = preds.loc[mask, f"Estimated{target_col}"].values
                y_true = preds.loc[mask, target_col].values
                
                if len(y_true) > 0:
                    rmse = np.sqrt(np.mean((y_hat - y_true)**2))
                    total_error += w * rmse
                    total_weight += w
                    
            return total_error / (total_weight + 1e-9)

        best_tokens = {t: tokens[t][len(tokens[t])//2] for t in token_names}
        best_score = float('inf')
        
        for cycle in range(3):
            print(f"Cycle {cycle+1}...")
            improved = False
            
            for token in token_names:
                current_best = best_tokens[token]
                possible_values = tokens[token]
                
                for val in possible_values:
                    if val == current_best: continue
                    test_tokens = best_tokens.copy()
                    test_tokens[token] = val
                    
                    try:
                        score = evaluate_config(test_tokens)
                        if score < best_score:
                            best_score = score
                            best_tokens = test_tokens
                            current_best = val
                            improved = True
                            print(f"  New best! {token}={val} -> Score={score:.4f}")
                    except Exception :
                        pass
                        
            if not improved: break
            
        print(f"Optimal Params: {best_tokens}")
        evaluate_config(best_tokens) 
        return self
#: </grid_search_hierarchical>