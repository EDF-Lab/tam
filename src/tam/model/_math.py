# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Core Mathematical Solver Functions.

This module hosts the pure and generic mathematical functions that form the
core of the Time series Additive Model (TAM) solver. These functions are designed to
operate on PyTorch tensors, often in batches (e.g., per-group), handling
linear algebra operations, system solving, and scoring metrics.
"""

from typing import Dict, List, Union, Callable, Optional, Sequence, Tuple
from collections import Counter
import logging
import numpy as np
import torch
from tam.common.utils import TORCH_DEVICE
from tam.common.hardware import hw

# Import abstract classes for type hinting
from .spectrum import BaseEffect, OffsetEffect, LinearEffect, FourierEffect, SplineEffect, CategoricalEffect, ChebyshevEffect, TensorProductEffect

logger = logging.getLogger(__name__)

#: <weighted_cov>
def _compute_weighted_covariances(
    phi: torch.Tensor,
    y_data: torch.Tensor,
    loss_L_star_L: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None
) -> tuple:
    r"""
    Computes the loss-weighted covariance matrices (LHS and RHS of normal equations).

    Calculates:
    - cov_X = Phi.H @ W @ (L*L) @ Phi  (Weighted Feature Covariance)
    - cov_XY = Phi.H @ W @ (L*L) @ Y   (Weighted Feature-Target Covariance)

    loss_L_star_L weights the output channel (shape (d_out, d_out)); sample_weights is an optional
    per-observation weight w (shape (..., n_samples, 1)) applied by a stable sqrt(W) row-scaling of both
    Phi and Y - the atom of any reweighted / GLM fit. sample_weights=None reproduces the plain
    least-squares path bit-for-bit.

    Args:
        phi: The design matrix. Shape: (..., n_samples, n_coeffs).
        y_data: The target tensor. Shape: (..., n_samples, d_out).
        loss_L_star_L: The loss-weighting matrix. Shape: (d_out, d_out).
        sample_weights: Optional per-observation weights. Shape: (..., n_samples, 1). Non-negative.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cov_X, cov_XY)
    """
    # Ensure all tensors are on the same device
    phi = phi.to(TORCH_DEVICE)
    y_data = y_data.to(TORCH_DEVICE)
    loss_L_star_L = loss_L_star_L.to(TORCH_DEVICE)

    # Cast y_data to match phi's dtype
    y_data_aligned = y_data.to(phi.dtype)
    is_unit_loss = loss_L_star_L.shape == (1, 1) and float(loss_L_star_L[0, 0]) == 1.0

    # L2-cache-tiled Gram accumulation avoids full (G, T, D) weighted copies on large series.
    if sample_weights is not None and is_unit_loss and phi.dim() == 3 and phi.shape[1] > 4096:
        n_groups, n_samples, n_coeffs = phi.shape
        cov_X = torch.zeros((n_groups, n_coeffs, n_coeffs), dtype=phi.dtype, device=TORCH_DEVICE)
        cov_XY = torch.zeros((n_groups, n_coeffs, y_data_aligned.shape[-1]), dtype=phi.dtype, device=TORCH_DEVICE)
        w_dev = sample_weights.to(device=TORCH_DEVICE, dtype=phi.dtype)
        for s0 in range(0, n_samples, 4096):
            s1 = min(s0 + 4096, n_samples)
            root_w = w_dev.narrow(-2, s0, s1 - s0).clamp_min(0).sqrt()
            phi_w = phi[:, s0:s1, :] * root_w
            y_w = y_data_aligned[:, s0:s1, :] * root_w
            phi_w_T = phi_w.mT
            cov_X.baddbmm_(phi_w_T, phi_w, beta=1.0, alpha=1.0)
            cov_XY.baddbmm_(phi_w_T, y_w, beta=1.0, alpha=1.0)
        return cov_X, cov_XY

    if sample_weights is not None:
        root_w = sample_weights.to(device=TORCH_DEVICE, dtype=phi.dtype).clamp_min(0).sqrt()
        phi = phi * root_w
        y_data_aligned = y_data_aligned * root_w

    if is_unit_loss:
        cov_XY = phi.mT @ y_data_aligned
        cov_X = phi.mT @ phi
    else:
        y_weighted = y_data_aligned @ loss_L_star_L
        cov_XY = phi.mT @ y_weighted
        if loss_L_star_L.shape[0] == 1:
            L_sqrt = loss_L_star_L[0, 0].sqrt()
            phi_weighted = phi * L_sqrt
            cov_X = phi_weighted.mT @ phi_weighted
        else:
            cov_X = phi.mT @ phi

    return cov_X, cov_XY


#: </weighted_cov>
def _is_rowwise_stateless_effects(effects_list: List[BaseEffect]) -> bool:
    r"""Returns True if every effect evaluates rows independently with no cross-row state or block orthogonalization."""
    from .spectrum import (
        RBFEffect,
        WaveletEffect,
        NeuralEffect,
        PriorSpectralEffect,
        DegreeDayEffect,
        WarpedCalendarEffect,
        KANEffect,
    )
    from .spectrum._foundation import StructuralPriorEffect

    if any(getattr(e, "_projection", None) is not None for e in effects_list):
        return False
    plain_stateless = (
        OffsetEffect,
        LinearEffect,
        FourierEffect,
        ChebyshevEffect,
        DegreeDayEffect,
        WarpedCalendarEffect,
        StructuralPriorEffect,
    )
    for effect in effects_list:
        if isinstance(effect, TensorProductEffect):
            if not _is_rowwise_stateless_effects(effect.effects):
                return False
        elif isinstance(effect, SplineEffect):
            if getattr(effect, "_cached_knots", None) is None:
                return False
        elif isinstance(effect, CategoricalEffect):
            if effect.topology == "graph" and getattr(effect, "_laplacian", None) is None:
                return False
            if effect.unseen == "mean" and getattr(effect, "_level_weights", None) is None:
                return False
        elif isinstance(effect, RBFEffect):
            if getattr(effect, "centers", None) is None:
                return False
        elif isinstance(effect, WaveletEffect):
            if getattr(effect, "locations", None) is None:
                return False
        elif isinstance(effect, NeuralEffect):
            if getattr(effect, "weights_list", None) is None:
                return False
        elif isinstance(effect, PriorSpectralEffect):
            if (
                isinstance(effect.prior, str)
                and effect.prior.lower() == "empirical"
                and not getattr(effect, "_is_initialized", False)
            ):
                return False
        elif isinstance(effect, KANEffect):
            if (
                getattr(effect, "_x_mean", None) is None
                or getattr(effect, "_u_mean", None) is None
                or getattr(effect, "knots_in", None) is None
            ):
                return False
        elif not isinstance(effect, plain_stateless):
            return False
    return True


def _streamed_weighted_covariances(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    effects_list: List[BaseEffect],
    loss_L_star_L: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    feature_columns: Optional[List[str]] = None,
    tile_size: int = 4096,
    dtype: Optional[torch.dtype] = None,
    return_y_sq: bool = False,
) -> tuple:
    r"""Flash-style tiled basis evaluation + L2-resident weighted Gram accumulation without materializing full Phi."""
    from .spectrum._factory import build_phi_from_effects

    target_dtype = torch.get_default_dtype() if dtype is None else dtype
    n_groups, n_samples = int(x_data.shape[0]), int(x_data.shape[1])
    total_d = sum(e.get_n_coeffs() for e in effects_list)
    d_out = int(y_data.shape[-1])
    loss_scalar = float(loss_L_star_L[0, 0])

    cov_X = torch.zeros((n_groups, total_d, total_d), dtype=target_dtype, device=TORCH_DEVICE)
    cov_XY = torch.zeros((n_groups, total_d, d_out), dtype=target_dtype, device=TORCH_DEVICE)
    y_sq = torch.zeros((n_groups,), dtype=target_dtype, device=TORCH_DEVICE) if return_y_sq else None

    for s0 in range(0, n_samples, tile_size):
        s1 = min(s0 + tile_size, n_samples)
        x_tile = x_data[:, s0:s1, :].to(TORCH_DEVICE)
        y_tile = y_data[:, s0:s1, :].to(device=TORCH_DEVICE, dtype=target_dtype)
        phi_tile = build_phi_from_effects(x_tile, effects_list, feature_columns=feature_columns).to(
            device=TORCH_DEVICE, dtype=target_dtype
        )
        if sample_weights is not None:
            root_w = sample_weights[:, s0:s1, :].to(device=TORCH_DEVICE, dtype=target_dtype).clamp_min(0).sqrt()
            phi_tile = phi_tile * root_w
            y_tile = y_tile * root_w
        phi_T = phi_tile.mT
        cov_X.baddbmm_(phi_T, phi_tile, beta=1.0, alpha=loss_scalar)
        cov_XY.baddbmm_(phi_T, y_tile, beta=1.0, alpha=loss_scalar)
        if return_y_sq:
            y_sq.add_((y_tile * y_tile).sum(dim=(1, 2)) * loss_scalar)

    if return_y_sq:
        return cov_X, cov_XY, y_sq
    return cov_X, cov_XY


def _merge_contiguous_spans(items: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Merges adjacent ``(orig_c0, orig_c1, local_c0, local_c1)`` slices into minimal contiguous copies."""
    if not items:
        return []
    merged = [items[0]]
    for oc0, oc1, lc0, lc1 in items[1:]:
        p_oc0, p_oc1, p_lc0, p_lc1 = merged[-1]
        if oc0 == p_oc1 and lc0 == p_lc1:
            merged[-1] = (p_oc0, oc1, p_lc0, lc1)
        else:
            merged.append((oc0, oc1, lc0, lc1))
    return merged


def _extract_onehot_index(
    effect: CategoricalEffect,
    x_eff: torch.Tensor,
    target_device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extracts 0-based category indices and optional in-range mask for a plain one-hot ``CategoricalEffect``."""
    x_cat = x_eff[..., 0] if (effect.topology == "graph" and x_eff.shape[-1] == len(effect.input_features)) else x_eff
    n_cat = effect.n_categories
    if n_cat == 1:
        return torch.zeros(x_cat.shape, dtype=torch.long, device=target_device), None
    raw_idx = torch.round((x_cat + 1.0) / 2.0 * (n_cat - 1)).long()
    idx = torch.clamp(raw_idx, 0, n_cat - 1).to(target_device)
    if effect.unseen == "edge" or effect.extrapolate == "constant":
        return idx, None
    valid_bool = (raw_idx >= 0) & (raw_idx < n_cat)
    if bool(valid_bool.all()):
        return idx, None
    return idx, valid_bool.to(target_device)


#: <structured_design>
class _StructuredDesign:
    r"""Exact block-sparse design representation separating dense, one-hot categorical, and ``te(smooth, cat)`` blocks."""

    def __init__(
        self,
        total_d: int,
        n_groups: int,
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        phi_dense: Optional[torch.Tensor],
        dense_spans: List[Tuple[int, int, int, int]],
        onehot_blocks: List[Tuple[int, int, int, torch.Tensor, Optional[torch.Tensor]]],
        te_blocks: List[Dict[str, object]],
        te_pairs: List[Tuple[int, int, torch.Tensor, List[Tuple[int, int, int, int]]]],
    ):
        self.total_d = total_d
        self.n_groups = n_groups
        self.n_samples = n_samples
        self.device = device
        self.dtype = dtype
        self.phi_dense = phi_dense
        self.dense_spans = dense_spans
        self.onehot_blocks = onehot_blocks
        self.te_blocks = te_blocks
        self.te_pairs = te_pairs

    @classmethod
    def try_build(
        cls,
        x_data: torch.Tensor,
        effects_list: List[BaseEffect],
        feature_columns: Optional[List[str]] = None,
    ) -> Optional["_StructuredDesign"]:
        """Constructs an exact structured design when >=50% of columns are categorical / ``te(smooth, cat)``."""
        from .spectrum._factory import _infer_feature_columns

        if x_data.dim() != 3 or x_data.shape[0] == 0 or x_data.shape[1] == 0:
            return None
        g_dim, t_dim = int(x_data.shape[0]), int(x_data.shape[1])
        total_d = sum(e.get_n_coeffs() for e in effects_list)
        if total_d < 40:
            return None
        if any(getattr(e, "_projection", None) is not None or getattr(e, "orthogonalize", False) for e in effects_list):
            return None
        if any(getattr(e, "stateful", False) for e in effects_list):
            return None

        if feature_columns is None:
            feature_columns = _infer_feature_columns(effects_list)
        name_to_idx = {name: i for i, name in enumerate(feature_columns)} if feature_columns else None
        base_dtype = torch.get_default_dtype()
        dev = TORCH_DEVICE
        margin_cache: Dict = {}

        dense_parts: List[torch.Tensor] = []
        raw_dense_spans: List[Tuple[int, int, int, int]] = []
        onehot_blocks: List[Tuple[int, int, int, torch.Tensor, Optional[torch.Tensor]]] = []
        te_blocks: List[Dict[str, object]] = []

        d_dense = 0
        c_pos = 0
        col_idx = 0

        for effect in effects_list:
            n_c = effect.get_n_coeffs()
            c0, c1 = c_pos, c_pos + n_c
            c_pos = c1

            if isinstance(effect, OffsetEffect):
                part = effect.transform(x_data).to(device=dev, dtype=base_dtype)
                dense_parts.append(part)
                raw_dense_spans.append((c0, c1, d_dense, d_dense + n_c))
                d_dense += n_c
                continue

            x_eff, cols_t, col_idx = _slice_effect_input_with_cols(x_data, effect, name_to_idx, col_idx)

            if _is_plain_onehot_cat(effect, allow_zero=True) and x_eff.dim() == 3 - int(not effect.multi_input):
                x_cat_check = x_eff[..., 0] if (effect.topology == "graph" and x_eff.shape[-1] == len(effect.input_features)) else x_eff
                if x_cat_check.dim() == 2:
                    idx, valid = _extract_onehot_index(effect, x_eff, dev)
                    onehot_blocks.append((c0, c1, n_c, idx, valid))
                    continue

            if (
                isinstance(effect, TensorProductEffect)
                and len(effect.effects) == 2
                and x_eff.dim() == 3
                and not any(getattr(m, "_projection", None) is not None for m in effect.effects)
            ):
                e0, e1 = effect.effects
                if (
                    _is_plain_onehot_cat(e0, allow_zero=True)
                    and _is_plain_onehot_cat(e1, allow_zero=True)
                    and (
                        effect.extrapolate == "continue"
                        or (effect.extrapolate == "constant" and e0.unseen == "edge" and e1.unseen == "edge")
                    )
                ):
                    n0_cols = len(e0.required_features)
                    idx0, val0 = _extract_onehot_index(e0, x_eff[..., :n0_cols].squeeze(-1) if n0_cols == 1 else x_eff[..., :n0_cols], dev)
                    idx1, val1 = _extract_onehot_index(e1, x_eff[..., n0_cols:].squeeze(-1) if len(e1.required_features) == 1 else x_eff[..., n0_cols:], dev)
                    joint_idx = idx0 * e1.n_categories + idx1
                    if effect.extrapolate == "constant":
                        joint_val = None
                    elif val0 is None:
                        joint_val = val1
                    elif val1 is None:
                        joint_val = val0
                    else:
                        joint_val = val0 & val1
                    onehot_blocks.append((c0, c1, n_c, joint_idx, joint_val))
                    continue

                if effect.extrapolate == "continue":
                    cat_side = -1
                    if _is_plain_onehot_cat(e1, allow_zero=True) and 2 <= e1.n_categories <= 32 and _is_rowwise_stateless_effects([e0]):
                        cat_side = 1
                    elif _is_plain_onehot_cat(e0, allow_zero=True) and 2 <= e0.n_categories <= 32 and _is_rowwise_stateless_effects([e1]):
                        cat_side = 0
                    if cat_side != -1:
                        n0_cols = len(e0.required_features)
                        if cat_side == 1:
                            other_eff, cat_eff = e0, e1
                            x_other, x_cat_in = x_eff[..., :n0_cols], x_eff[..., n0_cols:]
                            other_cols = cols_t[:n0_cols]
                        else:
                            cat_eff, other_eff = e0, e1
                            x_cat_in, x_other = x_eff[..., :n0_cols], x_eff[..., n0_cols:]
                            other_cols = cols_t[n0_cols:]
                        cat_sq = x_cat_in[..., 0] if (cat_eff.topology == "graph" or len(cat_eff.required_features) == 1) else x_cat_in.squeeze(-1)
                        idx_te, val_te = _extract_onehot_index(cat_eff, cat_sq, dev)
                        shared_idx = (g_dim == 1) or (
                            bool(torch.equal(idx_te, idx_te[0:1].expand_as(idx_te)))
                            and (val_te is None or bool(torch.equal(val_te, val_te[0:1].expand_as(val_te))))
                        )
                        if shared_idx:
                            if len(other_eff.required_features) == 1 and not other_eff.keeps_feature_dim:
                                x_other = x_other.squeeze(-1)
                            b_smooth = _eval_effect_with_margin_cache(other_eff, x_other, other_cols, margin_cache).to(
                                device=dev, dtype=base_dtype
                            )
                            if b_smooth.dim() == 3 and b_smooth.shape[-1] > 0:
                                if val_te is not None:
                                    b_smooth = b_smooth * val_te.unsqueeze(-1).to(base_dtype)
                                b_shared = (g_dim == 1) or bool(torch.equal(b_smooth, b_smooth[0:1].expand_as(b_smooth)))
                                if b_shared:
                                    b_smooth = b_smooth[0:1].contiguous()
                                n_cat = cat_eff.n_categories
                                d_sm = int(b_smooth.shape[-1])
                                idx_1d = idx_te[0]
                                perm = torch.argsort(idx_1d, stable=True)
                                cnts = torch.bincount(idx_1d, minlength=n_cat).tolist()
                                cuts = [0]
                                for cnt in cnts:
                                    cuts.append(cuts[-1] + cnt)
                                te_blocks.append(
                                    {
                                        "c0": c0,
                                        "c1": c1,
                                        "d_smooth": d_sm,
                                        "n_cat": n_cat,
                                        "cat_side": cat_side,
                                        "b_shared": b_shared,
                                        "B": b_smooth,
                                        "B_sorted": b_smooth[:, perm, :].contiguous(),
                                        "idx": idx_te[0:1],
                                        "perm": perm,
                                        "cuts": cuts,
                                    }
                                )
                                continue

            part = _eval_effect_with_margin_cache(effect, x_eff, cols_t, margin_cache).to(device=dev, dtype=base_dtype)
            dense_parts.append(part)
            raw_dense_spans.append((c0, c1, d_dense, d_dense + n_c))
            d_dense += n_c

        d_sparse = total_d - d_dense
        if d_sparse < max(24, total_d // 2):
            return None

        phi_dense = torch.cat(dense_parts, dim=-1).contiguous() if dense_parts else None
        dense_spans = _merge_contiguous_spans(raw_dense_spans)

        te_pairs: List[Tuple[int, int, torch.Tensor, List[Tuple[int, int, int, int]]]] = []
        for a in range(len(te_blocks)):
            idx_a = te_blocks[a]["idx"][0]
            ka_n = int(te_blocks[a]["n_cat"])
            for b in range(a + 1, len(te_blocks)):
                idx_b = te_blocks[b]["idx"][0]
                kb_n = int(te_blocks[b]["n_cat"])
                pair = idx_a * kb_n + idx_b
                perm_ab = torch.argsort(pair, stable=True)
                cnts = torch.bincount(pair, minlength=ka_n * kb_n).tolist()
                buckets: List[Tuple[int, int, int, int]] = []
                pos = 0
                for flat_k, cnt in enumerate(cnts):
                    if cnt > 0:
                        ka, kb = divmod(flat_k, kb_n)
                        buckets.append((ka, kb, pos, pos + cnt))
                        pos += cnt
                te_pairs.append((a, b, perm_ab, buckets))

        return cls(
            total_d=total_d,
            n_groups=g_dim,
            n_samples=t_dim,
            device=dev,
            dtype=base_dtype,
            phi_dense=phi_dense,
            dense_spans=dense_spans,
            onehot_blocks=onehot_blocks,
            te_blocks=te_blocks,
            te_pairs=te_pairs,
        )

    def _place_dense_cols(self, mat_target: torch.Tensor, row0: int, row1: int, src_right: torch.Tensor) -> None:
        """Copies ``src_right`` (G, R, d_dense) into ``mat_target[:, row0:row1, :]`` and its symmetric transpose."""
        for oc0, oc1, lc0, lc1 in self.dense_spans:
            blk = src_right[:, :, lc0:lc1]
            mat_target[:, row0:row1, oc0:oc1] = blk
            mat_target[:, oc0:oc1, row0:row1] = blk.mT

    def linear_predictor(self, theta: torch.Tensor) -> torch.Tensor:
        """Computes exact ``Phi @ theta`` without materializing sparse columns."""
        theta_dev = theta.to(device=self.device, dtype=self.dtype)
        g_dim, t_dim = self.n_groups, self.n_samples
        d_out = int(theta_dev.shape[-1])

        if self.phi_dense is not None:
            theta_d = torch.cat([theta_dev[..., oc0:oc1, :] for oc0, oc1, _, _ in self.dense_spans], dim=-2)
            eta = self.phi_dense @ theta_d
        else:
            eta = torch.zeros((g_dim, t_dim, d_out), dtype=self.dtype, device=self.device)

        for c0, c1, _, idx_h, valid_h in self.onehot_blocks:
            c_h = theta_dev[..., c0:c1, :]
            contrib = c_h.gather(1, idx_h.unsqueeze(-1).expand(g_dim, t_dim, d_out))
            if valid_h is not None:
                contrib = contrib * valid_h.unsqueeze(-1).to(self.dtype)
            eta.add_(contrib)

        for te in self.te_blocks:
            c0, c1 = int(te["c0"]), int(te["c1"])
            d_sm, n_cat, cat_side = int(te["d_smooth"]), int(te["n_cat"]), int(te["cat_side"])
            b_sm = te["B"]
            idx_m = te["idx"].expand(g_dim, t_dim)
            c_m = theta_dev[..., c0:c1, :]
            if cat_side == 1:
                beta_4 = c_m.view(g_dim, d_sm, n_cat, d_out)
            else:
                beta_4 = c_m.view(g_dim, n_cat, d_sm, d_out).transpose(1, 2)
            if n_cat <= 16:
                if bool(te["b_shared"]):
                    beta_mat = beta_4.permute(1, 0, 2, 3).reshape(d_sm, g_dim * n_cat * d_out)
                    scores = (b_sm[0] @ beta_mat).view(t_dim, g_dim, n_cat, d_out).transpose(0, 1)
                else:
                    scores = torch.bmm(b_sm, beta_4.reshape(g_dim, d_sm, n_cat * d_out)).view(g_dim, t_dim, n_cat, d_out)
                eta.add_(scores.gather(2, idx_m.unsqueeze(-1).unsqueeze(-1).expand(g_dim, t_dim, 1, d_out)).squeeze(2))
            else:
                flat_beta = beta_4.transpose(1, 2).reshape(g_dim, n_cat, d_sm * d_out)
                beta_sel = flat_beta.gather(1, idx_m.unsqueeze(-1).expand(g_dim, t_dim, d_sm * d_out)).view(g_dim, t_dim, d_sm, d_out)
                b_eval = b_sm.expand(g_dim, t_dim, d_sm) if bool(te["b_shared"]) else b_sm
                eta.add_(torch.sum(b_eval.unsqueeze(-1) * beta_sel, dim=-2))

        return eta

    def weighted_covariances(
        self,
        y_data: torch.Tensor,
        loss_L_star_L: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        return_y_sq: bool = False,
    ) -> tuple:
        """Computes exact ``(cov_X, cov_XY[, y_sq])`` using block-sparse one-hot and ``te(smooth, cat)`` reductions."""
        work_dtype = self.dtype if dtype is None else dtype
        g_dim, t_dim, d_tot = self.n_groups, self.n_samples, self.total_d
        y_w = y_data.to(device=self.device, dtype=work_dtype)
        d_out = int(y_w.shape[-1])
        loss_scalar = float(loss_L_star_L.to(device="cpu").item())

        if sample_weights is not None:
            w_eff = sample_weights.to(device=self.device, dtype=work_dtype).squeeze(-1).clamp_min(0)
            if loss_scalar != 1.0:
                w_eff = w_eff * loss_scalar
        else:
            w_eff = torch.full((g_dim, t_dim), loss_scalar, dtype=work_dtype, device=self.device)

        rw = w_eff.sqrt().unsqueeze(-1)
        wy = w_eff.unsqueeze(-1) * y_w
        y_rw = y_w * rw

        gx = torch.zeros((g_dim, d_tot, d_tot), dtype=work_dtype, device=self.device)
        gxy = torch.zeros((g_dim, d_tot, d_out), dtype=work_dtype, device=self.device)
        y_sq = (wy * y_w).sum(dim=(1, 2)) if return_y_sq else None

        phi_d = None
        phi_d_w_base = None
        if self.phi_dense is not None:
            phi_d = self.phi_dense if self.phi_dense.dtype == work_dtype else self.phi_dense.to(work_dtype)
            d_dense = int(phi_d.shape[-1])
            phi_d_rw = phi_d * rw
            phi_d_w_base = phi_d * w_eff.unsqueeze(-1)
            g_dd = phi_d_rw.mT @ phi_d_rw
            r_d = phi_d_rw.mT @ y_rw
            for oc0, oc1, lc0, lc1 in self.dense_spans:
                gxy[:, oc0:oc1, :] = r_d[:, lc0:lc1, :]
                for oc0b, oc1b, lc0b, lc1b in self.dense_spans:
                    gx[:, oc0:oc1, oc0b:oc1b] = g_dd[:, lc0:lc1, lc0b:lc1b]
        else:
            d_dense = 0

        # One-hot blocks: diagonal, x y, x dense, x onehot, x te
        oh_weights: List[torch.Tensor] = []
        for c0, c1, k_h, idx_h, val_h in self.onehot_blocks:
            w_h = w_eff if val_h is None else w_eff * val_h.to(work_dtype)
            oh_weights.append(w_h)

            d_diag = torch.zeros((g_dim, k_h), dtype=work_dtype, device=self.device).scatter_add_(1, idx_h, w_h)
            gx[:, c0:c1, c0:c1].diagonal(dim1=-2, dim2=-1).copy_(d_diag)

            wy_h = wy if val_h is None else wy * val_h.unsqueeze(-1).to(work_dtype)
            gxy[:, c0:c1, :] = torch.zeros((g_dim, k_h, d_out), dtype=work_dtype, device=self.device).scatter_add_(
                1, idx_h.unsqueeze(-1).expand(g_dim, t_dim, d_out), wy_h
            )

            if d_dense > 0:
                pw_h = phi_d_w_base if val_h is None else (phi_d * w_h.unsqueeze(-1))
                c_hd = torch.zeros((g_dim, k_h, d_dense), dtype=work_dtype, device=self.device).scatter_add_(
                    1, idx_h.unsqueeze(-1).expand(g_dim, t_dim, d_dense), pw_h
                )
                self._place_dense_cols(gx, c0, c1, c_hd)

        for i in range(len(self.onehot_blocks)):
            c0_i, c1_i, k_i, idx_i, val_i = self.onehot_blocks[i]
            for j in range(i + 1, len(self.onehot_blocks)):
                c0_j, c1_j, k_j, idx_j, val_j = self.onehot_blocks[j]
                if val_j is None:
                    w_ij = oh_weights[i]
                elif val_i is None:
                    w_ij = oh_weights[j]
                else:
                    w_ij = oh_weights[i] * val_j.to(work_dtype)
                c_ij = torch.zeros((g_dim, k_i * k_j), dtype=work_dtype, device=self.device).scatter_add_(
                    1, idx_i * k_j + idx_j, w_ij
                ).view(g_dim, k_i, k_j)
                gx[:, c0_i:c1_i, c0_j:c1_j] = c_ij
                gx[:, c0_j:c1_j, c0_i:c1_i] = c_ij.mT

        # te(smooth, cat) blocks: diagonal, x y, x dense, x onehot
        te_cast: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for te in self.te_blocks:
            c0, c1 = int(te["c0"]), int(te["c1"])
            d_sm, k_m, cat_side = int(te["d_smooth"]), int(te["n_cat"]), int(te["cat_side"])
            b_m = te["B"]
            b_ms = te["B_sorted"]
            if b_m.dtype != work_dtype:
                b_m = b_m.to(work_dtype)
                b_ms = b_ms.to(work_dtype)
            te_cast.append((b_m, b_ms))

            idx_m = te["idx"].expand(g_dim, t_dim)
            perm = te["perm"]
            cuts = te["cuts"]

            # onehot x te
            b_m_full = b_m.expand(g_dim, t_dim, d_sm) if (bool(te["b_shared"]) and g_dim > 1) else b_m
            for (c0_h, c1_h, k_h, idx_h, _), w_h in zip(self.onehot_blocks, oh_weights):
                wb = b_m_full * w_h.unsqueeze(-1)
                agg = torch.zeros((g_dim, k_h * k_m, d_sm), dtype=work_dtype, device=self.device).scatter_add_(
                    1, (idx_h * k_m + idx_m).unsqueeze(-1).expand(g_dim, t_dim, d_sm), wb
                ).view(g_dim, k_h, k_m, d_sm)
                blk_hm = agg.transpose(2, 3).reshape(g_dim, k_h, d_sm * k_m) if cat_side == 1 else agg.reshape(g_dim, k_h, k_m * d_sm)
                gx[:, c0_h:c1_h, c0:c1] = blk_hm
                gx[:, c0:c1, c0_h:c1_h] = blk_hm.mT

            w_s = w_eff[:, perm].unsqueeze(-1)
            wy_s = wy[:, perm, :]
            phi_d_s = phi_d[:, perm, :] if d_dense > 0 else None

            if cat_side == 1:
                te_diag = torch.zeros((g_dim, d_sm, k_m, d_sm, k_m), dtype=work_dtype, device=self.device)
                te_y = torch.zeros((g_dim, d_sm, k_m, d_out), dtype=work_dtype, device=self.device)
                te_d = torch.zeros((g_dim, d_sm, k_m, d_dense), dtype=work_dtype, device=self.device) if d_dense > 0 else None
            else:
                te_diag = torch.zeros((g_dim, k_m, d_sm, k_m, d_sm), dtype=work_dtype, device=self.device)
                te_y = torch.zeros((g_dim, k_m, d_sm, d_out), dtype=work_dtype, device=self.device)
                te_d = torch.zeros((g_dim, k_m, d_sm, d_dense), dtype=work_dtype, device=self.device) if d_dense > 0 else None

            for k in range(k_m):
                s, e = cuts[k], cuts[k + 1]
                if s == e:
                    continue
                bk = b_ms[:, s:e, :]
                if bk.shape[0] == 1 and g_dim > 1:
                    bk_exp = bk.expand(g_dim, e - s, d_sm)
                else:
                    bk_exp = bk
                bkw = bk * w_s[:, s:e, :]
                blk_kk = bk_exp.mT @ bkw
                blk_ky = bk_exp.mT @ wy_s[:, s:e, :]
                if cat_side == 1:
                    te_diag[:, :, k, :, k] = blk_kk
                    te_y[:, :, k, :] = blk_ky
                    if d_dense > 0:
                        te_d[:, :, k, :] = bkw.mT @ phi_d_s[:, s:e, :]
                else:
                    te_diag[:, k, :, k, :] = blk_kk
                    te_y[:, k, :, :] = blk_ky
                    if d_dense > 0:
                        te_d[:, k, :, :] = bkw.mT @ phi_d_s[:, s:e, :]

            gx[:, c0:c1, c0:c1] = te_diag.reshape(g_dim, d_sm * k_m, d_sm * k_m)
            gxy[:, c0:c1, :] = te_y.reshape(g_dim, d_sm * k_m, d_out)
            if d_dense > 0:
                self._place_dense_cols(gx, c0, c1, te_d.reshape(g_dim, d_sm * k_m, d_dense))

        # Off-diagonal te_a x te_b blocks
        for a, b, perm_ab, buckets in self.te_pairs:
            te_a, te_b = self.te_blocks[a], self.te_blocks[b]
            ca0, ca1 = int(te_a["c0"]), int(te_a["c1"])
            cb0, cb1 = int(te_b["c0"]), int(te_b["c1"])
            da, ka_n, side_a = int(te_a["d_smooth"]), int(te_a["n_cat"]), int(te_a["cat_side"])
            db, kb_n, side_b = int(te_b["d_smooth"]), int(te_b["n_cat"]), int(te_b["cat_side"])
            ba_s = te_cast[a][0][:, perm_ab, :]
            bb_s = te_cast[b][0][:, perm_ab, :] * w_eff[:, perm_ab].unsqueeze(-1)
            cross = torch.zeros((g_dim, ka_n, da, kb_n, db), dtype=work_dtype, device=self.device)
            for ka, kb, s, e in buckets:
                bak = ba_s[:, s:e, :]
                if bak.shape[0] == 1 and g_dim > 1:
                    bak = bak.expand(g_dim, e - s, da)
                cross[:, ka, :, kb, :] = torch.bmm(bak.mT, bb_s[:, s:e, :])
            if side_a == 1:
                cross = cross.transpose(1, 2)
            if side_b == 1:
                cross = cross.transpose(3, 4)
            blk_ab = cross.reshape(g_dim, da * ka_n, db * kb_n)
            gx[:, ca0:ca1, cb0:cb1] = blk_ab
            gx[:, cb0:cb1, ca0:ca1] = blk_ab.mT

        return (gx, gxy, y_sq) if return_y_sq else (gx, gxy)


#: </structured_design>
#: <group_split>
def _try_group_split_covariances(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    effects_list: List[BaseEffect],
    loss_L_star_L: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    feature_columns: Optional[List[str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Evaluates group-invariant effects once on ``x_data[0:1]`` instead of ``G`` times when ``G >= 2``."""
    from .spectrum._factory import _infer_feature_columns

    if x_data.dim() != 3 or x_data.shape[0] < 2 or x_data.shape[1] < 1:
        return None
    if loss_L_star_L.shape != (1, 1):
        return None
    if any(getattr(e, "_projection", None) is not None for e in effects_list):
        return None

    target_dtype = torch.get_default_dtype() if dtype is None else dtype
    if sample_weights is not None:
        w_dev = sample_weights.to(device=TORCH_DEVICE, dtype=target_dtype).clamp_min(0)
        if not bool(torch.equal(w_dev, w_dev[0:1].expand_as(w_dev))):
            return None
    else:
        w_dev = None

    if feature_columns is None:
        feature_columns = _infer_feature_columns(effects_list)
    name_to_idx = {name: i for i, name in enumerate(feature_columns)} if feature_columns else None

    g_dim, t_dim = int(x_data.shape[0]), int(x_data.shape[1])
    shared_items = []
    varying_items = []
    raw_s_spans: List[Tuple[int, int, int, int]] = []
    raw_v_spans: List[Tuple[int, int, int, int]] = []
    col_shared: Dict[Tuple[int, ...], bool] = {}

    d_s = 0
    d_v = 0
    c_pos = 0
    col_idx = 0

    for effect in effects_list:
        n_c = effect.get_n_coeffs()
        c0, c1 = c_pos, c_pos + n_c
        c_pos = c1

        if isinstance(effect, OffsetEffect):
            shared_items.append((effect, None, None))
            raw_s_spans.append((c0, c1, d_s, d_s + n_c))
            d_s += n_c
            continue

        x_eff, cols_t, col_idx = _slice_effect_input_with_cols(x_data, effect, name_to_idx, col_idx)
        is_shared = False
        if _is_rowwise_stateless_effects([effect]):
            s_flag = col_shared.get(cols_t)
            if s_flag is None:
                s_flag = bool(torch.equal(x_eff, x_eff[0:1].expand_as(x_eff)))
                col_shared[cols_t] = s_flag
            is_shared = s_flag
        if is_shared:
            shared_items.append((effect, x_eff[0:1], cols_t))
            raw_s_spans.append((c0, c1, d_s, d_s + n_c))
            d_s += n_c
        else:
            varying_items.append((effect, x_eff, cols_t))
            raw_v_spans.append((c0, c1, d_v, d_v + n_c))
            d_v += n_c

    d_tot = d_s + d_v
    if d_s < max(16, d_tot // 4):
        return None

    x0_dummy = x_data[0:1]
    margin_cache_s: Dict = {}
    s_parts = []
    for effect, x_eff_0, cols_t in shared_items:
        if isinstance(effect, OffsetEffect):
            s_parts.append(effect.transform(x0_dummy).to(device=TORCH_DEVICE, dtype=target_dtype))
        else:
            s_parts.append(_eval_effect_with_margin_cache(effect, x_eff_0, cols_t, margin_cache_s).to(device=TORCH_DEVICE, dtype=target_dtype))
    phi_s = torch.cat(s_parts, dim=-1).squeeze(0)  # (T, d_s)

    if d_v > 0:
        margin_cache_v: Dict = {}
        v_parts = [
            _eval_effect_with_margin_cache(effect, x_eff, cols_t, margin_cache_v).to(device=TORCH_DEVICE, dtype=target_dtype)
            for effect, x_eff, cols_t in varying_items
        ]
        phi_v = torch.cat(v_parts, dim=-1)  # (G, T, d_v)
    else:
        phi_v = None

    y_scaled = y_data.to(device=TORCH_DEVICE, dtype=target_dtype)
    if w_dev is not None:
        rw_0 = w_dev[0].sqrt()
        rw_g = w_dev.sqrt()
        phi_s = phi_s * rw_0
        if phi_v is not None:
            phi_v = phi_v * rw_g
        y_scaled = y_scaled * rw_g

    loss_val = float(loss_L_star_L.to(device="cpu").item())
    if loss_val != 1.0:
        l_sqrt = loss_L_star_L.to(device=TORCH_DEVICE, dtype=target_dtype)[0, 0].sqrt()
        phi_s = phi_s * l_sqrt
        if phi_v is not None:
            phi_v = phi_v * l_sqrt
        y_scaled = y_scaled * l_sqrt

    d_out = int(y_scaled.shape[-1])
    g_ss = phi_s.T @ phi_s
    y_T = y_scaled.permute(1, 0, 2).reshape(t_dim, g_dim * d_out)
    r_s = (phi_s.T @ y_T).view(d_s, g_dim, d_out).transpose(0, 1)

    cov_x = torch.empty((g_dim, d_tot, d_tot), dtype=target_dtype, device=TORCH_DEVICE)
    cov_xy = torch.empty((g_dim, d_tot, d_out), dtype=target_dtype, device=TORCH_DEVICE)

    s_spans = _merge_contiguous_spans(raw_s_spans)
    v_spans = _merge_contiguous_spans(raw_v_spans)

    if phi_v is not None:
        g_vv = phi_v.mT @ phi_v
        phi_v_T = phi_v.permute(1, 0, 2).reshape(t_dim, g_dim * d_v)
        g_sv = (phi_s.T @ phi_v_T).view(d_s, g_dim, d_v).transpose(0, 1)
        r_v = phi_v.mT @ y_scaled
        for ov0, ov1, lv0, lv1 in v_spans:
            cov_xy[:, ov0:ov1, :] = r_v[:, lv0:lv1, :]
            for ov0b, ov1b, lv0b, lv1b in v_spans:
                cov_x[:, ov0:ov1, ov0b:ov1b] = g_vv[:, lv0:lv1, lv0b:lv1b]
        for os0, os1, ls0, ls1 in s_spans:
            for ov0, ov1, lv0, lv1 in v_spans:
                blk = g_sv[:, ls0:ls1, lv0:lv1]
                cov_x[:, os0:os1, ov0:ov1] = blk
                cov_x[:, ov0:ov1, os0:os1] = blk.mT

    for os0, os1, ls0, ls1 in s_spans:
        cov_xy[:, os0:os1, :] = r_s[:, ls0:ls1, :]
        for os0b, os1b, ls0b, ls1b in s_spans:
            cov_x[:, os0:os1, os0b:os1b] = g_ss[ls0:ls1, ls0b:ls1b]

    return cov_x, cov_xy
#: </group_split>

#: <solve_system>
def _ridge_floor(
    n_samples: Union[int, float],
    size: int,
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> torch.Tensor:
    r"""
    The ridge added to every regularised system for numerical stability: ``1e-6 * n * I``.

    The covariances are sums over the n samples, so the effective system is
    ``(Phi'Phi / n) + P + 1e-6 * I``: a penalty below about 1e-6 (``ap < -6``) is dominated by this floor.
    The solver and the GCV score use this one helper, so the penalty GCV selects gives the same
    coefficients when the model is refitted with it.
    """
    return (1e-6 * n_samples) * torch.eye(size, device=device, dtype=dtype)


def solve_linear_system(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    penalty_M_star_M: torch.Tensor,
    n_samples: Union[int, float]
) -> torch.Tensor:
    r"""
    Solves the regularized linear system (batched).
    
    Solves for Coefficients (Beta):
    (cov_X + n_samples * P) @ Beta = cov_XY

    Args:
        cov_X: The weighted feature covariance matrix. Shape: (..., K, K).
        cov_XY: The weighted feature-target covariance. Shape: (..., K, d_out).
        penalty_M_star_M: The penalty matrix P. Shape: (K, K).
        n_samples: Scaling factor for the penalty term (usually sample count).

    Returns:
        torch.Tensor: Fitted coefficients. Shape: (..., K, d_out).
    """
    cov_X = cov_X.to(TORCH_DEVICE)
    cov_XY = cov_XY.to(TORCH_DEVICE)
    penalty_M_star_M = penalty_M_star_M.to(TORCH_DEVICE)

    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()

    # Expand penalty matrix dimensions to match batch size of cov_X
    dims_to_add = cov_X.dim() - penalty_M_star_M.dim()
    regularization_term = (n_samples * penalty_M_star_M).view(
        *([1] * dims_to_add), *penalty_M_star_M.shape
    )
    
    # LHS construction
    matrix_to_invert = cov_X + regularization_term
    
    # Ridge floor for numerical stability (shared with compute_gcv_score)
    jitter = _ridge_floor(n_samples, matrix_to_invert.shape[-1], matrix_to_invert.dtype, matrix_to_invert.device)
    matrix_to_invert += jitter.view(*([1] * dims_to_add), *jitter.shape)

    # Solve large symmetric positive definite normal systems via Cholesky when real and SPD
    if matrix_to_invert.shape[-1] >= 192 and not matrix_to_invert.is_complex() and hw.supports_linalg:
        try:
            l_chol, info = torch.linalg.cholesky_ex(matrix_to_invert)
            if info.numel() == 0 or int(info.max().item()) == 0:
                return torch.cholesky_solve(cov_XY, l_chol)
        except (RuntimeError, NotImplementedError):
            pass

    # Fallback solve: A^-1 @ B
    return hw.safe_solve(matrix_to_invert, cov_XY)
#: </solve_system>


def _predict_from_coeffs(
    phi_matrix: torch.Tensor,
    adaptive_coeffs: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes predictions using the linear model equation.
    
    Y_pred = Phi @ Coefficients

    Args:
        phi_matrix: Design matrix.
        adaptive_coeffs: Fitted coefficients.

    Returns:
        torch.Tensor: Predictions.
    """
    phi_matrix = phi_matrix.to(TORCH_DEVICE)
    adaptive_coeffs = adaptive_coeffs.to(TORCH_DEVICE)
    
    return phi_matrix @ adaptive_coeffs


def _slice_effect_input_with_cols(
    x_data: torch.Tensor,
    effect: BaseEffect,
    name_to_idx: Optional[Dict[str, int]],
    col_idx: int,
) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """Extracts the column slice and column indices expected by `effect.transform` without copying unrelated columns."""
    if effect.multi_input:
        req = effect.required_features
        if name_to_idx:
            if len(req) == 1:
                if req[0] not in name_to_idx:
                    raise ValueError(f"Multivariate sub-feature not found in data: '{req[0]}'.")
                idx = name_to_idx[req[0]]
                return x_data[..., idx : idx + 1], (idx,), col_idx
            try:
                indices = [name_to_idx[name] for name in req]
                return x_data[..., indices], tuple(indices), col_idx
            except KeyError as e:
                raise ValueError(f"Multivariate sub-feature not found in data: {e}.")
        n_inputs = len(req)
        if col_idx + n_inputs > x_data.shape[-1]:
            raise ValueError(f"Not enough columns for multivariate effect at index {col_idx}.")
        return x_data[..., col_idx : col_idx + n_inputs], tuple(range(col_idx, col_idx + n_inputs)), col_idx + n_inputs

    if name_to_idx:
        idx = name_to_idx.get(effect.feature_name)
        if idx is None:
            raise ValueError(f"Feature '{effect.feature_name}' not found in data.")
        return x_data[..., idx], (idx,), col_idx
    if col_idx >= x_data.shape[-1]:
        raise ValueError(f"Not enough columns for effect '{effect.feature_name}'.")
    return x_data[..., col_idx], (col_idx,), col_idx + 1


def _slice_effect_input(
    x_data: torch.Tensor,
    effect: BaseEffect,
    name_to_idx: Optional[Dict[str, int]],
    col_idx: int,
) -> Tuple[torch.Tensor, int]:
    """Extracts the column slice expected by `effect.transform` without copying unrelated columns."""
    x_eff, _, next_col = _slice_effect_input_with_cols(x_data, effect, name_to_idx, col_idx)
    return x_eff, next_col


def _is_plain_onehot_cat(effect: BaseEffect, allow_zero: bool = False) -> bool:
    """True when CategoricalEffect evaluates to pure 0/1 one-hot columns (no mean imputation or projection)."""
    if not isinstance(effect, CategoricalEffect):
        return False
    extrap_ok = (
        effect.extrapolate in ("continue", "constant")
        if not allow_zero
        else (effect.extrapolate == "continue" or (effect.extrapolate == "constant" and effect.unseen == "edge"))
    )
    unseen_ok = effect.unseen == "edge" if not allow_zero else effect.unseen in ("edge", "zero")
    return (
        effect.topology in ("nominal", "ordinal", "graph")
        and unseen_ok
        and extrap_ok
        and effect.n_categories >= (1 if allow_zero else 2)
        and not getattr(effect, "orthogonalize", False)
        and getattr(effect, "_projection", None) is None
    )


def _eval_effect_with_margin_cache(
    effect: BaseEffect,
    x_eff: Optional[torch.Tensor],
    col_indices: Optional[Tuple[int, ...]],
    margin_cache: Optional[Dict],
    eff_state: Optional[object] = None,
) -> torch.Tensor:
    """Evaluates an effect block, reusing identical stateless transforms from `margin_cache`."""
    from .spectrum._factory import _eval_effect_cached

    if isinstance(effect, OffsetEffect):
        return effect.transform(x_eff)
    if eff_state is None and margin_cache is not None and isinstance(effect, TensorProductEffect):
        return effect._eval_with_cache(
            x_eff, col_indices=col_indices, margin_cache=margin_cache, eval_margin_fn=_eval_effect_cached
        )
    return _eval_effect_cached(effect, x_eff, col_indices, margin_cache=margin_cache, state_in=eff_state)


def _contract_effect_block(
    effect: BaseEffect,
    x_eff: Optional[torch.Tensor],
    c_slice: torch.Tensor,
    target_dtype: torch.dtype,
    eff_state: Optional[object] = None,
    margin_cache: Optional[Dict] = None,
    col_indices: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    r"""Computes ``Phi_effect @ c_slice`` for a single effect, exploiting sparse categorical selection and shared margins."""
    target_dev = c_slice.device
    if isinstance(effect, CategoricalEffect) and _is_plain_onehot_cat(effect) and x_eff is not None and x_eff.dim() == 3 - int(not effect.multi_input) and c_slice.dim() == 3:
        x_cat = x_eff[..., 0] if (effect.topology == "graph" and x_eff.shape[-1] == len(effect.input_features)) else x_eff
        if x_cat.dim() == 2:
            x_idx = torch.clamp(torch.round((x_cat + 1.0) / 2.0 * (effect.n_categories - 1)).long(), 0, effect.n_categories - 1).to(device=target_dev)
            return c_slice.gather(1, x_idx.unsqueeze(-1).expand(-1, -1, c_slice.shape[-1]))

    if (
        eff_state is None
        and type(effect) is SplineEffect
        and getattr(effect, "_cached_knots", None) is not None
        and not getattr(effect, "multi_input", False)
        and getattr(effect, "_projection", None) is None
        and effect.spline_degree >= 1
        and effect.extrapolate in ("continue", "constant", "periodic", "linear", "saturation")
        and x_eff is not None
        and x_eff.dim() == 2
        and c_slice.dim() == 3
    ):
        return effect.contract_coefficients(x_eff, c_slice.to(dtype=target_dtype))

    if (
        isinstance(effect, TensorProductEffect)
        and len(effect.effects) == 2
        and effect.extrapolate == "continue"
        and not getattr(effect, "stateful", False)
        and eff_state is None
        and x_eff is not None
        and x_eff.dim() == 3
        and c_slice.dim() == 3
        and not getattr(effect, "orthogonalize", False)
        and getattr(effect, "_projection", None) is None
        and not any(getattr(m, "_projection", None) is not None for m in effect.effects)
    ):
        e0, e1 = effect.effects
        cat_side = 1 if _is_plain_onehot_cat(e1) else (0 if _is_plain_onehot_cat(e0) else -1)
        if cat_side != -1:
            n0_cols = len(e0.required_features)
            if cat_side == 1:
                other_eff, cat_eff = e0, e1
                x_other = x_eff[..., :n0_cols]
                x_cat_in = x_eff[..., n0_cols:]
                other_cols = None if col_indices is None else col_indices[:n0_cols]
            else:
                cat_eff, other_eff = e0, e1
                x_cat_in = x_eff[..., :n0_cols]
                x_other = x_eff[..., n0_cols:]
                other_cols = None if col_indices is None else col_indices[n0_cols:]
            if len(other_eff.required_features) == 1 and not other_eff.keeps_feature_dim:
                x_other = x_other.squeeze(-1)
            phi_other = _eval_effect_with_margin_cache(other_eff, x_other, other_cols, margin_cache).to(
                device=target_dev, dtype=target_dtype
            )
            if phi_other.dim() == 3:
                g_dim, t_dim, d_other = phi_other.shape
                d_out = c_slice.shape[-1]
                n_cat = cat_eff.n_categories
                x_cat = x_cat_in[..., 0] if (cat_eff.topology == "graph" or len(cat_eff.required_features) == 1) else x_cat_in.squeeze(-1)
                x_idx = torch.clamp(torch.round((x_cat + 1.0) / 2.0 * (n_cat - 1)).long(), 0, n_cat - 1).to(device=target_dev)
                if cat_side == 1:
                    beta = c_slice.view(g_dim, d_other, n_cat * d_out)
                else:
                    beta = c_slice.view(g_dim, n_cat, d_other, d_out).permute(0, 2, 1, 3).reshape(g_dim, d_other, n_cat * d_out)
                if n_cat <= 16:
                    scores = torch.bmm(phi_other, beta).view(g_dim, t_dim, n_cat, d_out)
                    return scores.gather(2, x_idx.unsqueeze(-1).unsqueeze(-1).expand(g_dim, t_dim, 1, d_out)).squeeze(2)
                beta_4 = beta.view(g_dim, d_other, n_cat, d_out).transpose(1, 2)  # (G, n_cat, D_other, d_out)
                flat_beta = beta_4.reshape(g_dim, n_cat, d_other * d_out)
                beta_sel = flat_beta.gather(1, x_idx.unsqueeze(-1).expand(g_dim, t_dim, d_other * d_out)).view(g_dim, t_dim, d_other, d_out)
                return torch.sum(phi_other.unsqueeze(-1) * beta_sel, dim=-2)

    block = _eval_effect_with_margin_cache(effect, x_eff, col_indices, margin_cache, eff_state=eff_state).to(
        device=target_dev, dtype=target_dtype
    )
    if block.dim() == 3 and c_slice.dim() == 3:
        return torch.bmm(block, c_slice)
    return block @ c_slice


def _contract_shared_group_block(
    effect: BaseEffect,
    x_eff: torch.Tensor,
    c_slice: torch.Tensor,
    target_dtype: torch.dtype,
    margin_cache: Dict,
    col_indices: Optional[Tuple[int, ...]],
) -> torch.Tensor:
    """Evaluates a group-invariant stateless effect once on ``x_eff[0:1]`` and contracts over all ``G`` groups."""
    g_dim, t_dim = int(x_eff.shape[0]), int(x_eff.shape[1])
    d_out = int(c_slice.shape[-1])
    target_dev = c_slice.device
    x_eff_0 = x_eff[0:1]

    if isinstance(effect, CategoricalEffect) and _is_plain_onehot_cat(effect) and x_eff.dim() == 3 - int(not effect.multi_input) and c_slice.dim() == 3:
        x_cat_0 = x_eff_0[..., 0] if (effect.topology == "graph" and x_eff_0.shape[-1] == len(effect.input_features)) else x_eff_0
        if x_cat_0.dim() == 2:
            x_idx_0 = torch.clamp(torch.round((x_cat_0 + 1.0) / 2.0 * (effect.n_categories - 1)).long(), 0, effect.n_categories - 1).to(device=target_dev)
            return c_slice.gather(1, x_idx_0.expand(g_dim, t_dim).unsqueeze(-1).expand(g_dim, t_dim, d_out))

    if (
        type(effect) is SplineEffect
        and getattr(effect, "_cached_knots", None) is not None
        and not getattr(effect, "multi_input", False)
        and getattr(effect, "_projection", None) is None
        and effect.spline_degree >= 1
        and effect.extrapolate in ("continue", "constant", "periodic", "linear", "saturation")
        and x_eff.dim() == 2
        and c_slice.dim() == 3
    ):
        return effect.contract_coefficients(x_eff_0, c_slice.to(dtype=target_dtype))

    if (
        isinstance(effect, TensorProductEffect)
        and len(effect.effects) == 2
        and effect.extrapolate == "continue"
        and not getattr(effect, "stateful", False)
        and x_eff.dim() == 3
        and c_slice.dim() == 3
        and not getattr(effect, "orthogonalize", False)
        and getattr(effect, "_projection", None) is None
        and not any(getattr(m, "_projection", None) is not None for m in effect.effects)
    ):
        e0, e1 = effect.effects
        cat_side = 1 if _is_plain_onehot_cat(e1) else (0 if _is_plain_onehot_cat(e0) else -1)
        if cat_side != -1:
            n0_cols = len(e0.required_features)
            if cat_side == 1:
                other_eff, cat_eff = e0, e1
                x_other_0 = x_eff_0[..., :n0_cols]
                x_cat_0 = x_eff_0[..., n0_cols:]
                other_cols = None if col_indices is None else col_indices[:n0_cols]
            else:
                cat_eff, other_eff = e0, e1
                x_cat_0 = x_eff_0[..., :n0_cols]
                x_other_0 = x_eff_0[..., n0_cols:]
                other_cols = None if col_indices is None else col_indices[n0_cols:]
            if len(other_eff.required_features) == 1 and not other_eff.keeps_feature_dim:
                x_other_0 = x_other_0.squeeze(-1)
            phi_other_0 = _eval_effect_with_margin_cache(other_eff, x_other_0, other_cols, margin_cache).to(
                device=target_dev, dtype=target_dtype
            )
            if phi_other_0.dim() == 3:
                d_other = int(phi_other_0.shape[-1])
                n_cat = cat_eff.n_categories
                x_cat_s = x_cat_0[..., 0] if (cat_eff.topology == "graph" or len(cat_eff.required_features) == 1) else x_cat_0.squeeze(-1)
                x_idx_0 = torch.clamp(torch.round((x_cat_s + 1.0) / 2.0 * (n_cat - 1)).long(), 0, n_cat - 1).to(device=target_dev)
                if cat_side == 1:
                    beta = c_slice.view(g_dim, d_other, n_cat, d_out)
                else:
                    beta = c_slice.view(g_dim, n_cat, d_other, d_out).transpose(1, 2)
                if n_cat <= 16:
                    # One (T, d_other) @ (d_other, G * n_cat * d_out) GEMM, then gather shared category index
                    beta_mat = beta.permute(1, 0, 2, 3).reshape(d_other, g_dim * n_cat * d_out)
                    scores = (phi_other_0[0] @ beta_mat).view(t_dim, g_dim, n_cat, d_out).transpose(0, 1)
                    return scores.gather(
                        2, x_idx_0.expand(g_dim, t_dim).unsqueeze(-1).unsqueeze(-1).expand(g_dim, t_dim, 1, d_out)
                    ).squeeze(2)
                beta_4 = beta.transpose(1, 2)  # (G, n_cat, d_other, d_out)
                flat_beta = beta_4.reshape(g_dim, n_cat, d_other * d_out)
                beta_sel = flat_beta.gather(
                    1, x_idx_0.expand(g_dim, t_dim).unsqueeze(-1).expand(g_dim, t_dim, d_other * d_out)
                ).view(g_dim, t_dim, d_other, d_out)
                return torch.sum(phi_other_0.unsqueeze(-1) * beta_sel, dim=-2)

    phi_0 = _eval_effect_with_margin_cache(effect, x_eff_0, col_indices, margin_cache).to(
        device=target_dev, dtype=target_dtype
    )
    if phi_0.dim() == 3 and c_slice.dim() == 3:
        d_eff = int(phi_0.shape[-1])
        c_mat = c_slice.permute(1, 0, 2).reshape(d_eff, g_dim * d_out)
        return (phi_0[0] @ c_mat).view(t_dim, g_dim, d_out).transpose(0, 1)
    return phi_0 @ c_slice


def _predict_from_components(
    x_data: torch.Tensor,
    adaptive_coeffs: torch.Tensor,
    effects_list: List[BaseEffect],
    feature_columns: Optional[List[str]] = None,
    state_in: Optional[Dict[int, object]] = None,
) -> torch.Tensor:
    r"""Computes ``Phi @ coeffs`` by accumulating effect blocks one by one without concatenating ``(G, T, D)``."""
    from .spectrum._factory import build_phi_from_effects, _project_blocks, _infer_feature_columns

    target_dtype = torch.promote_types(adaptive_coeffs.dtype, torch.get_default_dtype())
    coeffs_dev = adaptive_coeffs.to(device=TORCH_DEVICE, dtype=target_dtype)
    has_proj = any(getattr(e, "_projection", None) is not None for e in effects_list)

    if has_proj:
        components = build_phi_from_effects(
            x_data, effects_list, feature_columns=feature_columns, return_components=True, state_in=state_in
        )
        components = _project_blocks(effects_list, components)
        y_hat = torch.zeros(
            (x_data.shape[0], x_data.shape[1], coeffs_dev.shape[-1]), device=TORCH_DEVICE, dtype=target_dtype
        )
        coeff_idx = 0
        for effect, component in zip(effects_list, components):
            n_coeffs = effect.get_n_coeffs()
            block = component.to(device=TORCH_DEVICE, dtype=target_dtype)
            c_slice = coeffs_dev[..., coeff_idx : coeff_idx + n_coeffs, :]
            if block.dim() == 3 and c_slice.dim() == 3:
                y_hat.baddbmm_(block, c_slice)
            else:
                y_hat.add_(block @ c_slice)
            coeff_idx += n_coeffs
        return y_hat

    if feature_columns is None:
        feature_columns = _infer_feature_columns(effects_list)
    name_to_idx = {name: i for i, name in enumerate(feature_columns)} if feature_columns else None
    g_dim, t_dim = int(x_data.shape[0]), int(x_data.shape[1])
    y_hat = torch.zeros((g_dim, t_dim, coeffs_dev.shape[-1]), device=TORCH_DEVICE, dtype=target_dtype)
    shared_cache: Dict = {}
    full_cache: Dict = {}
    can_share_groups = (state_in is None and g_dim >= 2 and x_data.dim() == 3 and t_dim >= 16)
    col_shared: Dict[Tuple[int, ...], bool] = {}
    spline_buckets: Dict[Tuple, List[Tuple[SplineEffect, int, torch.Tensor, torch.Tensor, Tuple[int, ...]]]] = {}
    can_batch_splines = (state_in is None and x_data.dim() == 3 and coeffs_dev.dim() == 3)

    coeff_idx = 0
    col_idx = 0
    for eff_idx, effect in enumerate(effects_list):
        n_coeffs = effect.get_n_coeffs()
        c_slice = coeffs_dev[..., coeff_idx : coeff_idx + n_coeffs, :]
        eff_state = None if state_in is None else state_in.get(eff_idx, None)
        if isinstance(effect, OffsetEffect):
            y_hat.add_(c_slice)
        else:
            x_eff, cols_t, col_idx = _slice_effect_input_with_cols(x_data, effect, name_to_idx, col_idx)
            if (
                can_batch_splines
                and type(effect) is SplineEffect
                and getattr(effect, "_cached_knots", None) is not None
                and not getattr(effect, "multi_input", False)
                and effect.spline_degree >= 1
                and (n_coeffs - effect.spline_degree) >= 1
                and effect.extrapolate in ("continue", "constant", "periodic", "linear", "saturation")
                and x_eff.dim() == 2
                and len(cols_t) == 1
            ):
                bkey = (effect.spline_degree, n_coeffs, effect.extrapolate, effect._ood_slope_scale())
                spline_buckets.setdefault(bkey, []).append((effect, cols_t[0], c_slice, x_eff, cols_t))
            else:
                is_shared = False
                if can_share_groups and _is_rowwise_stateless_effects([effect]):
                    s_flag = col_shared.get(cols_t)
                    if s_flag is None:
                        s_flag = bool(torch.equal(x_eff, x_eff[0:1].expand_as(x_eff)))
                        col_shared[cols_t] = s_flag
                    is_shared = s_flag
                if is_shared:
                    y_hat.add_(_contract_shared_group_block(effect, x_eff, c_slice, target_dtype, shared_cache, cols_t))
                else:
                    y_hat.add_(
                        _contract_effect_block(
                            effect, x_eff, c_slice, target_dtype, eff_state=eff_state, margin_cache=full_cache, col_indices=cols_t
                        )
                    )
        coeff_idx += n_coeffs

    for bucket in spline_buckets.values():
        if len(bucket) == 1:
            effect, _, c_slice, x_eff, cols_t = bucket[0]
            is_shared = False
            if can_share_groups:
                s_flag = col_shared.get(cols_t)
                if s_flag is None:
                    s_flag = bool(torch.equal(x_eff, x_eff[0:1].expand_as(x_eff)))
                    col_shared[cols_t] = s_flag
                is_shared = s_flag
            if is_shared:
                y_hat.add_(_contract_shared_group_block(effect, x_eff, c_slice, target_dtype, shared_cache, cols_t))
            else:
                y_hat.add_(
                    _contract_effect_block(
                        effect, x_eff, c_slice, target_dtype, eff_state=None, margin_cache=full_cache, col_indices=cols_t
                    )
                )
        else:
            effs = [item[0] for item in bucket]
            cols_idx = [item[1] for item in bucket]
            c_batch = torch.stack([item[2] for item in bucket], dim=1)
            c0 = cols_idx[0]
            if cols_idx == list(range(c0, c0 + len(cols_idx))):
                x_batch = x_data[..., c0 : c0 + len(cols_idx)]
            else:
                x_batch = x_data[..., cols_idx]
            if can_share_groups and bool(torch.equal(x_batch, x_batch[0:1].expand_as(x_batch))):
                x_batch = x_batch[0:1]
            y_hat.add_(SplineEffect.contract_batched_splines(effs, x_batch, c_batch))
    return y_hat

#: <decompose>
def decomposition_names(effects_list: List[BaseEffect]) -> List[str]:
    r"""
    Names the additive component contributed by each effect, aligned with ``effects_list``.

    Smart Naming:
        - The intercept is 'offset'.
        - Unique features keep their name (e.g., 'temp').
        - Collisions (e.g., 's(x)' and 'l(x)') are prefixed (e.g., 's_x', 'l_x').
        - A collision that survives prefixing (two tensor products over the same features)
          is suffixed with its occurrence index, so no contribution silently overwrites another.

    ``decompose_prediction`` emits one ``effect_<name>`` column per entry. Callers that need
    the column of a specific formula term use this list rather than guessing from a feature
    name, which cannot resolve tensor products or several bases on one feature.

    Args:
        effects_list: List of effects defining the model structure.

    Returns:
        List[str]: One component name per effect, in the same order.
    """
    feature_names = [e.feature_name for e in effects_list if not isinstance(e, OffsetEffect)]
    name_counts = Counter(feature_names)

    # Map effect types to short prefixes for disambiguation
    type_map = {
        'linear': 'l', 'fourier': 'f', 'spline': 's', 'wavelet': 'w',
        'chebyshev': 'p', 'categorical_nominal': 'c', 'categorical_ordinal': 'c', 'categorical_fourier': 'c', 'categorical_graph': 'c',
        'neural': 'n', 'rbf_gauss': 'rbf', 'rbf_matern': 'rbf', 'rbf_tps': 'rbf', 'tree': 't', 'linear_tree': 'lt', 'pid': 'pid',
        'tensor_product': 'te', 'phys_spline': 'phys',
        'koopman': 'koopman',
        'foundation': 'foundation',
        'patch_basis': 'patch_basis',
        'temporal_operator': 'temp_op',
        'warped_calendar': 'wcal',
        'macro_trend': 'hp_trend',
        'degree_day': 'hdd_cdd',
        'thermal_inertia': 'thermal',
        'inertia': 'inertia', 'inertia_thermal': 'inertia', 'inertia_rotational': 'inertia', 'inertia_behavioral': 'inertia',
        'kan': 'kan',
        'single_index': 'si',
        'signature': 'sig',
        'prior_spectral': 'pspec',
        'hippo': 'hippo',
        'transformer': 'attn',
        'bispectrum': 'bispec',
        'scattering': 'wst',
        'tjepa': 'tjepa',
        'operator': 'fno',
        'hurst': 'hurst',
        'ssa': 'ssa',
        'phys_fourier': 'phys', 'phys_neural': 'phys', 'offset': 'offset'
    }

    names = []
    occurrences = Counter()
    for effect in effects_list:
        if isinstance(effect, OffsetEffect):
            name = "offset"
        elif name_counts[effect.feature_name] > 1:
            prefix = type_map.get(effect.effect_type, effect.effect_type)
            name = f"{prefix}_{effect.feature_name}"
        else:
            name = effect.feature_name
        occurrences[name] += 1
        names.append(name if occurrences[name] == 1 else f"{name}_{occurrences[name]}")
    return names


def _decompose_prediction_tensor(
    phi_matrix: torch.Tensor,
    adaptive_coeffs: torch.Tensor,
    effects_list: List[BaseEffect],
) -> Dict[str, torch.Tensor]:
    r"""
    Decomposes the total prediction into contributions from individual features.

    Iterates through the global design matrix and coefficient vector, slicing
    them according to the size of each effect to compute `Phi_j @ Beta_j`.

    Smart Naming:
        - Unique features keep their name (e.g., 'temp').
        - Collisions (e.g., 's(x)' and 'l(x)') are prefixed (e.g., 's_x', 'l_x').

    Args:
        phi_matrix: Global design matrix.
        adaptive_coeffs: Global fitted coefficients.
        effects_list: List of effects defining the model structure.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping feature names to their
        contribution tensors (real-valued).
    """
    phi_matrix = phi_matrix.to(TORCH_DEVICE)
    adaptive_coeffs = adaptive_coeffs.to(TORCH_DEVICE)

    decomposed_effects = {}
    names = decomposition_names(effects_list)

    coeff_idx = 0
    for effect, final_name in zip(effects_list, names):
        n_coeffs = effect.get_n_coeffs()

        # Slice the global matrices
        phi_slice = phi_matrix[..., :, coeff_idx : coeff_idx + n_coeffs]
        coeffs_slice = adaptive_coeffs[..., coeff_idx : coeff_idx + n_coeffs, :]

        # Compute contribution: Phi_j @ Beta_j
        contribution = (phi_slice @ coeffs_slice).squeeze(-1) # Assumes d_out=1

        decomposed_effects[final_name] = contribution
        coeff_idx += n_coeffs

    return decomposed_effects


def _decompose_from_components(
    x_data: torch.Tensor,
    adaptive_coeffs: torch.Tensor,
    effects_list: List[BaseEffect],
) -> Dict[str, torch.Tensor]:
    r"""
    Same contributions as ``_decompose_prediction_tensor(build_phi_from_effects(x_data, effects_list), ...)``, streamed per block.

    Each block ``Phi_j`` is contracted with its own coefficients as soon as it is built, so the concatenated
    ``(groups, rows, D)`` design is never materialised (a block-streamed product in the spirit of tiled attention kernels,
    Dao et al., 2022): the memory traffic of the full concatenation and its re-slicing is removed.
    """
    from .spectrum._factory import build_phi_from_effects, _project_blocks, _infer_feature_columns

    has_proj = any(getattr(e, "_projection", None) is not None for e in effects_list)
    if has_proj:
        components = build_phi_from_effects(x_data, effects_list, return_components=True)
        components = _project_blocks(effects_list, components)
        dtype = components[0].dtype
        for component in components[1:]:
            dtype = torch.promote_types(dtype, component.dtype)
        adaptive_coeffs = adaptive_coeffs.to(TORCH_DEVICE)

        decomposed_effects = {}
        coeff_idx = 0
        for effect, final_name, component in zip(effects_list, decomposition_names(effects_list), components):
            n_coeffs = effect.get_n_coeffs()
            block = component.to(device=TORCH_DEVICE, dtype=dtype)
            coeffs_slice = adaptive_coeffs[..., coeff_idx : coeff_idx + n_coeffs, :]
            decomposed_effects[final_name] = (block @ coeffs_slice).squeeze(-1)
            coeff_idx += n_coeffs
        return decomposed_effects

    target_dtype = torch.promote_types(adaptive_coeffs.dtype, torch.get_default_dtype())
    coeffs_dev = adaptive_coeffs.to(device=TORCH_DEVICE, dtype=target_dtype)
    feature_columns = _infer_feature_columns(effects_list)
    name_to_idx = {name: i for i, name in enumerate(feature_columns)} if feature_columns else None
    g_dim, t_dim = x_data.shape[0], x_data.shape[1]

    decomposed_effects = {}
    shared_cache: Dict = {}
    full_cache: Dict = {}
    can_share_groups = (g_dim >= 2 and x_data.dim() == 3 and t_dim >= 16)
    col_shared: Dict[Tuple[int, ...], bool] = {}
    coeff_idx = 0
    col_idx = 0
    for effect, final_name in zip(effects_list, decomposition_names(effects_list)):
        n_coeffs = effect.get_n_coeffs()
        c_slice = coeffs_dev[..., coeff_idx : coeff_idx + n_coeffs, :]
        if isinstance(effect, OffsetEffect):
            decomposed_effects[final_name] = c_slice.squeeze(-1).expand(g_dim, t_dim).clone()
        else:
            x_eff, cols_t, col_idx = _slice_effect_input_with_cols(x_data, effect, name_to_idx, col_idx)
            is_shared = False
            if can_share_groups and _is_rowwise_stateless_effects([effect]):
                s_flag = col_shared.get(cols_t)
                if s_flag is None:
                    s_flag = bool(torch.equal(x_eff, x_eff[0:1].expand_as(x_eff)))
                    col_shared[cols_t] = s_flag
                is_shared = s_flag
            if is_shared:
                decomposed_effects[final_name] = _contract_shared_group_block(
                    effect, x_eff, c_slice, target_dtype, shared_cache, cols_t
                ).squeeze(-1)
            else:
                decomposed_effects[final_name] = _contract_effect_block(
                    effect, x_eff, c_slice, target_dtype, margin_cache=full_cache, col_indices=cols_t
                ).squeeze(-1)
        coeff_idx += n_coeffs
    return decomposed_effects
#: </decompose>

#: <gcv_score>
def compute_gcv_score(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    Y_sq: torch.Tensor,
    penalty_M_star_M: torch.Tensor,
    lambda_p: float,
    n_samples: int,
    gamma: float = 1.0,
    trace_method: str = "exact",
    n_probes: int = 32,
    seed: int = 42
) -> torch.Tensor:
    r"""
    Computes the Generalized Cross Validation (GCV) score for a given lambda_p.
    
    GCV Formula (approx):
        GCV(lambda_p) = (1/n * RSS) / (1 - gamma/n * Tr(S))^2
    
    Numerical Optimization:
    Instead of computing the trace of the full Smoothing matrix S (N x N),
    we utilize the cyclic property of the trace to compute it on the 
    inverted covariance matrix (K x K), where K << N.
    When trace_method='hutchinson' (or trace_method='auto' with K > 2000), uses Hutchinson Rademacher stochastic
    trace estimation without explicitly inverting A.

    Args:
        cov_X: Weighted feature covariance (Phi'Phi). Shape (B, K, K).
        cov_XY: Feature-Target covariance (Phi'Y). Shape (B, K, 1).
        Y_sq: Sum of squared targets Y'Y, one scalar per group. Shape (B,).
        penalty_M_star_M: Penalty structure P. Shape (K, K).
        lambda_p: The regularization strength scalar to evaluate.
        n_samples: Number of samples (n).
        gamma: Multiplier for the effective degrees of freedom to prevent 
               overfitting in autocorrelated data (typically 1.4 - 1.5).
        trace_method: 'exact', 'hutchinson', or 'auto'.
        n_probes: Number of Rademacher probe vectors for Hutchinson trace estimator.
        seed: Deterministic seed for Rademacher probes.

    Returns:
        torch.Tensor: The mean GCV score across the batch (scalar).
    """
    batch_size = cov_X.shape[0]
    
    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()
        
    #  Construct the Regularized System: A = (Phi'Phi + lambda_p * n * P)
    reg_term = (n_samples * lambda_p * penalty_M_star_M).unsqueeze(0) # Broadcast to batch
    
    A = cov_X + reg_term
    
    # Same ridge floor as solve_linear_system, so the scored model is the one fit() would return
    A = A + _ridge_floor(n_samples, A.shape[-1], cov_X.dtype, cov_X.device)
    
    K = cov_X.shape[-1]
    use_hutchinson = trace_method == "hutchinson" or (trace_method == "auto" and K > 2000)

    #  Solve for Coefficients and Compute Exact Trace via Shared Cholesky
    if not use_hutchinson:
        L_chol, info = torch.linalg.cholesky_ex(A)
        if not bool(info.any()):
            coeffs = torch.cholesky_solve(cov_XY, L_chol)
            A_inv = torch.cholesky_inverse(L_chol)
            cross_term = 2 * torch.sum(coeffs * cov_XY, dim=1).squeeze(-1)
            quad_term = (coeffs.mT @ cov_X @ coeffs).squeeze(-1).squeeze(-1)
            rss = Y_sq - cross_term + quad_term
            min_rss = torch.min(rss)
            if min_rss < -1e-8 * torch.max(torch.abs(Y_sq)).clamp(min=1.0):
                logger.debug("GCV: negative RSS %.3e clamped to 0 (rounding in Y'Y - 2B'Phi'Y + B'Phi'PhiB).", min_rss.item())
            rss = torch.clamp(rss, min=0.0)
            mse = rss / n_samples
            trace_H = (cov_X * A_inv).sum(dim=(-2, -1))
            denom_factor = torch.clamp(1.0 - (gamma * trace_H / n_samples), min=1e-6)
            return torch.mean(mse / (denom_factor ** 2))

    try:
        coeffs = hw.safe_solve(A, cov_XY) # Shape: (B, K, 1)
    except RuntimeError:
        return torch.tensor(float('inf'), device=cov_X.device)
    
    #  Compute Numerator: MSE (Mean Squared Error) on Training set
    # Using expansion: ||Y - Phi B||^2 = Y'Y - 2 B' Phi'Y + B' Phi'Phi B
        
    # Cross term: 2 * real(Beta^H @ cov_XY)
    cross_term = 2 * torch.sum(coeffs * cov_XY, dim=1).squeeze(-1)
    
    # Quadratic term: Beta^H @ cov_X @ Beta
    quad_term = (coeffs.mT @ cov_X @ coeffs).squeeze(-1).squeeze(-1)
    
    # Residual Sum of Squares (RSS)
    rss = Y_sq - cross_term + quad_term
    # The expansion can dip below 0 by rounding when the fit is near-exact; clamp instead of folding it with abs()
    min_rss = torch.min(rss)
    if min_rss < -1e-8 * torch.max(torch.abs(Y_sq)).clamp(min=1.0):
        logger.debug("GCV: negative RSS %.3e clamped to 0 (rounding in Y'Y - 2B'Phi'Y + B'Phi'PhiB).", min_rss.item())
    rss = torch.clamp(rss, min=0.0)
    mse = rss / n_samples
    
    if use_hutchinson:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
        z_cpu = torch.randint(0, 2, (K, n_probes), generator=gen, dtype=cov_X.dtype) * 2.0 - 1.0
        # Rademacher probes
        Z = z_cpu.to(device=cov_X.device, dtype=cov_X.dtype).unsqueeze(0).expand(batch_size, K, n_probes)
        AZ = hw.safe_solve(A, Z)
        cov_AZ = cov_X @ AZ
        trace_H = torch.mean(torch.sum(Z * cov_AZ, dim=1), dim=-1)
        denom_factor = torch.clamp(1.0 - (gamma * trace_H / n_samples), min=1e-6)
        return torch.mean(mse / (denom_factor ** 2))
    #  Compute Denominator: Effective Degrees of Freedom (Trace of S)
    # Tr(S) = Tr(cov_X @ A^-1)
    # Since K is small (<1000), direct inversion on GPU is efficient.
    A_inv = torch.linalg.inv(A) 
    
    # Tr(cov_X @ A^-1) in O(K^2)
    trace_H = (cov_X * A_inv).sum(dim=(-2, -1))
    
    # Protect against division by zero if model interpolates perfectly
    denom_factor = 1.0 - (gamma * trace_H / n_samples)
    denom_factor = torch.clamp(denom_factor, min=1e-6)
    
    #  Final GCV Score
    gcv = mse / (denom_factor ** 2)
    
    return torch.mean(gcv)


def compute_gcv_score_and_grad(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    Y_sq: torch.Tensor,
    penalty_blocks: List[torch.Tensor],
    spans: List[ tuple ],
    alpha_ps: torch.Tensor,
    n_samples: int,
    gamma: float = 1.4
) -> tuple:
    r"""
    Computes the GCV score and its exact analytical gradient with respect to
    log10 smoothing coordinates alpha_j = log10(lambda_j) (Wood 2004/2011).

    For A(lambda) = cov_X + n * sum_j 10^{alpha_j} S_j + ridge_floor:
      - d(Beta)/d(lambda_j) = -n * A^{-1} S_j Beta
      - d(RSS)/d(lambda_j) = 2 * n^2 * lambda_j * Beta^T P S_j A^{-1} cov_X Beta
      - d(Tr(S))/d(lambda_j) = -n * Tr(A^{-1} cov_X A^{-1} S_j)
      - d/d(alpha_j) = (ln 10) * lambda_j * d/d(lambda_j)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (scalar_gcv, grad_alpha of shape (J,))
    """
    dtype = cov_X.dtype
    device = cov_X.device
    B, K, _ = cov_X.shape
    J = len(penalty_blocks)
    ln10 = float(torch.log(torch.tensor(10.0, dtype=dtype)))

    lambdas = (10.0 ** alpha_ps.to(device=device, dtype=dtype))
    P_total = torch.zeros((K, K), dtype=dtype, device=device)
    S_blocks_full = []
    for j in range(J):
        s_start, s_end = spans[j]
        S_j = penalty_blocks[j].to(device=device, dtype=dtype)
        if S_j.is_sparse:
            S_j = S_j.to_dense()
        P_total[s_start:s_end, s_start:s_end] += lambdas[j] * S_j
        S_full = torch.zeros((K, K), dtype=dtype, device=device)
        S_full[s_start:s_end, s_start:s_end] = S_j
        S_blocks_full.append(S_full)

    A = cov_X + (n_samples * P_total).unsqueeze(0) + _ridge_floor(n_samples, K, dtype, device)
    A_inv = torch.linalg.inv(A)
    coeffs = A_inv @ cov_XY

    cross_term = 2.0 * torch.sum(coeffs * cov_XY, dim=1).squeeze(-1)
    quad_term = (coeffs.mT @ cov_X @ coeffs).squeeze(-1).squeeze(-1)
    rss = torch.clamp(Y_sq - cross_term + quad_term, min=1e-12)
    mse = rss / n_samples

    H_matrix = cov_X @ A_inv
    trace_H = torch.einsum('bii->b', H_matrix)
    denom = torch.clamp(1.0 - (gamma * trace_H / n_samples), min=1e-6)
    gcv_per_group = mse / (denom ** 2)
    gcv_mean = torch.mean(gcv_per_group)

    # Residual of the normal equations (without the penalty term), used by the derivative of the residual sum of squares
    r_cov = (cov_X @ coeffs) - cov_XY  # (B, K, 1)
    # Precompute A_inv @ cov_X @ A_inv for trace derivative
    Ainv_covX_Ainv = A_inv @ H_matrix  # (B, K, K)

    grad_alpha = torch.zeros(J, dtype=dtype, device=device)
    for j in range(J):
        s_start, s_end = spans[j]
        S_j = penalty_blocks[j].to(device=device, dtype=dtype)
        if S_j.is_sparse:
            S_j = S_j.to_dense()
        # d(Beta)/d(lambda_j) = -n * A_inv[:, :, s_start:s_end] @ (S_j @ coeffs[:, s_start:s_end, :])
        Sj_beta = S_j.unsqueeze(0) @ coeffs[:, s_start:s_end, :]  # (B, k_j, 1)
        dBeta_dlam = -float(n_samples) * (A_inv[:, :, s_start:s_end] @ Sj_beta)  # (B, K, 1)
        dRSS_dlam = 2.0 * torch.sum(r_cov * dBeta_dlam, dim=1).squeeze(-1)  # (B,)

        # d(Tr(S))/d(lambda_j) = -n * Tr(Ainv_covX_Ainv[:, s_start:s_end, s_start:s_end] @ S_j)
        sub_mat = Ainv_covX_Ainv[:, s_start:s_end, s_start:s_end] @ S_j.unsqueeze(0)
        dTr_dlam = -float(n_samples) * torch.einsum('bii->b', sub_mat)

        # Quotient rule on GCV_b = (RSS_b / n) / denom_b^2
        dMSE_dlam = dRSS_dlam / float(n_samples)
        ddenom_dlam = -(gamma / float(n_samples)) * dTr_dlam
        dGCV_dlam = (dMSE_dlam * denom - 2.0 * mse * ddenom_dlam) / (denom ** 3)
        grad_alpha[j] = torch.mean(dGCV_dlam) * lambdas[j] * ln10

    return gcv_mean, grad_alpha
#: </gcv_score>

#: <solve_sparse_cg>
def solve_sparse_cg(
    compute_Av: Callable[[torch.Tensor], torch.Tensor], 
    b: torch.Tensor, 
    x0: Optional[torch.Tensor] = None, 
    tol: float = 1e-4, 
    max_iter: int = 1000,
    preconditioner: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None
) -> torch.Tensor:
    r"""
    Solves the linear system Ax = b using the matrix-free (Preconditioned) Conjugate Gradient method.
    
    This solver never explicitly constructs the dense covariance matrix A. Instead, 
    it relies on a closure `compute_Av` that evaluates the matrix-vector product 
    (A @ v) dynamically. Supports optional Jacobi (inverse-diagonal tensor) or
    Block-Jacobi (callable closure M^{-1} r) preconditioning to accelerate convergence
    on multi-scale penalized designs.

    Args:
        compute_Av: A closure evaluating (A @ v) for a given vector v.
        b: The right-hand side target vector (\Phi^* Y).
        x0: Initial guess for the coefficients.
        tol: Tolerance threshold for the residual norm.
        max_iter: Maximum number of iterations before stopping.
        preconditioner: Optional inverse-diagonal tensor M_inv (Jacobi) or callable M_inv(r) (Block-Jacobi).

    Returns:
        torch.Tensor: The optimized coefficient vector.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    def apply_precond(vec: torch.Tensor) -> torch.Tensor:
        """Applies the preconditioner to a vector or to a block of vectors."""
        if preconditioner is None:
            return vec
        if callable(preconditioner):
            return preconditioner(vec)
        return preconditioner * vec

    r = b - compute_Av(x)
    z = apply_precond(r)
    p = z.clone()
    
    is_batched = b.dim() >= 3
    sum_dims = tuple(range(1, b.dim())) if is_batched else tuple(range(b.dim()))

    rz_old = torch.sum(r * z, dim=sum_dims, keepdim=True)
    norm_b = torch.sqrt(torch.sum(b * b, dim=sum_dims, keepdim=True))
    norm_b = torch.clamp(norm_b, min=1e-12)

    for _ in range(max_iter):
        Ap = compute_Av(p)
        p_Ap = torch.sum(p * Ap, dim=sum_dims, keepdim=True)
        
        valid_mask = p_Ap > 1e-12
        alpha_cg = torch.zeros_like(rz_old)
        alpha_cg[valid_mask] = rz_old[valid_mask] / p_Ap[valid_mask]
        
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        
        r_norm_sq = torch.sum(r * r, dim=sum_dims, keepdim=True)
        if torch.all(torch.sqrt(r_norm_sq) / norm_b < tol):
            break
            
        z = apply_precond(r)
        rz_new = torch.sum(r * z, dim=sum_dims, keepdim=True)

        beta = torch.zeros_like(rz_new)
        beta[valid_mask] = rz_new[valid_mask] / torch.clamp(rz_old[valid_mask], min=1e-18)
        
        p = z + beta * p
        rz_old = rz_new

    return x
#: </solve_sparse_cg>


#: <solve_block_qr>
def solve_block_qr(
    phi_chunks: Sequence[torch.Tensor],
    y_chunks: Sequence[torch.Tensor],
    penalty_M_star_M: torch.Tensor,
    n_samples: Optional[int] = None
) -> torch.Tensor:
    r"""
    Incremental Tall-Skinny Block-QR solver for penalized least squares.

    Avoids squaring the condition number \kappa(\Phi^T \Phi) = \kappa(\Phi)^2 by
    maintaining an upper-triangular factor R \in R^{K x K} across row chunks via
    reduced Householder QR updates, and appending the square-root penalty factor
    L_P (where n * P + ridge_floor = L_P^T L_P) as a final synthetic row block.

    Args:
        phi_chunks: Sequence of design matrix chunks, each of shape (B, N_c, K) or (N_c, K).
        y_chunks: Sequence of target chunks, each of shape (B, N_c, d_out) or (N_c, d_out).
        penalty_M_star_M: Penalty matrix P of shape (K, K).
        n_samples: Total sample count across chunks (inferred if None).

    Returns:
        torch.Tensor: Fitted coefficients of shape (B, K, d_out) or (K, d_out).
    """
    if not phi_chunks:
        raise ValueError("phi_chunks must contain at least one tensor chunk.")

    unbatched = (phi_chunks[0].dim() == 2)
    p_list = [p.unsqueeze(0) if p.dim() == 2 else p for p in phi_chunks]
    y_list = [y.unsqueeze(0) if y.dim() == 2 else y for y in y_chunks]

    B, _, K = p_list[0].shape
    d_out = y_list[0].shape[-1]
    dtype = p_list[0].dtype
    device = p_list[0].device

    total_n = int(sum(p.shape[1] for p in p_list)) if n_samples is None else int(n_samples)

    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()
    P_scaled = total_n * penalty_M_star_M.to(device=device, dtype=dtype) + _ridge_floor(total_n, K, dtype, device)
    # Symmetric eigendecomposition for exact square-root factor L_P: L_P^T @ L_P = P_scaled
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (P_scaled + P_scaled.mT))
    L_P = torch.diag(torch.sqrt(torch.clamp(eigvals, min=0.0))) @ eigvecs.mT  # (K, K)

    R_curr = torch.zeros((B, K, K), dtype=dtype, device=device)
    qty_curr = torch.zeros((B, K, d_out), dtype=dtype, device=device)

    for phi_c, y_c in zip(p_list, y_list):
        phi_c = phi_c.to(device=device, dtype=dtype)
        y_c = y_c.to(device=device, dtype=dtype)
        stacked_mat = torch.cat([R_curr, phi_c], dim=1)          # (B, K + N_c, K)
        stacked_rhs = torch.cat([qty_curr, y_c], dim=1)          # (B, K + N_c, d_out)
        Q_step, R_curr = torch.linalg.qr(stacked_mat, mode='reduced')
        qty_curr = Q_step.mT @ stacked_rhs

    # Append square-root penalty block [L_P, 0]
    L_P_batch = L_P.unsqueeze(0).expand(B, K, K)
    zero_rhs = torch.zeros((B, K, d_out), dtype=dtype, device=device)
    final_mat = torch.cat([R_curr, L_P_batch], dim=1)
    final_rhs = torch.cat([qty_curr, zero_rhs], dim=1)
    Q_fin, R_fin = torch.linalg.qr(final_mat, mode='reduced')
    qty_fin = Q_fin.mT @ final_rhs

    coeffs = hw.safe_solve(R_fin, qty_fin)
    return coeffs.squeeze(0) if unbatched else coeffs
#: </solve_block_qr>


#: <sqrt_group_lasso>
def solve_sqrt_group_lasso(
    phi: torch.Tensor,
    y: torch.Tensor,
    group_spans: List[tuple],
    c_scale: float = 1.1,
    alpha_level: float = 0.05,
    max_iter: int = 150,
    tol: float = 1e-5,
    penalty_matrix: Optional[torch.Tensor] = None,
    debiased_refit: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    r"""
    Tuning-free Pivotal Square-Root Group LASSO with debiased L2 Primal refit.

    Minimizes the pivotal self-normalizing objective (Belloni, Chernozhukov & Wang 2011/2014):
        min_theta  (1 / sqrt(T)) ||y - Phi theta||_2 + (lambda_0 / T) sum_g sqrt(p_g) ||theta_g||_2
    where lambda_0 = c_scale * sqrt(T) is independent of the unknown noise variance sigma^2.
    After identifying the active effect groups S_hat = {g : ||theta_g||_2 > 1e-12} (standardized solution), performs an
    exact debiased L2 Primal solve restricted to S_hat.

    Args:
        phi: Design matrix of shape (T, K) or (1, T, K).
        y: Target vector of shape (T, 1) or (1, T, 1).
        group_spans: List of (start_col, end_col) tuples for each additive effect group.
        c_scale: Pivotal multiplier c > 1 (default 1.1).
        alpha_level: Nominal test level in (0, 1) used to calibrate threshold.
        max_iter: Maximum proximal gradient iterations.
        tol: Relative parameter change convergence threshold.
        penalty_matrix: Optional K x K smoothness penalty matrix for the debiased L2 refit.
        debiased_refit: If True, refits active groups using exact penalized L2 normal equations.

    Returns:
        Tuple[torch.Tensor, List[int]]: (fitted_theta of shape matching input, active_group_indices)
    """
    was_3d = (phi.dim() == 3)
    X = phi.squeeze(0) if was_3d else phi
    Y = y.squeeze(0) if y.dim() == 3 else (y.unsqueeze(-1) if y.dim() == 1 else y)
    T, K = X.shape
    dtype = X.dtype
    device = X.device

    # Column-standardize X for pivotal group thresholding
    col_norms = torch.clamp(torch.linalg.norm(X, dim=0) / np.sqrt(float(T)), min=1e-8)
    X_std = X / col_norms.unsqueeze(0)

    # Estimate largest singular value squared L = ||X_std^T X_std||_2 / T via 10 power iterations
    v = torch.ones((K, 1), dtype=dtype, device=device) / np.sqrt(float(K))
    for _ in range(10):
        v_next = X_std.mT @ (X_std @ v)
        v = v_next / torch.clamp(torch.linalg.norm(v_next), min=1e-12)
    L_lip = float(torch.linalg.norm(X_std @ v).item() ** 2) / float(T)
    L_lip = max(L_lip, 1e-4)

    # Pivotal penalty level calibrated via Gaussian quantile approximation
    z_alpha = float(torch.erfinv(torch.tensor(1.0 - alpha_level / max(len(group_spans), 1), dtype=torch.float64)).item() * np.sqrt(2.0))
    base_lam = float(c_scale) * max(z_alpha, 1.0) / np.sqrt(float(T))

    theta = torch.zeros((K, 1), dtype=dtype, device=device)
    y_scale = max(float(torch.linalg.norm(Y).item()) / np.sqrt(float(T)), 1e-300)  # the floors below are relative to the units of the response
    for _ in range(max_iter):
        theta_old = theta.clone()
        residual = Y - (X_std @ theta)
        res_norm = float(torch.linalg.norm(residual).item()) / np.sqrt(float(T))
        if res_norm < 1e-10 * y_scale:
            break

        # Gradient of sqrt-RSS: -(X_std^T residual) / (T * res_norm)
        grad = -(X_std.mT @ residual) / (float(T) * res_norm)
        step = 1.0 / (L_lip / max(res_norm, 1e-4 * y_scale))
        u = theta - step * grad

        # Group-wise block soft-thresholding (Leave intercept / 1-column offset unpenalized if index 0 has size 1 and constant)
        theta_new = torch.zeros_like(u)
        for g_idx, (s0, s1) in enumerate(group_spans):
            p_g = s1 - s0
            u_g = u[s0:s1]
            # Do not zero out a pure intercept column
            if p_g == 1 and torch.allclose(X[:, s0], torch.ones_like(X[:, s0]), atol=1e-5):
                theta_new[s0:s1] = u_g
                continue
            thresh = step * base_lam * (1.0 + 0.25 * np.sqrt(float(p_g)))
            norm_ug = float(torch.linalg.norm(u_g).item())
            if norm_ug > thresh:
                theta_new[s0:s1] = (1.0 - thresh / norm_ug) * u_g

        theta = theta_new
        if float(torch.linalg.norm(theta - theta_old).item()) <= tol * max(float(torch.linalg.norm(theta_old).item()), 1.0):
            break

    theta_unscaled = theta / col_norms.unsqueeze(-1)
    # The block soft-thresholding sets an inactive group exactly to zero: read the active groups on the standardized solution,
    # where the test does not depend on the units of the response.
    active_groups = [
        g_idx for g_idx, (s0, s1) in enumerate(group_spans)
        if float(torch.linalg.norm(theta[s0:s1]).item()) > 1e-12
    ]
    if not active_groups:
        active_groups = [0]

    if debiased_refit:
        active_cols = []
        for g_idx in active_groups:
            s0, s1 = group_spans[g_idx]
            active_cols.extend(range(s0, s1))
        idx_t = torch.tensor(active_cols, dtype=torch.long, device=device)
        X_act = X[:, idx_t]
        cov_X_act = (X_act.mT @ X_act).unsqueeze(0)
        cov_XY_act = (X_act.mT @ Y).unsqueeze(0)
        if penalty_matrix is not None:
            P_dense = penalty_matrix.to_dense() if penalty_matrix.is_sparse else penalty_matrix
            P_act = P_dense.to(device=device, dtype=dtype)[idx_t][:, idx_t]
        else:
            P_act = torch.zeros((len(active_cols), len(active_cols)), dtype=dtype, device=device)
        beta_act = solve_linear_system(cov_X_act, cov_XY_act, P_act, T).squeeze(0)
        theta_final = torch.zeros((K, 1), dtype=dtype, device=device)
        theta_final[idx_t] = beta_act
    else:
        theta_final = theta_unscaled

    return (theta_final.unsqueeze(0) if was_3d else theta_final), active_groups


def compute_scale_stability_penalty(u_proj: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    r"""
    Computes the nested scale-stability variance penalty (Collarin et al., 2025):
        R_var(u) = (Var(u) - target_var)^2
    preventing scale indeterminacy between inner index projections and outer spline bases.
    """
    var_u = torch.var(u_proj, dim=-2 if u_proj.dim() >= 2 else 0, unbiased=False)
    return torch.mean((var_u - float(target_var)) ** 2)
#: </sqrt_group_lasso>
