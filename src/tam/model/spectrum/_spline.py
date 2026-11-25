# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the P-Spline effect (B-Splines + Difference Penalty)."""

import torch
import numpy as np

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import eval_1d_dedup

#: <init_knots>
class SplineEffect(BaseEffect):
    r"""
    Implements a P-Spline effect.
    
    Robust implementation with manual knot vector construction to handle 
    sparse or constant data distributions.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_knots: int, 
        spline_degree: int, 
        penalty_order: int, 
        lambda_p: float,
        extrapolate: str,
        m_pen: int = 1
    ):
        super().__init__(feature_name, "spline", lambda_p, extrapolate)
        self.n_knots = n_knots
        self.spline_degree = spline_degree
        self.penalty_order = penalty_order

        self._n_coeffs = self.n_knots + self.spline_degree
        self._cached_knots = None
        self._knots_key = None
        self._device_knots_cache = {}
        self._active_table_cache = {}
        self._dense_inv_cache = {}
        self._batched_table_cache = None

        # m_pen > 1 splits the roughness penalty in m_pen locally weighted parts, each with its own smoothing parameter.
        self.m_pen = int(m_pen)
        n_differences = self._n_coeffs - self.penalty_order
        if self.m_pen < 1 or self.m_pen > max(n_differences, 1):
            raise ValueError(f"m_pen must be between 1 and the {max(n_differences, 1)} differences of the penalty of s({feature_name}); got {m_pen}.")
        self._part_lambdas = None

    def penalty_coordinates(self) -> list:
        r"""The smoothing weight of each part of the penalty (one per ``m_pen``)."""
        if self.m_pen == 1:
            return [self.lambda_p]
        return list(self._part_lambdas) if self._part_lambdas is not None else [self.lambda_p] * self.m_pen

    def set_penalty_coordinates(self, values) -> None:
        values = [float(v) for v in values]
        if len(values) != self.m_pen:
            raise ValueError(f"s({self.feature_name}) has {self.m_pen} penalty coordinates, got {len(values)}.")
        if self.m_pen == 1:
            self.lambda_p = values[0]
            return
        self._part_lambdas = values
        self.lambda_p = float(np.exp(np.mean(np.log([max(v, 1e-300) for v in values]))))  # the geometric mean stands for the effect where one weight is read

    def _part_weights(self, n_rows: int) -> torch.Tensor:
        r"""Weights ``W`` of the ``m_pen`` parts over the ``n_rows`` differences of the penalty: B-splines of degree ``min(3, m_pen - 1)``
        on the index of the difference, non-negative and adding up to one in every row (a partition of unity)."""
        dtype = torch.get_default_dtype()
        if self.m_pen == 1:
            return torch.ones(n_rows, 1, device=TORCH_DEVICE, dtype=dtype)
        degree = min(3, self.m_pen - 1)
        n_internal = self.m_pen - degree - 1
        internal = torch.linspace(0.0, 1.0, n_internal + 2, dtype=dtype)[1:-1]
        knots = torch.cat([torch.zeros(degree + 1, dtype=dtype), internal, torch.ones(degree + 1, dtype=dtype)])
        position = torch.linspace(0.0, 1.0, n_rows, dtype=dtype) if n_rows > 1 else torch.full((1,), 0.5, dtype=dtype)
        weights = self._cox_de_boor(position, knots, degree)
        return (weights / weights.sum(dim=-1, keepdim=True)).to(TORCH_DEVICE)

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Sets the knots from the full training tensor, so that the memory probe (one row per group) never decides them."""
        if self._cached_knots is None:
            self._get_knots(x_cols)

    def summary_details(self) -> str:
        return f"Knots={self.n_knots}, Diff(d={self.penalty_order})" + (f", m_pen={self.m_pen}" if self.m_pen > 1 else "")

    def get_n_coeffs(self) -> int:
        return self._n_coeffs

    def _get_knots(self, x_data: torch.Tensor, is_dummy: bool = False) -> torch.Tensor:
        r"""Manually constructs the full knot vector with correct padding.

        Once set from the training data, the knots are never recomputed: a prediction must not depend on how many values
        it is evaluated on. A single value (one row per group, or the lone out-of-range value of a frame) is evaluated on
        the trained knots like any other.
        """
        if self._cached_knots is not None:
            return self._cached_knots

        valid_x = x_data[torch.isfinite(x_data)]
        if len(valid_x) == 0:
            x_min, x_max = 0.0, 1.0
        else:
            x_min, x_max = valid_x.min().item(), valid_x.max().item()
        
        if np.isclose(x_min, x_max):
            x_max += 1e-3
            x_min -= 1e-3

        internal_knots = torch.linspace(
            x_min, x_max, self.n_knots + 1,
            device=x_data.device, dtype=torch.get_default_dtype()
        )[1:-1]

        k = self.spline_degree
        t_left = torch.full((k + 1,), x_min, device=x_data.device, dtype=torch.get_default_dtype())
        t_right = torch.full((k + 1,), x_max, device=x_data.device, dtype=torch.get_default_dtype())

        knots = torch.cat([t_left, internal_knots, t_right])

        if not is_dummy:
            self._cached_knots = knots
            self._knots_key = tuple(knots.detach().cpu().tolist()) if knots.numel() <= 256 else None
            self._n_coeffs = len(self._cached_knots) - (self.spline_degree + 1)
            self._device_knots_cache = {(knots.device, knots.dtype): knots}
            self._active_table_cache = {}
            self._dense_inv_cache = {}
            self._batched_table_cache = None

        return knots
#: </init_knots>

#: <cox_de_boor>
#: <ood_slope_scale>
    def _ood_slope_scale(self) -> float:
        r"""Scales clamped boundary derivatives by ``2 / k`` via Marsden's Greville-abscissa identity for REML-fitted splines.

        Under clamped end-knots and a penalty of order ``p >= 2``, the last two Greville abscissae on an interior step ``h``
        sit at distances ``h/3`` and ``h`` from the endpoint, inflating ``f'(boundary)`` by ``k / 2`` relative to the linear
        secant slope (de Boor, 1978; Eilers & Marx, 1996). Gated on REML fits so fixed-penalty models remain bit-compatible,
        and skipped inside Kronecker tensor products where the orthogonal margin penalty already stabilizes boundary magnitudes.
        """
        if not getattr(self, "_reml_fitted", False) or getattr(self, "_in_tensor_product", False):
            return 1.0
        if self.spline_degree >= 2 and self.penalty_order >= 2:
            return 2.0 / float(self.spline_degree)
        return 1.0
#: </ood_slope_scale>

    def _resolve_device_knots(self, knots: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
        if knots is self._cached_knots:
            dcache = getattr(self, "_device_knots_cache", None)
            if dcache is None:
                dcache = {}
                self._device_knots_cache = dcache
            key = (device, dtype)
            k_dev = dcache.get(key)
            if k_dev is None or k_dev.device != device or k_dev.dtype != dtype:
                k_dev = knots if (knots.device == device and knots.dtype == dtype) else knots.to(device=device, dtype=dtype)
                dcache[key] = k_dev
            return k_dev, True
        return knots.to(device=device, dtype=dtype), False

    def _get_active_span_tables(self, knots: torch.Tensor, degree: int, is_uniform: bool) -> tuple:
        if is_uniform:
            tcache = getattr(self, "_active_table_cache", None)
            if tcache is None:
                tcache = {}
                self._active_table_cache = tcache
            key = (id(knots), int(degree), knots.device, knots.dtype)
            entry = tcache.get(key)
            if entry is not None:
                return entry

        n_coeffs = int(knots.numel()) - (degree + 1)
        n_panels = n_coeffs - degree
        t_min_t = knots[degree]
        t_max_knot_t = knots[-(degree + 1)]
        t_max_clamp_t = t_max_knot_t - 1e-6
        inv_h = (float(n_panels) / max(float((t_max_knot_t - t_min_t).item()), 1e-12)) if (is_uniform and degree >= 1 and n_panels > 1) else 0.0
        panel_spans = degree + torch.arange(n_panels, device=knots.device, dtype=torch.long).unsqueeze(-1)
        tables = []
        for d in range(1, degree + 1):
            offsets = torch.arange(d, device=knots.device, dtype=torch.long)
            p_right = panel_spans + (1 + offsets)
            p_left = p_right - d
            pt_left = knots[p_left]
            pt_right = knots[p_right]
            p_dt = pt_right - pt_left
            p_inv_dt = torch.where(p_dt > 1e-8, 1.0 / p_dt.clamp_min(1e-8), torch.zeros_like(p_dt))
            tables.append((pt_left, pt_right, p_inv_dt))
        col_offsets = torch.arange(0, degree + 1, device=knots.device, dtype=torch.long)
        search_knots = None if (n_panels <= 1 or (is_uniform and degree >= 1)) else knots[degree + 1 : degree + n_panels].contiguous()
        entry = (n_coeffs, n_panels, t_min_t, t_max_clamp_t, inv_h, search_knots, tuple(tables), col_offsets)
        if is_uniform:
            tcache[key] = entry
        return entry

    def _eval_active_span_bases(self, x: torch.Tensor, knots: torch.Tensor, degree: int, is_uniform: bool = False) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        n_coeffs, n_panels, t_min_t, t_max_clamp_t, inv_h, search_knots, tables, col_offsets = self._get_active_span_tables(
            knots, degree, is_uniform=is_uniform
        )
        x_c = torch.clamp(x, min=t_min_t, max=t_max_clamp_t)
        if n_panels <= 1:
            span_rel = torch.zeros_like(x_c, dtype=torch.long)
        elif search_knots is None:
            span_rel = ((x_c - t_min_t) * inv_h).to(torch.long).clamp_(0, n_panels - 1)
        else:
            span_rel = torch.searchsorted(search_knots, x_c, right=True)

        bases = torch.ones((*x_c.shape, 1), dtype=x.dtype, device=x.device)
        x_u = x_c.unsqueeze(-1)
        flat_rel = span_rel.reshape(-1)
        for pt_left, pt_right, p_inv_dt in tables:
            d = int(pt_left.shape[-1])
            t_left = pt_left.index_select(0, flat_rel).view(*x_c.shape, d)
            t_right = pt_right.index_select(0, flat_rel).view(*x_c.shape, d)
            inv_dt = p_inv_dt.index_select(0, flat_rel).view(*x_c.shape, d)

            term = bases * inv_dt
            next_bases = torch.empty((*x_c.shape, d + 1), dtype=x.dtype, device=x.device)
            next_bases[..., :-1] = (t_right - x_u) * term
            next_bases[..., -1] = 0.0
            next_bases[..., 1:].add_((x_u - t_left) * term)
            bases = next_bases
        return span_rel, bases, n_coeffs, col_offsets

    def _cox_de_boor_active_span(self, x: torch.Tensor, knots: torch.Tensor, degree: int, is_uniform: bool = False) -> torch.Tensor:
        r"""Active-span B-spline evaluation in ``O(T k^2)`` independent of ``n_knots`` (Algorithm ``BSPLVB``, de Boor 1978, Ch. X)."""
        span_rel, bases, n_coeffs, col_offsets = self._eval_active_span_bases(x, knots, degree, is_uniform=is_uniform)
        out = torch.zeros((*x.shape, n_coeffs), dtype=x.dtype, device=x.device)
        out.scatter_(-1, span_rel.unsqueeze(-1) + col_offsets, bases)
        return out

    def _get_dense_inv_tables(self, knots: torch.Tensor, degree: int, is_uniform: bool) -> list:
        if is_uniform:
            dcache = getattr(self, "_dense_inv_cache", None)
            if dcache is None:
                dcache = {}
                self._dense_inv_cache = dcache
            key = (id(knots), int(degree), knots.device, knots.dtype)
            entry = dcache.get(key)
            if entry is not None:
                return entry
        steps = []
        for d in range(1, degree + 1):
            k_left_a, k_left_b = knots[d:-1], knots[:-d - 1]
            dt_left = k_left_a - k_left_b
            inv_left = torch.where(dt_left > 1e-8, 1.0 / dt_left.clamp_min(1e-8), torch.zeros_like(dt_left))
            k_right_a, k_right_b = knots[d + 1:], knots[1:-d]
            dt_right = k_right_a - k_right_b
            inv_right = torch.where(dt_right > 1e-8, 1.0 / dt_right.clamp_min(1e-8), torch.zeros_like(dt_right))
            steps.append((k_left_b, inv_left, k_right_a, inv_right))
        if is_uniform:
            dcache[key] = steps
        return steps

    def _cox_de_boor_core(self, x: torch.Tensor, knots: torch.Tensor, degree: int, is_uniform: bool = False) -> torch.Tensor:
        r"""Evaluates B-spline basis functions; dispatches to active-span ``BSPLVB`` (de Boor, 1978) whenever beneficial."""
        n_panels = int(knots.numel()) - 2 * degree - 1
        if n_panels >= 4 and x.numel() * int(knots.numel()) >= 4096:
            return self._cox_de_boor_active_span(x, knots, degree, is_uniform=is_uniform)

        x_exp = x.unsqueeze(-1)
        x_c = torch.clamp(x_exp, min=knots[degree], max=knots[-(degree + 1)] - 1e-6)
        bases = ((x_c >= knots[:-1]) & (x_c < knots[1:])).to(x.dtype)

        for k_l, inv_l, k_r, inv_r in self._get_dense_inv_tables(knots, degree, is_uniform=is_uniform):
            bases = (x_c - k_l) * (bases[..., :-1] * inv_l) + (k_r - x_c) * (bases[..., 1:] * inv_r)

        return bases

    def _cox_de_boor(self, x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
        r"""Evaluates B-spline basis with token deduplication on repeated inputs and active-span ``BSPLVB``."""
        knots_dev, is_uniform = self._resolve_device_knots(knots, x.device, x.dtype)
        return eval_1d_dedup(x, lambda xv: self._cox_de_boor_core(xv, knots_dev, degree, is_uniform=is_uniform))

#: <contract_coefficients>
    def contract_coefficients(self, x_col: torch.Tensor, c_slice: torch.Tensor) -> torch.Tensor:
        r"""Evaluates ``Phi(x_col) @ c_slice`` directly over the ``degree + 1`` active basis functions without allocating ``(G, T, K)``."""
        dev, dtype = c_slice.device, c_slice.dtype
        deg = self.spline_degree
        knots_dev, is_uniform = self._resolve_device_knots(self._cached_knots, dev, dtype)
        x_in = x_col.to(device=dev, dtype=dtype)
        g_out, _, d_out = c_slice.shape
        g_x = int(x_in.shape[0])

        is_ood = None
        x_bound = None
        if self.extrapolate == "continue":
            x_eval = x_in
        elif self.extrapolate == "periodic":
            x_eval = self._wrap_periodic(x_in)
        else:
            x_bound = x_in.clamp(min=-1.0, max=1.0)
            x_eval = x_bound
            if self.extrapolate in ("linear", "saturation"):
                is_ood = x_in != x_bound

        span_rel, bases, _, col_offsets = self._eval_active_span_bases(x_eval, knots_dev, deg, is_uniform=is_uniform)
        idx = span_rel.unsqueeze(-1) + col_offsets
        if g_x == 1 and g_out == 1 and d_out == 1:
            c_sel = c_slice[0, :, 0][idx]
            y = (bases * c_sel).sum(dim=-1, keepdim=True)
        elif g_x == 1:
            c_sel = c_slice[:, idx[0], :]
            y = (bases.unsqueeze(-1) * c_sel).sum(dim=-2)
        else:
            g_ar = torch.arange(g_out, device=dev).view(g_out, 1, 1)
            c_sel = c_slice[g_ar, idx]
            y = (bases.unsqueeze(-1) * c_sel).sum(dim=-2)

        if is_ood is not None and bool(is_ood.any()):
            eps = 1e-5
            scale = self._ood_slope_scale()
            if g_x == 1:
                _, t_i = torch.nonzero(is_ood, as_tuple=True)
                delta = x_in[0, t_i] - x_bound[0, t_i]
                x_eps = x_bound[0, t_i] - eps * torch.sign(delta)
                span_e, bases_e, _, _ = self._eval_active_span_bases(x_eps, knots_dev, deg, is_uniform=is_uniform)
                idx_e = span_e.unsqueeze(-1) + col_offsets
                c_sel_e = c_slice[:, idx_e, :]
                y_eps = (bases_e.unsqueeze(0).unsqueeze(-1) * c_sel_e).sum(dim=-2)
                slope_y = ((y[:, t_i, :] - y_eps) / eps) * scale
                dist = delta.abs()
                mult = dist if self.extrapolate == "linear" else (1.0 - torch.exp(-2.0 * dist)) / 2.0
                y[:, t_i, :] = y[:, t_i, :] + slope_y * mult.view(1, -1, 1)
            else:
                g_i, t_i = torch.nonzero(is_ood, as_tuple=True)
                delta = x_in[g_i, t_i] - x_bound[g_i, t_i]
                x_eps = x_bound[g_i, t_i] - eps * torch.sign(delta)
                span_e, bases_e, _, _ = self._eval_active_span_bases(x_eps, knots_dev, deg, is_uniform=is_uniform)
                idx_e = span_e.unsqueeze(-1) + col_offsets
                c_sel_e = c_slice[g_i.unsqueeze(-1), idx_e]
                y_eps = (bases_e.unsqueeze(-1) * c_sel_e).sum(dim=-2)
                slope_y = ((y[g_i, t_i, :] - y_eps) / eps) * scale
                dist = delta.abs()
                mult = dist if self.extrapolate == "linear" else (1.0 - torch.exp(-2.0 * dist)) / 2.0
                y[g_i, t_i, :] = y[g_i, t_i, :] + slope_y * mult.unsqueeze(-1)
        return y
#: </contract_coefficients>

    @staticmethod
    def contract_batched_splines(
        effects: list["SplineEffect"], x_batch: torch.Tensor, c_batch: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluates and contracts ``M`` homogeneous 1D splines in a single vectorized active-span pass."""
        dev, dtype = c_batch.device, c_batch.dtype
        anchor = effects[0]
        deg = anchor.spline_degree
        n_coeffs = anchor.get_n_coeffs()
        n_panels = n_coeffs - deg
        m_cnt = len(effects)
        extrap = anchor.extrapolate
        slope_scale = anchor._ood_slope_scale()

        cache = getattr(anchor, "_batched_table_cache", None)
        stacked = None
        if cache is not None:
            c_refs, c_dev, c_dtype, c_deg, payload = cache
            if (
                c_dev == dev
                and c_dtype == dtype
                and c_deg == deg
                and len(c_refs) == m_cnt
                and all(e._cached_knots is r for e, r in zip(effects, c_refs))
            ):
                stacked = payload

        if stacked is None:
            per_eff = [e._get_active_span_tables(e._resolve_device_knots(e._cached_knots, dev, dtype)[0], deg, is_uniform=True) for e in effects]
            t_min_1d = torch.stack([t[2] for t in per_eff])          # (M,)
            t_max_clamp_1d = torch.stack([t[3] for t in per_eff])    # (M,)
            inv_h_1d = torch.tensor([t[4] for t in per_eff], device=dev, dtype=dtype)  # (M,)
            flat_tables = tuple(
                (
                    torch.cat([t[6][step][0] for t in per_eff], dim=0).contiguous(),
                    torch.cat([t[6][step][1] for t in per_eff], dim=0).contiguous(),
                    torch.cat([t[6][step][2] for t in per_eff], dim=0).contiguous(),
                )
                for step in range(deg)
            )
            col_offsets = torch.arange(0, deg + 1, device=dev, dtype=torch.long)
            m_panel_offsets = torch.arange(m_cnt, device=dev, dtype=torch.long) * n_panels
            m_coeff_offsets = torch.arange(m_cnt, device=dev, dtype=torch.long) * n_coeffs
            refs = tuple(e._cached_knots for e in effects)
            stacked = (t_min_1d, t_max_clamp_1d, inv_h_1d, flat_tables, col_offsets, m_panel_offsets, m_coeff_offsets)
            anchor._batched_table_cache = (refs, dev, dtype, deg, stacked)

        t_min_1d, t_max_clamp_1d, inv_h_1d, flat_tables, col_offsets, m_panel_offsets, m_coeff_offsets = stacked
        x_in = x_batch.to(device=dev, dtype=dtype)
        g_x, t_dim, _ = x_in.shape
        g_out, _, _, d_out = c_batch.shape

        def _eval_m(x_vals: torch.Tensor, m_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            if m_idx is None:
                view_s = (*([1] * (x_vals.dim() - 1)), m_cnt)
                lo_t, hi_t, ih_t = t_min_1d.view(*view_s), t_max_clamp_1d.view(*view_s), inv_h_1d.view(*view_s)
                m_off = m_panel_offsets.view(*view_s)
            else:
                lo_t, hi_t, ih_t = t_min_1d[m_idx], t_max_clamp_1d[m_idx], inv_h_1d[m_idx]
                m_off = m_panel_offsets[m_idx]
            x_c = torch.maximum(torch.minimum(x_vals, hi_t), lo_t)
            if n_panels <= 1:
                span_rel = torch.zeros_like(x_c, dtype=torch.long)
            else:
                span_rel = ((x_c - lo_t) * ih_t).to(torch.long).clamp_(0, n_panels - 1)
            flat_idx = (span_rel + m_off).reshape(-1)
            bases = torch.ones((*x_c.shape, 1), dtype=dtype, device=dev)
            x_u = x_c.unsqueeze(-1)
            for pt_l, pt_r, p_idt in flat_tables:
                d = int(pt_l.shape[-1])
                tl = pt_l.index_select(0, flat_idx).view(*x_c.shape, d)
                tr = pt_r.index_select(0, flat_idx).view(*x_c.shape, d)
                idt = p_idt.index_select(0, flat_idx).view(*x_c.shape, d)
                term = bases * idt
                nb = torch.empty((*x_c.shape, d + 1), dtype=dtype, device=dev)
                nb[..., :-1] = (tr - x_u) * term
                nb[..., -1] = 0.0
                nb[..., 1:].add_((x_u - tl) * term)
                bases = nb
            return span_rel, bases

        is_ood = None
        x_bound = None
        if extrap == "continue":
            x_eval = x_in
        elif extrap == "periodic":
            x_eval = BaseEffect._wrap_periodic(x_in)
        else:
            x_bound = x_in.clamp(min=-1.0, max=1.0)
            x_eval = x_bound
            if extrap in ("linear", "saturation"):
                is_ood = x_in != x_bound

        span_rel, bases = _eval_m(x_eval)
        c_flat = c_batch.reshape(g_out, m_cnt * n_coeffs, d_out)
        c_idx = span_rel.unsqueeze(-1) + col_offsets.view(1, 1, 1, deg + 1) + m_coeff_offsets.view(1, 1, m_cnt, 1)
        has_ood = is_ood is not None and bool(is_ood.any())

        if not has_ood:
            if g_x == 1 and g_out == 1 and d_out == 1:
                c_sel = c_flat[0, :, 0][c_idx]
                return (bases * c_sel).sum(dim=(-2, -1)).unsqueeze(-1)
            if g_x == 1:
                c_sel = c_flat[:, c_idx[0], :]
                return (bases[0].unsqueeze(0).unsqueeze(-1) * c_sel).sum(dim=(-3, -2))
            g_ar = torch.arange(g_out, device=dev).view(g_out, 1, 1, 1)
            c_sel = c_flat[g_ar, c_idx]
            return (bases.unsqueeze(-1) * c_sel).sum(dim=(-3, -2))

        # Sparse linear / saturation extrapolation per OOD entry
        eps = 1e-5
        if g_x == 1:
            c_sel = c_flat[:, c_idx[0], :]  # (g_out, T, M, deg+1, d_out)
            y_terms = (bases[0].unsqueeze(0).unsqueeze(-1) * c_sel).sum(dim=-2)  # (g_out, T, M, d_out)
            _, t_i, m_i = torch.nonzero(is_ood, as_tuple=True)
            delta = x_in[0, t_i, m_i] - x_bound[0, t_i, m_i]
            x_eps = x_bound[0, t_i, m_i] - eps * torch.sign(delta)
            span_e, bases_e = _eval_m(x_eps, m_idx=m_i)
            idx_e = span_e.unsqueeze(-1) + col_offsets + m_coeff_offsets[m_i].unsqueeze(-1)
            c_sel_e = c_flat[:, idx_e, :]  # (g_out, N_ood, deg+1, d_out)
            y_eps = (bases_e.unsqueeze(0).unsqueeze(-1) * c_sel_e).sum(dim=-2)
            slope_y = ((y_terms[:, t_i, m_i, :] - y_eps) / eps) * slope_scale
            dist = delta.abs()
            mult = dist if extrap == "linear" else (1.0 - torch.exp(-2.0 * dist)) / 2.0
            y_terms[:, t_i, m_i, :] = y_terms[:, t_i, m_i, :] + slope_y * mult.view(1, -1, 1)
            return y_terms.sum(dim=2)

        g_ar = torch.arange(g_out, device=dev).view(g_out, 1, 1, 1)
        c_sel = c_flat[g_ar, c_idx]
        y_terms = (bases.unsqueeze(-1) * c_sel).sum(dim=-2)
        g_i, t_i, m_i = torch.nonzero(is_ood, as_tuple=True)
        delta = x_in[g_i, t_i, m_i] - x_bound[g_i, t_i, m_i]
        x_eps = x_bound[g_i, t_i, m_i] - eps * torch.sign(delta)
        span_e, bases_e = _eval_m(x_eps, m_idx=m_i)
        idx_e = span_e.unsqueeze(-1) + col_offsets + m_coeff_offsets[m_i].unsqueeze(-1)
        c_sel_e = c_flat[g_i.unsqueeze(-1), idx_e]
        y_eps = (bases_e.unsqueeze(-1) * c_sel_e).sum(dim=-2)
        slope_y = ((y_terms[g_i, t_i, m_i, :] - y_eps) / eps) * slope_scale
        dist = delta.abs()
        mult = dist if extrap == "linear" else (1.0 - torch.exp(-2.0 * dist)) / 2.0
        y_terms[g_i, t_i, m_i, :] = y_terms[g_i, t_i, m_i, :] + slope_y * mult.unsqueeze(-1)
        return y_terms.sum(dim=2)
#: </cox_de_boor>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""Builds the pure-PyTorch B-spline design matrix."""
        x_shape = x_col.shape

        is_dummy = (x_col.shape[-1] == 1 if x_col.dim() > 0 else False)
        
        knots = self._get_knots(x_col, is_dummy=is_dummy)
        phi_matrix = self._cox_de_boor(x_col, knots, self.spline_degree)
        
        return phi_matrix.reshape(*x_shape, -1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds the finite difference penalty matrix entirely on GPU."""
        n_coeffs = self.get_n_coeffs()
        
        I = torch.eye(n_coeffs, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        
        if self.penalty_order == 0:
            D = I
        else:
            D = torch.diff(I, n=self.penalty_order, dim=0)

        if self.m_pen == 1:
            M_star_M = self.lambda_p * (D.T @ D)
            return M_star_M
        weights = self._part_weights(D.shape[0])
        lambdas = self.penalty_coordinates()
        return sum(lambdas[r] * (D.T @ (weights[:, r:r + 1] * D)) for r in range(self.m_pen))
#: </penalty_matrix>