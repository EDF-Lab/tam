# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Restricted marginal likelihood (Laplace REML) criterion for the smoothing parameters of ``StaticTAM.auto_fit(criterion='reml')``.

Treating the coefficients as a Gaussian random effect with precision ``T_eff P(rho)``, ``P = P_0 + sum_c exp(rho_c) P_c`` and integrating them out gives, for a
(working) system ``Phi'W Phi``, ``Phi'W z`` and ``z'W z``, per group,

    l(rho) = -1/2 log|A| + 1/2 log|P|_+ - (T_eff - M0)/2 log(phi_hat)        A = Phi'W Phi + T_eff P + ridge,   phi_hat = D_p / (T_eff - M0),

with ``D_p = z'W z - theta' Phi'W z`` the penalized residual sum of squares at ``theta = A^-1 Phi'W z``, ``M0`` the dimension of the null space of the penalty and
``|P|_+`` the determinant on the penalized subspace. With a known scale ``phi`` (Poisson, binomial) the last term is ``-D_p / (2 phi)``. The gradient in ``rho``
is analytic (``d log|A| = tr(A^-1 dP)``, ``d D_p = theta' dP theta`` by the envelope theorem). The groups add up.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .spectrum import build_phi_from_effects

_TINY = 1e-300
_NULL_TOLERANCE = 1e-9


#: <penalty_basis>
class PenaltyBasis:
    r"""The penalty of the effects as ``P_0 + sum_c lambda_c P_c``: one ``P_c`` per smoothing coordinate, found from the effects themselves.

    Every penalty of the library is linear in its smoothing weights, so ``P_c`` is the difference of the penalty with the weight ``c`` at 2 and at 1 (the
    other weights at 1), and ``P_0`` is what remains (a fixed ridge, for instance). ``U`` spans the penalized subspace (the range of ``P_0 + sum_c P_c``,
    the same whatever the positive weights), of dimension ``rank``; the null space has dimension ``null_dim``.
    """

    def __init__(self, effects_list):
        dtype = torch.float64
        spans, start = [], 0
        for effect in effects_list:
            width = effect.get_n_coeffs()
            spans.append((start, start + width))
            start += width
        self.total = start
        self.blocks: List[Tuple[int, int, torch.Tensor]] = []  # (start, end, Q_c block)
        self.constant = torch.zeros(self.total, self.total, dtype=dtype)
        effect_local: List[Tuple[int, int, int, int, torch.Tensor, List[torch.Tensor]]] = []
        for effect, (begin, end) in zip(effects_list, spans):
            saved = effect.penalty_coordinates()
            count = len(saved)
            c_start = len(self.blocks)
            reference = self._block(effect, [1.0] * count, dtype)
            summed = torch.zeros_like(reference)
            local_qs: List[torch.Tensor] = []
            for c in range(count):
                weights = [1.0] * count
                weights[c] = 2.0
                q = self._block(effect, weights, dtype) - reference
                self.blocks.append((begin, end, q))
                local_qs.append(q)
                summed = summed + q
            effect.set_penalty_coordinates(saved)
            const_e = reference - summed
            self.constant[begin:end, begin:end] = const_e
            effect_local.append((begin, end, c_start, len(self.blocks), const_e, local_qs))
        self.n_coordinates = len(self.blocks)

        # Per-effect block spectral analysis avoids O(D^3) global eigendecomposition (Wood 2011; Wood, Goude & Shaw 2015)
        raw_blocks = []
        global_max = 0.0
        for begin, end, c_start, c_end, const_e, local_qs in effect_local:
            unit_e = const_e.clone()
            for q in local_qs:
                unit_e = unit_e + q
            dim_e = unit_e.shape[0]
            if dim_e == 0:
                continue
            off_diag_e = unit_e - torch.diag(torch.diagonal(unit_e))
            unit_scale = max(float(unit_e.abs().max()), 1.0)
            if dim_e <= 1 or float(off_diag_e.abs().max()) <= 1e-14 * unit_scale:
                vals_e = torch.diagonal(unit_e)
                vecs_e = None
            else:
                vals_e, vecs_e = torch.linalg.eigh(0.5 * (unit_e + unit_e.T))
            block_max = float(vals_e.abs().max())
            if block_max > global_max:
                global_max = block_max
            raw_blocks.append((begin, end, c_start, c_end, const_e, local_qs, vals_e, vecs_e))

        global_tol = _NULL_TOLERANCE * max(global_max, _TINY)
        self.rank = 0
        self._subspace_blocks: List[Tuple[int, int, torch.Tensor]] = []
        self._basis_cache: Optional[torch.Tensor] = None
        self._reduced_constant_cache: Optional[torch.Tensor] = None
        self._reduced_cache: Optional[List[torch.Tensor]] = None

        # Block-structured spectral representations for O(1) / O(r_e) log|P|_+ and d(log|P|_+)/drho (Wood 2011)
        self._scalar_terms: List[Tuple[int, int, float]] = []
        self._commuting_groups: List[Tuple[List[int], np.ndarray, List[np.ndarray]]] = []
        self._dense_groups: List[Tuple[List[int], torch.Tensor, List[torch.Tensor]]] = []
        self._const_group_logs: float = 0.0
        self._const_group_rank: int = 0
        self._const_sign_ok: bool = True

        for begin, end, c_start, c_end, const_e, local_qs, vals_e, vecs_e in raw_blocks:
            keep_e = vals_e > global_tol
            r_e = int(keep_e.sum())
            if r_e == 0:
                continue
            self.rank += r_e
            coords_e = list(range(c_start, c_end))
            u_e = torch.eye(end - begin, dtype=dtype)[:, keep_e] if vecs_e is None else vecs_e[:, keep_e]
            self._subspace_blocks.append((begin, end, u_e))
            const_norm = float(const_e.abs().max()) if const_e.numel() > 0 else 0.0

            if len(coords_e) == 0:
                const_red = u_e.T @ (0.5 * (const_e + const_e.T)) @ u_e
                s_c, ld_c = torch.linalg.slogdet(const_red)
                if float(s_c) <= 0.0:
                    self._const_sign_ok = False
                else:
                    self._const_group_logs += float(ld_c)
                    self._const_group_rank += r_e
                continue

            if len(coords_e) == 1 and const_norm <= global_tol:
                pos_vals = vals_e[keep_e].clamp_min(_TINY)
                self._scalar_terms.append((coords_e[0], r_e, float(torch.log(pos_vals).sum())))
                continue

            const_red = u_e.T @ (0.5 * (const_e + const_e.T)) @ u_e
            q_reds = [u_e.T @ (0.5 * (q + q.T)) @ u_e for q in local_qs]

            # Test simultaneous diagonalizability (exact for Kronecker sums in TensorProductEffect)
            comb = 0.37 * const_red
            for idx_m, qr in enumerate(q_reds):
                comb = comb + (1.0 + 0.17 * (idx_m + 1)) * qr
            _, eigvecs = torch.linalg.eigh(0.5 * (comb + comb.T))
            all_mats = [const_red] + q_reds
            commutes = True
            diags: List[np.ndarray] = []
            for mat in all_mats:
                proj = eigvecs.T @ mat @ eigvecs
                diag_vec = torch.diagonal(proj)
                recon = eigvecs @ torch.diag(diag_vec) @ eigvecs.T
                scale = max(float(torch.linalg.norm(mat)), 1e-12)
                if float(torch.linalg.norm(recon - mat)) > 1e-8 * scale:
                    commutes = False
                    break
                diags.append(np.maximum(diag_vec.detach().cpu().numpy().astype(np.float64), 0.0))

            if commutes:
                self._commuting_groups.append((coords_e, diags[0], diags[1:]))
            else:
                self._dense_groups.append((coords_e, const_red, q_reds))

        self.null_dim = self.total - self.rank

    def _ensure_global_subspace(self) -> None:
        if self._basis_cache is not None:
            return
        basis_mat = torch.zeros(self.total, self.rank, dtype=torch.float64)
        col = 0
        for begin, end, u_e in self._subspace_blocks:
            width = u_e.shape[1]
            basis_mat[begin:end, col:col + width] = u_e
            col += width
        self._basis_cache = basis_mat
        self._reduced_constant_cache = basis_mat.T @ self.constant @ basis_mat
        self._reduced_cache = [basis_mat[b:e].T @ q @ basis_mat[b:e] for b, e, q in self.blocks]

    @property
    def basis(self) -> torch.Tensor:
        self._ensure_global_subspace()
        return self._basis_cache

    @property
    def _reduced_constant(self) -> torch.Tensor:
        self._ensure_global_subspace()
        return self._reduced_constant_cache

    @property
    def _reduced(self) -> List[torch.Tensor]:
        self._ensure_global_subspace()
        return self._reduced_cache

    @staticmethod
    def _block(effect, weights, dtype) -> torch.Tensor:
        effect.set_penalty_coordinates(weights)
        matrix = effect.build_penalty_matrix()
        if matrix.is_sparse:
            matrix = matrix.to_dense()
        return matrix.detach().to(device="cpu", dtype=dtype)

    def penalty(self, rho) -> torch.Tensor:
        r"""``P_0 + sum_c exp(rho_c) P_c`` (without the factor ``T_eff``)."""
        result = self.constant.clone()
        for weight, (begin, end, q) in zip(np.exp(np.asarray(rho, dtype=float)), self.blocks):
            result[begin:end, begin:end] += float(weight) * q
        return result

    def reduced_penalty(self, rho) -> torch.Tensor:
        r"""The penalty on the penalized subspace, ``U' P U``."""
        result = self._reduced_constant.clone()
        for weight, reduced in zip(np.exp(np.asarray(rho, dtype=float)), self._reduced):
            result = result + float(weight) * reduced
        return result

    def block_logdet_and_traces(
        self,
        lam: np.ndarray,
        rho: np.ndarray,
        n: float,
        compute_grad: bool = True,
        hess_out: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        r"""O(1)/O(r_e) block-spectral computation of ``log|n P(rho)|_+`` and ``tr((n P)^+ n dP_c)``.

        ``hess_out`` (m, m), when given with ``compute_grad``, is filled with ``tr((nP)^+ T_c (nP)^+ T_k)`` where ``T_c = n lam_c P_c``.
        """
        if not self._const_sign_ok:
            return False, 0.0, None
        log_n = float(np.log(n))
        log_det_s = (self._const_group_rank * log_n + self._const_group_logs) if self._const_group_rank > 0 else 0.0
        traces = np.zeros(self.n_coordinates, dtype=np.float64) if compute_grad else None
        if hess_out is not None:
            hess_out[:] = 0.0

        for c, r_c, ldet_q in self._scalar_terms:
            log_det_s += r_c * (log_n + float(rho[c])) + ldet_q
            if compute_grad:
                traces[c] = float(r_c)
                if hess_out is not None:
                    hess_out[c, c] = float(r_c)

        for coords_e, nu_0, nu_list in self._commuting_groups:
            mu_e = nu_0.copy()
            for c, nu_m in zip(coords_e, nu_list):
                mu_e += float(lam[c]) * nu_m
            mu_e *= n
            if not np.all(mu_e > 0.0):
                return False, 0.0, None
            log_det_s += float(np.sum(np.log(mu_e)))
            if compute_grad:
                inv_mu = 1.0 / mu_e
                for c, nu_m in zip(coords_e, nu_list):
                    traces[c] = float(np.sum((n * float(lam[c])) * nu_m * inv_mu))
                if hess_out is not None:
                    shares = np.stack([(n * float(lam[c])) * nu_m * inv_mu for c, nu_m in zip(coords_e, nu_list)])
                    hess_out[np.ix_(coords_e, coords_e)] = shares @ shares.T

        for coords_e, const_red, q_reds in self._dense_groups:
            mat_e = const_red.clone()
            for c, qr in zip(coords_e, q_reds):
                mat_e = mat_e + float(lam[c]) * qr
            mat_e = n * mat_e
            sign_e, ldet_e = torch.linalg.slogdet(mat_e)
            if float(sign_e) <= 0.0:
                return False, 0.0, None
            log_det_s += float(ldet_e)
            if compute_grad:
                inv_e = torch.linalg.inv(mat_e)
                for c, qr in zip(coords_e, q_reds):
                    traces[c] = float((inv_e * (n * float(lam[c]) * qr).T).sum())
                if hess_out is not None:
                    shares = [inv_e @ (n * float(lam[c]) * qr) for c, qr in zip(coords_e, q_reds)]
                    for a, c in enumerate(coords_e):
                        for b, k in enumerate(coords_e):
                            if b < a:
                                continue
                            value = float((shares[a] * shares[b].T).sum())
                            hess_out[c, k] = value
                            hess_out[k, c] = value

        return True, log_det_s, traces
#: </penalty_basis>


def _prepare_cached_blocks(
    basis: PenaltyBasis,
    n_samples: int,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    cov_x: Optional[torch.Tensor] = None,
):
    r"""Precomputes device penalty blocks with 1D diagonal flags and optional invariant base system ``Phi'W Phi + n P_0 + ridge``."""
    const_dev = basis.constant.to(device=device, dtype=dtype)
    blocks_dev = []
    for begin, end, q in basis.blocks:
        q_dev = q.to(device=device, dtype=dtype)
        diag_q = torch.diagonal(q_dev).contiguous()
        is_diag = bool(torch.equal(q_dev, torch.diag(diag_q)))
        blocks_dev.append((begin, end, q_dev, is_diag, diag_q))
    ridge = 1e-6 * float(n_samples) * torch.eye(size, dtype=dtype, device=device)
    base_system = (cov_x + float(n_samples) * const_dev + ridge) if cov_x is not None else None
    all_diag = bool(blocks_dev) and all(b[3] for b in blocks_dev)
    seen_spans = {(b, e) for b, e, _, is_diag, _ in blocks_dev if not is_diag}
    dense_trace_cost = sum((size - b) * (e - b) ** 2 for b, e in seen_spans)
    use_tri_inv = bool(blocks_dev) and size >= 32 and (all_diag or dense_trace_cost <= 0.65 * (size ** 3) / 3.0)
    eye_d = torch.eye(size, dtype=dtype, device=device) if use_tri_inv else None
    return (const_dev, blocks_dev, ridge, base_system, use_tri_inv, eye_d)


#: <reml_hessian>
def _reml_hessian(beta, inverse, blocks_dev, lam, n, trace_all, quad_all, phi_tilde, dof, unknown_scale, hess_s, grad_f):
    r"""Analytic Hessian in ``rho`` of minus the REML criterion: ``diag(g) + G/2 hS - 1/2 sum_g tr(A^-1 T_c A^-1 T_k) - sum_g u_c'v_k / phi_g - [unknown scale] sum_g q_c q_k / (2 nu phi_g^2)``.

    ``T_c = n lam_c P_c``, ``v_c = T_c theta``, ``u_c = A^-1 v_c``, ``q_c = theta' v_c``. The trace matrix is built from the columns ``Z_c = A^-1[:, block_c] T_c`` of all coordinates side by side.
    """
    groups, size = beta.shape[0], beta.shape[1]
    m = len(blocks_dev)
    z_parts, rows, owner = [], [], []
    u_cols, v_cols = [], []
    for c, (begin, end, q_dev, is_diag, diag_q) in enumerate(blocks_dev):
        scale = n * float(lam[c])
        inv_cols = inverse[:, :, begin:end]
        beta_block = beta[:, begin:end, :]
        if is_diag:
            weight = scale * diag_q
            z = inv_cols * weight
            v_block = beta_block * weight.view(1, -1, 1)
        else:
            weight = scale * q_dev
            z = inv_cols @ weight
            v_block = weight @ beta_block
        z_parts.append(z)
        rows.extend(range(begin, end))
        owner.extend([c] * (end - begin))
        u_cols.append(z @ beta_block)  # (G, D, 1)
        v_full = torch.zeros(groups, size, 1, dtype=beta.dtype, device=beta.device)
        v_full[:, begin:end, :] = v_block
        v_cols.append(v_full)
    z_all = torch.cat(z_parts, dim=2)  # (G, D, W)
    width = z_all.shape[2]
    rows_t = torch.tensor(rows, dtype=torch.long, device=beta.device)
    owner_t = torch.tensor(owner, dtype=torch.long, device=beta.device)
    pair = torch.zeros(width, width, dtype=beta.dtype, device=beta.device)
    step = max(1, int(64 * 1024 * 1024 // max(width * width * 8, 1)))
    for g0 in range(0, groups, step):
        r = z_all[g0:g0 + step][:, rows_t, :]
        pair += (r * r.transpose(1, 2)).sum(dim=0)
    rows_sum = torch.zeros(m, width, dtype=beta.dtype, device=beta.device).index_add_(0, owner_t, pair)
    trace_pair = torch.zeros(m, m, dtype=beta.dtype, device=beta.device).index_add_(1, owner_t, rows_sum)
    u_mat = torch.cat(u_cols, dim=2)  # (G, D, m)
    v_mat = torch.cat(v_cols, dim=2)
    uv = (u_mat.transpose(1, 2) @ v_mat) / phi_tilde.view(-1, 1, 1)  # (G, m, m)
    hess = np.diag(np.asarray(grad_f, dtype=np.float64)) + 0.5 * groups * hess_s
    hess -= 0.5 * trace_pair.cpu().numpy() + uv.sum(dim=0).cpu().numpy()
    if unknown_scale:
        hess -= (torch.einsum("gc,gk->ck", quad_all, quad_all / (phi_tilde ** 2).unsqueeze(1)) / (2.0 * dof)).cpu().numpy()
    return 0.5 * (hess + hess.T)

#: </reml_hessian>


#: <reml_objective>
def reml_objective(
    rho,
    cov_x: torch.Tensor,
    cov_xy: torch.Tensor,
    y_sq: torch.Tensor,
    basis: PenaltyBasis,
    n_samples: int,
    known_scale: Optional[float] = None,
    compute_grad: bool = True,
    _cached_device_blocks=None,
    compute_hess: bool = False,
    hess_out: Optional[np.ndarray] = None,
    efs_out: Optional[dict] = None,
) -> Tuple[float, np.ndarray]:
    r"""Minus the REML criterion summed over the groups and its gradient in ``rho = log(lambda)``.

    Args:
        rho: log smoothing weights, one per coordinate of ``basis``.
        cov_x, cov_xy, y_sq: ``Phi'W Phi`` (G, D, D), ``Phi'W z`` (G, D, 1) and ``z'W z`` (G,), float64.
        basis: the penalty of the effects.
        n_samples: rows of a group, ``T_eff`` (the scale of the penalty, as in the fit).
        known_scale: the scale ``phi`` when it is known (Poisson, binomial); estimated from the penalized residual sum of squares otherwise.
        compute_grad: if ``False``, skips ``cholesky_inverse`` and returns an empty gradient array (faster for coarse probing).
        compute_hess, hess_out: with ``compute_grad``, fill ``hess_out`` (m, m) with the analytic Hessian of the returned criterion in ``rho`` (the return value is unchanged).
        efs_out: when a dict, receives ``num`` and ``den`` of the fixed-point update ``rho_c += log(num_c / den_c)`` (it is the zero of the gradient).
    """
    rho = np.asarray(rho, dtype=float)
    n = float(n_samples)
    lam = np.exp(rho)
    size = cov_x.shape[-1]
    if _cached_device_blocks is not None:
        if len(_cached_device_blocks) == 6:
            const_dev, blocks_dev, ridge, base_system, all_diag, eye_d = _cached_device_blocks
        elif len(_cached_device_blocks) == 4:
            const_dev, blocks_dev, ridge, base_system = _cached_device_blocks
            all_diag = bool(blocks_dev) and all(b[3] for b in blocks_dev)
            eye_d = torch.eye(size, dtype=cov_x.dtype, device=cov_x.device) if all_diag and size >= 32 else None
        else:
            const_dev, blocks_dev, ridge = _cached_device_blocks
            base_system = None
            all_diag = bool(blocks_dev) and all(b[3] for b in blocks_dev)
            eye_d = torch.eye(size, dtype=cov_x.dtype, device=cov_x.device) if all_diag and size >= 32 else None
    else:
        const_dev, blocks_dev, ridge, base_system, all_diag, eye_d = _prepare_cached_blocks(
            basis, n_samples, size, cov_x.dtype, cov_x.device, cov_x
        )

    hess_s = np.zeros((basis.n_coordinates, basis.n_coordinates)) if (compute_hess and compute_grad) else None
    sign_ok, log_det_s, trace_s_arr = basis.block_logdet_and_traces(lam, rho, n, compute_grad=compute_grad, hess_out=hess_s)
    if not sign_ok:
        empty_g = np.zeros(basis.n_coordinates, dtype=np.float64) if compute_grad else np.empty(0, dtype=np.float64)
        return float("inf"), empty_g

    if base_system is not None and base_system.shape[0] == 1:
        system = base_system.clone()
        sys0 = system[0]
        sys0_diag = sys0.diagonal()
        for weight, block in zip(lam, blocks_dev):
            begin, end, q_dev, is_diag, diag_q = block
            nw = n * float(weight)
            if is_diag:
                sys0_diag[begin:end].add_(diag_q, alpha=nw)
            else:
                sys0[begin:end, begin:end].add_(q_dev, alpha=nw)
    else:
        penalty = const_dev.clone()
        penalty_diag = penalty.diagonal()
        for weight, block in zip(lam, blocks_dev):
            begin, end, q_dev, is_diag, diag_q = block
            w_f = float(weight)
            if is_diag:
                penalty_diag[begin:end].add_(diag_q, alpha=w_f)
            else:
                penalty[begin:end, begin:end].add_(q_dev, alpha=w_f)
        system = (cov_x + n * penalty + ridge) if base_system is None else (base_system + n * (penalty - const_dev))
    factor = torch.linalg.cholesky(system)
    beta = torch.cholesky_solve(cov_xy, factor)
    log_det_a = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)  # (G,)
    deviance = (y_sq - (beta * cov_xy).sum(dim=(1, 2))).clamp_min(1e-300)  # D_p, (G,)

    dof = n - basis.null_dim
    if known_scale is None:
        value = -0.5 * log_det_a + 0.5 * log_det_s - 0.5 * dof * torch.log(deviance / dof)
    else:
        value = -0.5 * log_det_a + 0.5 * log_det_s - deviance / (2.0 * known_scale)
    if not compute_grad:
        return -float(value.sum()), np.empty(0, dtype=np.float64)

    if eye_d is not None:
        l_inv = torch.linalg.solve_triangular(factor, eye_d.expand_as(factor), upper=False)
        inv_diag_full = (l_inv * l_inv).sum(dim=-2) if any(b[3] for b in blocks_dev) else None
        inv_blocks = {}
        inverse = None
    else:
        inverse = torch.cholesky_inverse(factor)  # (G, D, D)
        l_inv = None
        inv_diag_full = None
        inv_blocks = None
    gradient = np.zeros(basis.n_coordinates)
    groups = float(cov_x.shape[0])
    keep_parts = hess_s is not None or efs_out is not None
    trace_parts, quad_parts = [], []
    for c, block in enumerate(blocks_dev):
        begin, end, q_dev, is_diag, diag_q = block
        scale = n * float(lam[c])
        beta_block = beta[:, begin:end, 0]
        if is_diag:
            inv_diag = inv_diag_full[:, begin:end] if inv_diag_full is not None else torch.diagonal(inverse[:, begin:end, begin:end], dim1=-2, dim2=-1)
            trace_a = (inv_diag * diag_q).sum(dim=-1) * scale  # tr(A^-1 dP_c), (G,)
            quad = (beta_block.square() * diag_q).sum(dim=-1) * scale
        else:
            if l_inv is not None:
                inv_b = inv_blocks.get((begin, end))
                if inv_b is None:
                    m_sub = l_inv[:, begin:, begin:end]
                    inv_b = m_sub.mT @ m_sub
                    inv_blocks[(begin, end)] = inv_b
                trace_a = (inv_b * q_dev.T).sum(dim=(1, 2)) * scale  # tr(A^-1 dP_c) without dense inverse
            else:
                trace_a = (inverse[:, begin:end, begin:end] * q_dev.T).sum(dim=(1, 2)) * scale  # tr(A^-1 dP_c), (G,)
            quad = ((beta_block @ q_dev) * beta_block).sum(dim=-1) * scale  # theta' dP_c theta
        if known_scale is None:
            scale_term = -dof / (2.0 * deviance) * quad
        else:
            scale_term = -quad / (2.0 * known_scale)
        gradient[c] = float((-0.5 * trace_a + scale_term).sum()) + 0.5 * groups * trace_s_arr[c]
        if keep_parts:
            trace_parts.append(trace_a)
            quad_parts.append(quad)
    if keep_parts:
        phi_tilde = deviance / dof if known_scale is None else torch.full_like(deviance, float(known_scale))
        trace_all, quad_all = torch.stack(trace_parts, dim=1), torch.stack(quad_parts, dim=1)  # (G, m)
        if efs_out is not None:
            efs_out["num"] = groups * np.asarray(trace_s_arr) - trace_all.sum(dim=0).cpu().numpy()
            efs_out["den"] = (quad_all / phi_tilde.unsqueeze(1)).sum(dim=0).cpu().numpy()
    if hess_s is not None:
        if inverse is None:
            inverse = torch.cholesky_inverse(factor)
        import time as _time

        t_hess = _time.perf_counter()
        hess_out[:] = _reml_hessian(
            beta, inverse, blocks_dev, lam, n, trace_all, quad_all, phi_tilde, dof, known_scale is None, hess_s, -gradient
        )
        if efs_out is not None:
            efs_out["hess_seconds"] = _time.perf_counter() - t_hess
    return -float(value.sum()), -gradient
#: </reml_objective>


#: <reml_newton_search>
def _efs_step(num, den):
    """Fixed-point step of the log smoothing weights: the zero of the gradient is ``num == den``."""
    step = np.zeros_like(num)
    both = (num > 1e-12) & (den > 1e-12)
    step[both] = np.clip(np.log(num[both] / den[both]), -5.0, 5.0)
    step[(num > 1e-12) & (den <= 1e-12)] = 5.0
    return step


def _solve_reml_efs_newton(start, evaluate, value_only, low_rho, high_rho, m):
    """Few fixed-point steps, then a safeguarded Newton search with the analytic Hessian over the active coordinates.

    ``evaluate(rho, hess)`` returns value, gradient, Hessian (or ``None``), the fixed-point terms and the seconds spent on the Hessian;
    ``value_only(rho)`` returns the value alone (1e300 when the system is not positive definite).
    """
    import time

    t0 = time.perf_counter()
    info = {"efs_steps": 0, "hessian_evals": 0, "newton_steps": 0, "rejected": 0, "frozen": 0, "hess_seconds": 0.0}
    x = np.array(start, dtype=np.float64)
    f = value_only(x)
    best = (f, x.copy())
    reason = "newton_step_limit"
    iterations = 0
    if high_rho - low_rho < 1e-12:  # one admissible value: nothing to search
        info["solver_seconds"] = time.perf_counter() - t0
        out = type("Result", (), {})()
        out.x, out.fun, out.nit = best[1], best[0], 0
        return out, "newton_bounds", info

    for _ in range(3):  # fixed-point warm start
        _, _, _, efs_terms, _ = evaluate(x, False)
        step = _efs_step(efs_terms["num"], efs_terms["den"])
        if float(np.max(np.abs(step))) < 0.25:
            break
        accepted, gain, scale = False, 0.0, 1.0
        for _ in range(4):
            trial = np.clip(x + scale * step, low_rho, high_rho)
            f_t = value_only(trial)
            if f_t < f:
                gain, x, f, accepted = f - f_t, trial, f_t, True
                break
            scale *= 0.5
        if not accepted:
            break
        info["efs_steps"] += 1
        iterations += 1
        if f < best[0]:
            best = (f, x.copy())
        if gain < 1.0:
            break

    converged = False
    for _ in range(30):
        iterations += 1
        f, g, hess, _, hess_seconds = evaluate(x, True)
        info["hessian_evals"] += 1
        info["hess_seconds"] += hess_seconds
        if f < best[0]:
            best = (f, x.copy())
        diag_scale = max(float(np.max(np.abs(np.diag(hess)))), 1e-300)
        out_low = (x <= low_rho + 1e-10) & (g > 0.0)
        out_high = (x >= high_rho - 1e-10) & (g < 0.0)
        flat = (np.abs(np.diag(hess)) < 1e-6 * diag_scale) & (np.abs(g) < 1e-5)  # re-entered as soon as the gradient grows
        idx = np.flatnonzero(~(out_low | out_high | flat))
        info["frozen"] = int(m - len(idx))
        if len(idx) == 0:
            reason = "newton_bounds"
            break
        ga = g[idx]
        values, vectors = np.linalg.eigh(hess[np.ix_(idx, idx)])
        scale = max(float(np.max(np.abs(values))), 1e-300)
        proj = vectors.T @ ga
        decrement = 0.5 * float(np.sum(proj ** 2 / np.maximum(np.abs(values), 1e-6 * scale)))
        if values[0] > -1e-3 * scale and decrement < 0.05 and float(np.max(np.abs(ga))) < 1e-3:  # curvature below that is a flat ridge, not a descent direction
            reason = "newton_ok"
            break
        direction = np.zeros(m)
        direction[idx] = -(vectors @ (proj / np.maximum(np.abs(values), 1e-6 * scale)))
        top = float(np.max(np.abs(direction)))
        if top > 5.0:
            direction *= 5.0 / top
        floor = 1e-10 * max(abs(f), 1.0)
        accepted = False
        for attempt in range(2):
            t, halvings = 1.0, 12 if attempt == 0 else 8
            for halving in range(halvings + 1):
                trial = np.clip(x + t * direction, low_rho, high_rho)
                predicted = -float(g @ (trial - x))
                if predicted < floor:  # the model predicts nothing left to gain
                    if halving == 0:
                        converged = True
                    break
                f_t = value_only(trial)
                if f_t <= f - 1e-4 * predicted and f_t < f:
                    x, f, accepted = trial, f_t, True
                    break
                if t * float(np.max(np.abs(direction))) < 1e-6:
                    break
                t *= 0.5
            if accepted or converged:
                break
            info["rejected"] += 1
            direction = np.zeros(m)  # one capped steepest-descent step
            direction[idx] = -ga
            top = float(np.max(np.abs(direction)))
            if top > 2.0:
                direction *= 2.0 / top
        if converged:
            reason = "newton_ok"
            break
        if not accepted:
            reason = "newton_step_limit"
            break
        info["newton_steps"] += 1
        if f < best[0]:
            best = (f, x.copy())
    else:
        reason = "newton_step_limit"
    info["solver_seconds"] = time.perf_counter() - t0
    out = type("Result", (), {})()
    out.x, out.fun, out.nit = best[1], best[0], iterations
    return out, reason, info

#: </reml_newton_search>


def reml_working_system(x_data, y_data, effects_list, loss_matrix, sample_weights, design_matrix, basis):
    r"""``Phi'W Phi`` (G, D, D), ``Phi'W z`` (G, D, 1) and ``z'W z`` (G,) in float64 of a (working) system, on the cached design matrix when there is one."""
    from ._math import (
        _compute_weighted_covariances,
        _is_rowwise_stateless_effects,
        _streamed_weighted_covariances,
    )

    n_groups, n_samples = x_data.shape[0], x_data.shape[1]
    loss = loss_matrix.to(torch.float64)
    structured = getattr(design_matrix, "_tam_structured", None) if design_matrix is not None else None
    if structured is not None and loss.shape == (1, 1):
        y_all = y_data.to(device=structured.device, dtype=torch.float64)
        w_all = None if sample_weights is None else sample_weights.to(device=structured.device, dtype=torch.float64)
        cov_x_dev, cov_xy_dev, y_sq_dev = structured.weighted_covariances(
            y_all, loss.to(structured.device), w_all, dtype=torch.float64, return_y_sq=True
        )
        cov_x, cov_xy, y_sq = cov_x_dev.cpu(), cov_xy_dev.cpu(), y_sq_dev.cpu()
    elif design_matrix is not None:
        phi = design_matrix.to(dtype=torch.float64)
        y_all = y_data.to(device=phi.device, dtype=torch.float64)
        w_all = None if sample_weights is None else sample_weights.to(device=phi.device, dtype=torch.float64)
        loss_dev = loss.to(device=phi.device)
        cov_x_dev, cov_xy_dev = _compute_weighted_covariances(phi, y_all, loss_dev, w_all)
        cov_x = cov_x_dev.cpu()
        cov_xy = cov_xy_dev.cpu()
        weighted = y_all if w_all is None else y_all * w_all.clamp_min(0).sqrt()
        y_sq = ((weighted ** 2 * loss_dev[0, 0]).sum(dim=(1, 2))).cpu()
    else:
        can_stream = (
            n_samples * basis.total * 8 > 512 * 1024 * 1024
            and loss.shape == (1, 1)
            and _is_rowwise_stateless_effects(effects_list)
        )
        if can_stream:
            y_all = y_data.to(torch.float64)
            w_all = None if sample_weights is None else sample_weights.to(torch.float64)
            cov_x_dev, cov_xy_dev, y_sq_dev = _streamed_weighted_covariances(
                x_data, y_all, effects_list, loss, sample_weights=w_all, tile_size=16384, dtype=torch.float64, return_y_sq=True
            )
            cov_x, cov_xy, y_sq = cov_x_dev.cpu(), cov_xy_dev.cpu(), y_sq_dev.cpu()
        else:
            covs_x, covs_xy, y_sq_list = [], [], []
            for g in range(n_groups):
                phi = build_phi_from_effects(x_data[g:g + 1], effects_list).to(torch.float64)
                y = y_data[g:g + 1].to(device=phi.device, dtype=torch.float64)
                w = None if sample_weights is None else sample_weights[g:g + 1].to(device=phi.device, dtype=torch.float64)
                loss_dev = loss.to(device=phi.device)
                c_x, c_xy = _compute_weighted_covariances(phi, y, loss_dev, w)
                covs_x.append(c_x.cpu())
                covs_xy.append(c_xy.cpu())
                weighted = y.to(c_x.device) if w is None else y.to(c_x.device) * w.to(c_x.device).clamp_min(0).sqrt()
                y_sq_list.append(((weighted ** 2 * loss_dev[0, 0].to(c_x.device)).sum()).cpu())
            cov_x, cov_xy = torch.cat(covs_x, dim=0), torch.cat(covs_xy, dim=0)
            y_sq = torch.stack(y_sq_list)
    return cov_x, cov_xy, y_sq


#: <smart_solve_reml>
def smart_solve_reml(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    effects_list,
    loss_matrix: torch.Tensor,
    alpha_p_bounds: Tuple[float, float],
    alpha_p_list=None,
    sample_weights: Optional[torch.Tensor] = None,
    known_scale: Optional[float] = None,
    verbose: bool = False,
    init_rho: Optional[np.ndarray] = None,
    design_matrix: Optional[torch.Tensor] = None,
    basis: Optional[PenaltyBasis] = None,
    stop_info: Optional[dict] = None,
):
    r"""Choose the smoothing weights of every coordinate by maximising the REML criterion in ``log lambda``.

    The best of a grid of start points (a Cholesky screen of all candidates in float64) goes to a short fixed-point warm start and a safeguarded Newton search with the analytic
    Hessian of the criterion (``_solve_reml_efs_newton``). ``stop_info``, when a dict, receives ``reason`` (``"newton_ok"``, ``"newton_bounds"`` or ``"newton_step_limit"``),
    ``n_iter`` and the counts and times of the search (fixed-point steps, Newton steps, Hessian evaluations, seconds in the Hessian, the solver and the screen).

    Applies analytical log-determinant gradients and generalized-inverse penalty subspace factorization
    (Wood, 2011; Wood & Fasiolo, 2017), caching Cholesky factors and reusing precomputed designs across PQL steps.
    """
    from ._math import solve_linear_system

    n_groups, n_samples = x_data.shape[0], x_data.shape[1]
    if basis is None:
        basis = PenaltyBasis(effects_list)
    if basis.total > 7500:
        raise MemoryError(f"Feature dimension D={basis.total} is too massive for the REML criterion (dense Cholesky of the {basis.total} x {basis.total} system).")
    cov_x, cov_xy, y_sq = reml_working_system(x_data, y_data, effects_list, loss_matrix, sample_weights, design_matrix, basis)

    cached_device_blocks = _prepare_cached_blocks(basis, n_samples, cov_x.shape[-1], cov_x.dtype, cov_x.device, cov_x)

    low, high = (min(alpha_p_list), max(alpha_p_list)) if alpha_p_list is not None else alpha_p_bounds
    low_rho, high_rho = low * np.log(10.0), high * np.log(10.0)

    def objective_val(rho_vec):
        val, _ = reml_objective(
            rho_vec, cov_x, cov_xy, y_sq, basis, n_samples, known_scale,
            compute_grad=False, _cached_device_blocks=cached_device_blocks,
        )
        return val if np.isfinite(val) else 1e300

    rho_formula = np.array(
        [np.log(max(float(c), 1e-300)) for e in effects_list for c in e.penalty_coordinates()],
        dtype=np.float64,
    )
    # Trace-ratio balanced initial coordinates equalize tr(X_j'WX_j) vs n * lam_c * tr(Q_c) (Wood & Fasiolo, 2017)
    cov_mean = cov_x.mean(dim=0)
    cov_diag = torch.diagonal(cov_mean)
    rho_offset = np.zeros(basis.n_coordinates, dtype=np.float64)
    for c, block in enumerate(cached_device_blocks[1]):
        begin, end, _, _, diag_q = block
        tr_cov_c = float(cov_diag[begin:end].sum())
        tr_q_c = float(diag_q.sum())
        if tr_q_c > 1e-12:
            rho_offset[c] = np.log(max(tr_cov_c, 1e-12) / (float(n_samples) * tr_q_c))

    candidates: List[np.ndarray] = []
    if init_rho is None:
        grid = np.linspace(max(low_rho, -12.0 * np.log(10.0)), min(high_rho, 4.0 * np.log(10.0)), 6)
        for v in grid:
            candidates.append(np.full(basis.n_coordinates, v, dtype=np.float64))
        candidates.append(np.clip(rho_formula, low_rho, high_rho))
        for alpha_shift in (-4.0, -2.0, 0.0, 2.0):
            candidates.append(np.clip(rho_offset + alpha_shift * np.log(10.0), low_rho, high_rho))
    else:
        candidates.append(np.clip(np.asarray(init_rho, dtype=np.float64), low_rho, high_rho))
        candidates.append(np.clip(rho_formula, low_rho, high_rho))
        for alpha_shift in (-2.0, 0.0, 2.0):
            candidates.append(np.clip(rho_offset + alpha_shift * np.log(10.0), low_rho, high_rho))
        for alpha_fallback in (-2.0, 1.0):
            v = min(max(alpha_fallback * np.log(10.0), low_rho), high_rho)
            candidates.append(np.full(basis.n_coordinates, v, dtype=np.float64))

    def _score_candidates_batched(cands: List[np.ndarray]) -> List[Tuple[float, int, np.ndarray]]:
        const_dev, blocks_dev, ridge_dev = cached_device_blocks[:3]
        n_f = float(n_samples)
        valid_meta = []
        scores = [1e300] * len(cands)
        penalties = []
        for idx, cand in enumerate(cands):
            rho_c = np.asarray(cand, dtype=float)
            lam_c = np.exp(rho_c)
            sign_ok, log_det_s, _ = basis.block_logdet_and_traces(lam_c, rho_c, n_f, compute_grad=False)
            if not sign_ok:
                continue
            p_c = const_dev.clone()
            p_diag = p_c.diagonal()
            for w, (b, e, q_d, is_diag, diag_q) in zip(lam_c, blocks_dev):
                w_f = float(w)
                if is_diag:
                    p_diag[b:e].add_(diag_q, alpha=w_f)
                else:
                    p_c[b:e, b:e].add_(q_d, alpha=w_f)
            penalties.append(n_f * p_c)
            valid_meta.append((idx, float(log_det_s)))

        if not valid_meta:
            return [(1e300, i, cands[i]) for i in range(len(cands))]

        size_d = cov_x.shape[-1]
        mem_est = len(valid_meta) * n_groups * size_d * size_d * 8
        if len(valid_meta) > 1 and mem_est <= 256 * 1024 * 1024:
            p_stack = torch.stack(penalties, dim=0).unsqueeze(1)
            systems = cov_x.unsqueeze(0) + p_stack + ridge_dev.unsqueeze(0).unsqueeze(0)
            dof = n_f - basis.null_dim
            factors, info = torch.linalg.cholesky_ex(systems)
            if int(info.max().item()) == 0:
                rhs = cov_xy.unsqueeze(0).expand(len(valid_meta), -1, -1, -1)
                betas = torch.cholesky_solve(rhs, factors)
                log_det_a = 2.0 * torch.log(torch.diagonal(factors, dim1=-2, dim2=-1)).sum(dim=-1)
                deviance = (y_sq.unsqueeze(0) - (betas * rhs).sum(dim=(-2, -1))).clamp_min(1e-300)
                ldet_s_t = torch.tensor([m[1] for m in valid_meta], dtype=cov_x.dtype, device=cov_x.device).unsqueeze(1)
                if known_scale is None:
                    vals = -0.5 * log_det_a + 0.5 * ldet_s_t - 0.5 * dof * torch.log(deviance / dof)
                else:
                    vals = -0.5 * log_det_a + 0.5 * ldet_s_t - deviance / (2.0 * known_scale)
                totals = (-vals.sum(dim=-1)).tolist()
                for (orig_idx, _), total in zip(valid_meta, totals):
                    scores[orig_idx] = total if np.isfinite(total) else 1e300
                return [(scores[i], i, cands[i]) for i in range(len(cands))]

        return [(objective_val(cand), idx, cand) for idx, cand in enumerate(cands)]

    import time as _clock

    screen_start = _clock.perf_counter()
    scored = _score_candidates_batched(candidates)
    screen_seconds = _clock.perf_counter() - screen_start
    best_start_val, _, start = min(scored, key=lambda item: (item[0], item[1]))

    def evaluate(rho_vec, with_hess):
        m_c = basis.n_coordinates
        hess = np.zeros((m_c, m_c)) if with_hess else None
        terms: dict = {}
        value, gradient = reml_objective(
            rho_vec, cov_x, cov_xy, y_sq, basis, n_samples, known_scale,
            compute_grad=True, _cached_device_blocks=cached_device_blocks,
            compute_hess=with_hess, hess_out=hess, efs_out=terms,
        )
        if not np.isfinite(value):
            return 1e300, np.zeros(m_c), (np.eye(m_c) if with_hess else None), {"num": np.zeros(m_c), "den": np.zeros(m_c)}, 0.0
        return float(value), np.asarray(gradient, dtype=np.float64), hess, terms, float(terms.get("hess_seconds", 0.0))

    result, stop_reason, search_info = _solve_reml_efs_newton(
        np.array(start, dtype=np.float64), evaluate, objective_val, low_rho, high_rho, basis.n_coordinates
    )
    if stop_info is not None:
        stop_info.update(reason=stop_reason, n_iter=int(result.nit), screen_seconds=screen_seconds, **search_info)
    use_opt = float(result.fun) <= best_start_val
    rho = result.x if use_opt else start
    best_val = float(result.fun) if use_opt else float(best_start_val)
    lam = np.exp(rho)
    position = 0
    for effect in effects_list:
        count = effect.n_penalty_coordinates
        effect.set_penalty_coordinates([float(v) for v in lam[position:position + count]])
        effect._reml_fitted = True
        position += count
    penalty = basis.penalty(rho).to(torch.get_default_dtype())
    coefficients = solve_linear_system(cov_x.to(torch.get_default_dtype()), cov_xy.to(torch.get_default_dtype()), penalty, n_samples)
    if verbose:
        print(f"[REML] {result.nit} iterations ({stop_reason}), criterion {-best_val:.4f}")
    return coefficients, lam, -best_val
#: </smart_solve_reml>
