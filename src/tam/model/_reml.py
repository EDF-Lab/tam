# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Restricted marginal likelihood (Laplace REML) criterion for the smoothing parameters of ``StaticTAM.auto_fit(criterion='reml')``.

Treating the coefficients as a Gaussian random effect with precision ``n S(rho)``, ``S = S0 + sum_c exp(rho_c) Q_c`` and integrating them out gives, for a
(working) system ``X'WX``, ``X'Wy`` and ``y'Wy``, per group,

    l(rho) = -1/2 log|A| + 1/2 log|S|_+ - (n - M0)/2 log(phi_hat)        A = X'WX + n S + ridge,   phi_hat = D_p / (n - M0),

with ``D_p = y'Wy - beta' X'Wy`` the penalized residual sum of squares at ``beta = A^-1 X'Wy``, ``M0`` the dimension of the null space of the penalty and
``|S|_+`` the determinant on the penalized subspace. With a known scale ``phi`` (Poisson, binomial) the last term is ``-D_p / (2 phi)``. The gradient in ``rho``
is analytic (``d log|A| = tr(A^-1 dS)``, ``d D_p = beta' dS beta`` by the envelope theorem). The groups add up.
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
    r"""The penalty of the effects as ``S0 + sum_c lambda_c Q_c``: one ``Q_c`` per smoothing coordinate, found from the effects themselves.

    Every penalty of the library is linear in its smoothing weights, so ``Q_c`` is the difference of the penalty with the weight ``c`` at 2 and at 1 (the
    other weights at 1), and ``S0`` is what remains (a fixed ridge, for instance). ``U`` spans the penalized subspace (the range of ``S0 + sum_c Q_c``,
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
        for effect, (begin, end) in zip(effects_list, spans):
            saved = effect.penalty_coordinates()
            count = len(saved)
            reference = self._block(effect, [1.0] * count, dtype)
            summed = torch.zeros_like(reference)
            for c in range(count):
                weights = [1.0] * count
                weights[c] = 2.0
                q = self._block(effect, weights, dtype) - reference
                self.blocks.append((begin, end, q))
                summed = summed + q
            effect.set_penalty_coordinates(saved)
            self.constant[begin:end, begin:end] = reference - summed
        self.n_coordinates = len(self.blocks)
        unit = self.constant.clone()
        for begin, end, q in self.blocks:
            unit[begin:end, begin:end] += q
        values, vectors = torch.linalg.eigh(0.5 * (unit + unit.T))
        keep = values > _NULL_TOLERANCE * float(values.abs().max().clamp_min(_TINY))
        self.basis = vectors[:, keep]
        self.rank = int(keep.sum())
        self.null_dim = self.total - self.rank
        self._reduced_constant = self.basis.T @ self.constant @ self.basis
        self._reduced = [self.basis[b:e].T @ q @ self.basis[b:e] for b, e, q in self.blocks]

    @staticmethod
    def _block(effect, weights, dtype) -> torch.Tensor:
        effect.set_penalty_coordinates(weights)
        matrix = effect.build_penalty_matrix()
        if matrix.is_sparse:
            matrix = matrix.to_dense()
        return matrix.detach().to(device="cpu", dtype=dtype)

    def penalty(self, rho) -> torch.Tensor:
        r"""``S0 + sum_c exp(rho_c) Q_c`` (without the factor ``n``)."""
        result = self.constant.clone()
        for weight, (begin, end, q) in zip(np.exp(np.asarray(rho, dtype=float)), self.blocks):
            result[begin:end, begin:end] += float(weight) * q
        return result

    def reduced_penalty(self, rho) -> torch.Tensor:
        r"""The penalty on the penalized subspace, ``U' S U``."""
        result = self._reduced_constant.clone()
        for weight, reduced in zip(np.exp(np.asarray(rho, dtype=float)), self._reduced):
            result = result + float(weight) * reduced
        return result
#: </penalty_basis>


#: <reml_objective>
def reml_objective(
    rho,
    cov_x: torch.Tensor,
    cov_xy: torch.Tensor,
    y_sq: torch.Tensor,
    basis: PenaltyBasis,
    n_samples: int,
    known_scale: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    r"""Minus the REML criterion summed over the groups and its gradient in ``rho = log(lambda)``.

    Args:
        rho: log smoothing weights, one per coordinate of ``basis``.
        cov_x, cov_xy, y_sq: ``X'WX`` (G, D, D), ``X'Wy`` (G, D, 1) and ``y'Wy`` (G,), float64.
        basis: the penalty of the effects.
        n_samples: rows of a group (the scale of the penalty, as in the fit).
        known_scale: the scale ``phi`` when it is known (Poisson, binomial); estimated from the penalized residual sum of squares otherwise.
    """
    rho = np.asarray(rho, dtype=float)
    n = float(n_samples)
    lam = np.exp(rho)
    groups, size = cov_x.shape[0], cov_x.shape[-1]
    penalty = n * basis.penalty(rho).to(cov_x.device)
    ridge = 1e-6 * n * torch.eye(size, dtype=cov_x.dtype, device=cov_x.device)
    system = cov_x + penalty + ridge
    factor = torch.linalg.cholesky(system)
    beta = torch.cholesky_solve(cov_xy, factor)
    log_det_a = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)  # (G,)
    inverse = torch.cholesky_inverse(factor)  # (G, D, D)
    deviance = (y_sq - (beta * cov_xy).sum(dim=(1, 2))).clamp_min(1e-300)  # D_p, (G,)

    reduced = n * basis.reduced_penalty(rho)
    sign, log_det_s = torch.linalg.slogdet(reduced)
    reduced_inverse = torch.linalg.inv(reduced)
    dof = n - basis.null_dim
    if known_scale is None:
        value = -0.5 * log_det_a + 0.5 * log_det_s - 0.5 * dof * torch.log(deviance / dof)
    else:
        value = -0.5 * log_det_a + 0.5 * log_det_s - deviance / (2.0 * known_scale)
    gradient = np.zeros(basis.n_coordinates)
    for c, ((begin, end, q), reduced_q) in enumerate(zip(basis.blocks, basis._reduced)):
        scaled_q = n * lam[c] * q.to(cov_x.device)
        trace_a = (inverse[:, begin:end, begin:end] * scaled_q.T).sum(dim=(1, 2))  # tr(A^-1 dS_c), (G,)
        trace_s = (reduced_inverse * (n * lam[c] * reduced_q).T).sum()
        beta_block = beta[:, begin:end, 0]
        quad = torch.einsum("gi,ij,gj->g", beta_block, scaled_q, beta_block)  # beta' dS_c beta
        if known_scale is None:
            scale_term = -dof / (2.0 * deviance) * quad
        else:
            scale_term = -quad / (2.0 * known_scale)
        gradient[c] = float((-0.5 * trace_a + 0.5 * trace_s + scale_term).sum())
    return -float(value.sum()) if sign > 0 else float("inf"), -gradient
#: </reml_objective>


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
):
    r"""Choose the smoothing weights of every coordinate by maximising the REML criterion (L-BFGS-B on ``log lambda``).

    Same contract as ``smart_solve_gcv``: returns ``(coefficients, best_lambda_ps, score)``, with the weights set on the effects; ``score`` is the REML
    criterion (larger is better). The search starts from the best common weight of a coarse grid. ``alpha_p_list`` only narrows the range searched.
    """
    from scipy.optimize import minimize
    from ._math import _compute_weighted_covariances, solve_linear_system

    n_groups, n_samples = x_data.shape[0], x_data.shape[1]
    basis = PenaltyBasis(effects_list)
    if basis.total > 7500:
        raise MemoryError(f"Feature dimension D={basis.total} is too massive for the REML criterion (dense Cholesky of the {basis.total} x {basis.total} system).")
    covs_x, covs_xy, y_sq = [], [], []
    for g in range(n_groups):
        phi = build_phi_from_effects(x_data[g:g + 1], effects_list).to(torch.float64).cpu()
        y = y_data[g:g + 1].to(torch.float64).cpu()
        w = None if sample_weights is None else sample_weights[g:g + 1].to(torch.float64).cpu()
        loss = loss_matrix.to(torch.float64).cpu()
        cov_x, cov_xy = _compute_weighted_covariances(phi, y, loss, w)
        covs_x.append(cov_x.cpu())
        covs_xy.append(cov_xy.cpu())
        weighted = y if w is None else y * w.clamp_min(0).sqrt()
        y_sq.append((weighted ** 2 * loss[0, 0]).sum())
    cov_x, cov_xy = torch.cat(covs_x, dim=0), torch.cat(covs_xy, dim=0)
    y_sq = torch.stack(y_sq)

    low, high = (min(alpha_p_list), max(alpha_p_list)) if alpha_p_list is not None else alpha_p_bounds
    low_rho, high_rho = low * np.log(10.0), high * np.log(10.0)

    def objective(rho):
        value, gradient = reml_objective(rho, cov_x, cov_xy, y_sq, basis, n_samples, known_scale)
        if not np.isfinite(value):
            return 1e300, np.zeros_like(gradient)
        return value, gradient

    grid = np.linspace(max(low_rho, -16.0 * np.log(10.0)), min(high_rho, 6.0 * np.log(10.0)), 12)
    starts = [(objective(np.full(basis.n_coordinates, v))[0], v) for v in grid]
    start = np.full(basis.n_coordinates, min(starts)[1])
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(low_rho, high_rho)] * basis.n_coordinates)
    rho = result.x if result.fun <= min(starts)[0] else start
    lam = np.exp(rho)
    position = 0
    for effect in effects_list:
        count = effect.n_penalty_coordinates
        effect.set_penalty_coordinates([float(v) for v in lam[position:position + count]])
        position += count
    penalty = basis.penalty(rho).to(torch.get_default_dtype())
    coefficients = solve_linear_system(cov_x.to(torch.get_default_dtype()), cov_xy.to(torch.get_default_dtype()), penalty, n_samples)
    if verbose:
        print(f"[REML] {result.nit} iterations, criterion {-result.fun:.4f}")
    return coefficients, lam, float(-min(result.fun, min(starts)[0]))
#: </smart_solve_reml>
