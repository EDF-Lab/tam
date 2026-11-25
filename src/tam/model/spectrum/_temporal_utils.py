# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Shared helpers of the causal temporal spectrum effects.

Pure functions, no state: device and dtype follow the input. A temporal effect reads its input as
``(B, T, D)`` (B groups, T time steps, D columns), carries a causal history between two calls, and never mixes
two groups. The helpers below give every such effect the same layout handling, the same history padding and the
same linear recurrence, so that the effects differ only by their feature map.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

from scipy.fft import next_fast_len
from scipy.signal import lfilter
import torch


def to_btd(x: torch.Tensor, multi_input: bool) -> Tuple[torch.Tensor, Tuple[int, ...], str]:
    r"""Brings an effect input to ``(B, T, D)``; returns ``(x3, original shape, layout)``.

    ``(B, T, D)`` is kept; ``(B, T)`` becomes ``(B, T, 1)``; ``(N, D)`` (a tensor-product margin or an out-of-range
    probe, read with ``multi_input=True``) becomes one group ``(1, N, D)``; ``(T,)`` becomes ``(1, T, 1)``.
    """
    if x.dim() == 3:
        return x, tuple(x.shape), "btd"
    if x.dim() == 2:
        if multi_input:
            return x.unsqueeze(0), tuple(x.shape), "nd"
        return x.unsqueeze(-1), tuple(x.shape), "bt"
    if x.dim() == 1:
        return x.reshape(1, -1, 1), tuple(x.shape), "t"
    raise ValueError(f"A temporal effect reads a (B, T, D), (B, T), (N, D) or (T,) input, got {x.dim()} dimensions.")


def from_btk(phi: torch.Tensor, orig_shape: Tuple[int, ...], layout: str) -> torch.Tensor:
    r"""Inverse of :func:`to_btd` for a feature map ``(B, T, K)``: ``(B, T, K)``, ``(N, K)`` or ``(T, K)`` as the input was."""
    if layout in ("btd", "bt"):
        return phi
    if layout in ("nd", "t"):
        return phi[0]
    raise ValueError(f"Unknown layout '{layout}'.")


def causal_history(x3: torch.Tensor, n_hist: int, state_in: Optional[Dict[str, Any]], key: str = "hist") -> torch.Tensor:
    r"""Prepends ``n_hist`` past steps to ``x3``: the carried state when given, else the first row replicated.

    Returns ``(B, n_hist + T, D)``. The state must have shape ``(B, n_hist, D)``: continuing needs the same groups and the
    same columns as the pass that produced it.
    """
    if n_hist == 0:
        return x3
    B, _, D = x3.shape
    if state_in is not None and key in state_in:
        h = state_in[key].to(device=x3.device, dtype=x3.dtype)
        if tuple(h.shape) != (B, n_hist, D):
            raise ValueError(
                f"state '{key}' has shape {tuple(h.shape)}, expected {(B, n_hist, D)}: "
                "continue_state needs the same groups and features as training"
            )
        return torch.cat([h, x3], dim=1)
    return torch.cat([x3[:, :1].expand(B, n_hist, D), x3], dim=1)


def tail_state(x_hist: torch.Tensor, n_hist: int) -> torch.Tensor:
    r"""The last ``n_hist`` steps of ``x_hist``, detached: the state to carry to the next call."""
    if n_hist == 0:
        return x_hist[:, :0]
    return x_hist[:, -n_hist:].detach().clone()


def causal_window_unfold(x_hist: torch.Tensor, w: int, t_len: Optional[int] = None) -> torch.Tensor:
    r"""Sliding windows of length ``w`` over ``x_hist`` ``(B, L, D)``: ``(B, n, w, D)`` with ``n = L - w + 1``.

    With ``t_len`` the last ``t_len`` windows are returned (the window ``t`` ends at the step ``t`` of the current pass).
    That needs a history of at least ``w - 1`` steps before the pass.
    """
    L = x_hist.shape[1]
    if L < w:
        raise ValueError(f"the history ({L} steps) is shorter than the window ({w})")
    windows = x_hist.unfold(1, w, 1).permute(0, 1, 3, 2)
    if t_len is not None:
        if windows.shape[1] < t_len:
            raise ValueError(f"the history gives {windows.shape[1]} windows, {t_len} are needed: it must hold at least w - 1 steps before the pass")
        windows = windows[:, windows.shape[1] - t_len:]
    return windows


def _as_channel(v: Union[torch.Tensor, float], C: int, ref: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float64, device=ref.device)
    return t.expand(C).clone() if t.dim() == 0 else t.reshape(C)


def causal_linear_recurrence(
    u: torch.Tensor,
    a: Union[torch.Tensor, float],
    b: Union[torch.Tensor, float],
    y_prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Solves :math:`y_t = a\,y_{t-1} + b\,u_t` along time for every group and channel.

    ``u`` is ``(B, T, C)``; ``a`` and ``b`` are scalars or ``(C,)``; ``y_prev`` is ``(B, C)`` (zeros when None), the value
    of :math:`y_{-1}`. Needs :math:`|a| \le 1`. Up to 63 steps the recurrence is a loop; beyond, it is the convolution of
    ``u`` with the kernel :math:`b\,a^s` by FFT, in :math:`O(T \log T)` time and :math:`O(T)` memory. Float32 input is
    computed in float64 and cast back.
    """
    B, T, C = u.shape
    a_t = _as_channel(a, C, u)
    b_t = _as_channel(b, C, u)
    if bool((a_t.abs() > 1.0).any()):
        raise ValueError("causal_linear_recurrence needs |a| <= 1")
    work = torch.float64
    u64 = u.to(work)
    y0 = torch.zeros(B, C, dtype=work, device=u.device) if y_prev is None else y_prev.to(device=u.device, dtype=work)
    if T < 64:
        y = y0
        out = []
        for t in range(T):
            y = a_t * y + b_t * u64[:, t]
            out.append(y)
        res = torch.stack(out, dim=1) if out else u64[:, :0]
    elif bool((a_t == 1.0).all()):
        res = torch.cumsum(u64 * b_t.view(1, 1, C), dim=1) + y0[:, None, :]
    elif u.device.type == "cpu" and bool((a_t == a_t[0]).all()) and bool((b_t == b_t[0]).all()):
        a0 = float(a_t[0].item())
        b0 = float(b_t[0].item())
        u_np = u64.detach().numpy()
        zi = (a0 * y0).detach().numpy()[:, None, :]
        out_np, _ = lfilter([b0], [1.0, -a0], u_np, axis=1, zi=zi)
        res = torch.from_numpy(out_np)
    elif u.device.type == "cpu" and C <= 64:
        u_np = u64.detach().numpy()
        a_np = a_t.detach().numpy()
        b_np = b_t.detach().numpy()
        y0_np = y0.detach().numpy()
        out_np = torch.empty_like(u64).numpy()
        for c in range(C):
            ac = float(a_np[c])
            bc = float(b_np[c])
            out_np[:, :, c], _ = lfilter([bc], [1.0, -ac], u_np[:, :, c], axis=1, zi=(ac * y0_np[:, c])[:, None])
        res = torch.from_numpy(out_np)
    else:
        s = torch.arange(T, dtype=work, device=u.device)
        n = int(next_fast_len(2 * T - 1, real=True))
        u_t = u64.transpose(1, 2).contiguous()                                # (B, C, T)
        u_spec = torch.fft.rfft(u_t, n, dim=-1)                                # (B, C, n // 2 + 1)
        if bool((a_t == a_t[0]).all()) and bool((b_t == b_t[0]).all()):
            a0 = a_t[0]
            b0 = b_t[0]
            kernel_1 = b0 * torch.pow(a0, s)                                   # (T,)
            kernel_spec = torch.fft.rfft(kernel_1, n).view(1, 1, -1)
            carry_1 = torch.pow(a0, s + 1.0).view(1, T, 1)
            forced_response = torch.fft.irfft(u_spec * kernel_spec, n, dim=-1)[..., :T].transpose(1, 2)
            res = forced_response + carry_1 * y0[:, None, :]
        else:
            kernel = b_t[:, None] * torch.pow(a_t[:, None], s[None, :])        # (C, T)
            kernel_spec = torch.fft.rfft(kernel, n, dim=-1).unsqueeze(0)       # (1, C, n // 2 + 1)
            forced_response = torch.fft.irfft(u_spec * kernel_spec, n, dim=-1)[..., :T].transpose(1, 2)
            homogeneous_carry = torch.pow(a_t[:, None], (s + 1.0)[None, :]).T  # (T, C)
            res = forced_response + homogeneous_carry[None] * y0[:, None, :]
    return res.to(u.dtype)


def eval_1d_dedup(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    r"""Evaluates a row-wise 1D basis function ``fn`` once per unique value when repeated inputs dominate."""
    if x.numel() >= 8192:
        x_flat = x.reshape(-1)
        step = max(1, x_flat.numel() // 256)
        if torch.unique(x_flat[::step]).numel() <= 128:
            uniq, inv = torch.unique(x_flat, return_inverse=True)
            if uniq.numel() * 4 <= x_flat.numel():
                phi_uniq = fn(uniq)
                return phi_uniq.index_select(0, inv).view(*x.shape, phi_uniq.shape[-1])
    return fn(x)


def _cox_de_boor_core(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    r"""Evaluates B-spline basis functions; dispatches large supports to active-span ``BSPLVB`` (de Boor, 1978)."""
    n_coeffs = knots.numel() - (degree + 1)
    n_panels = n_coeffs - degree
    if degree >= 1 and n_panels >= 1 and x.numel() * knots.numel() >= 262144:
        t_min = knots[degree]
        t_max_knot = knots[-(degree + 1)]
        x_c = torch.clamp(x, min=t_min, max=t_max_knot - 1e-6)
        if n_panels <= 1:
            span_rel = torch.zeros_like(x_c, dtype=torch.long)
        else:
            panel_knots = knots[degree + 1 : degree + n_panels].contiguous()
            span_rel = torch.searchsorted(panel_knots, x_c, right=True).clamp_(0, n_panels - 1)
        panel_spans = degree + torch.arange(n_panels, device=x.device, dtype=torch.long).unsqueeze(-1)
        bases = torch.ones((*x_c.shape, 1), dtype=x.dtype, device=x.device)
        x_u = x_c.unsqueeze(-1)
        flat_rel = span_rel.reshape(-1)
        for d in range(1, degree + 1):
            offsets = torch.arange(d, device=x.device, dtype=torch.long)
            p_right = panel_spans + (1 + offsets)
            p_left = p_right - d
            pt_left = knots[p_left]
            pt_right = knots[p_right]
            p_dt = pt_right - pt_left
            p_inv_dt = torch.where(p_dt > 1e-8, 1.0 / p_dt.clamp_min(1e-8), torch.zeros_like(p_dt))

            t_left = pt_left.index_select(0, flat_rel).view(*x_c.shape, d)
            t_right = pt_right.index_select(0, flat_rel).view(*x_c.shape, d)
            inv_dt = p_inv_dt.index_select(0, flat_rel).view(*x_c.shape, d)

            term = bases * inv_dt
            next_bases = torch.empty((*x_c.shape, d + 1), dtype=x.dtype, device=x.device)
            next_bases[..., :-1] = (t_right - x_u) * term
            next_bases[..., -1] = 0.0
            next_bases[..., 1:].add_((x_u - t_left) * term)
            bases = next_bases

        out = torch.zeros((*x_c.shape, n_coeffs), dtype=x.dtype, device=x.device)
        col_offsets = torch.arange(0, degree + 1, device=x.device, dtype=torch.long)
        out.scatter_(-1, span_rel.unsqueeze(-1) + col_offsets, bases)
        return out

    x_exp = x.unsqueeze(-1)
    x_c = torch.clamp(x_exp, min=knots[degree], max=knots[-(degree + 1)] - 1e-6)
    bases = ((x_c >= knots[:-1]) & (x_c < knots[1:])).to(x.dtype)

    for d in range(1, degree + 1):
        dt_left = knots[d:-1] - knots[:-d - 1]
        inv_left = torch.where(dt_left > 1e-8, 1.0 / dt_left.clamp_min(1e-8), torch.zeros_like(dt_left))
        dt_right = knots[d + 1:] - knots[1:-d]
        inv_right = torch.where(dt_right > 1e-8, 1.0 / dt_right.clamp_min(1e-8), torch.zeros_like(dt_right))
        bases = ((x_c - knots[:-d - 1]) * inv_left) * bases[..., :-1] + (
            (knots[d + 1:] - x_c) * inv_right
        ) * bases[..., 1:]

    return bases


def cox_de_boor(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluates the B-spline basis of a knot vector by active-span BSPLVB and vectorized Cox-de Boor recursion.

    ``x`` has shape ``(N,)`` and the result ``(N, len(knots) - degree - 1)``; the input is clamped to the knot range.
    """
    knots = knots.to(device=x.device, dtype=x.dtype)
    return eval_1d_dedup(x, lambda xv: _cox_de_boor_core(xv, knots, degree))


def build_clamped_knots(k: int, degree: int, device: torch.device, dtype: torch.dtype, x_min: float = -1.0, x_max: float = 1.0) -> torch.Tensor:
    """Clamped B-spline knots on ``[x_min, x_max]``: ``degree + 1`` repeated end knots and ``k - 1`` equispaced interior knots, ``k + degree`` basis columns."""
    internal = torch.linspace(x_min, x_max, k + 1, device=device, dtype=dtype)[1:-1]
    t_left = torch.full((degree + 1,), x_min, device=device, dtype=dtype)
    t_right = torch.full((degree + 1,), x_max, device=device, dtype=dtype)
    return torch.cat([t_left, internal, t_right])


def eval_clamped_spline_with_linear_tails(x_norm: torch.Tensor, knots: torch.Tensor, degree: int, eps: float = 1e-3) -> torch.Tensor:
    """B-spline basis of ``x_norm`` (shape ``(N,)``, scaled to ``[-1, 1]``) with a linear continuation beyond the ends.

    Outside ``[-1, 1]`` the row is the value at the end plus the one-sided slope at the end (a forward difference of step ``eps``) times the distance:
    the extrapolated function is the straight line tangent to the spline at the boundary. Returns ``(N, len(knots) - degree - 1)``.
    """
    device, dtype = x_norm.device, x_norm.dtype
    phi = cox_de_boor(x_norm, knots, degree)
    right_mask = x_norm > 1.0
    left_mask = x_norm < -1.0
    if right_mask.any():
        phi_end = cox_de_boor(torch.tensor([1.0], device=device, dtype=dtype), knots, degree)
        phi_in = cox_de_boor(torch.tensor([1.0 - eps], device=device, dtype=dtype), knots, degree)
        slope = (phi_end - phi_in) / eps
        phi[right_mask] = phi_end + slope * (x_norm[right_mask] - 1.0).unsqueeze(-1)
    if left_mask.any():
        phi_end = cox_de_boor(torch.tensor([-1.0], device=device, dtype=dtype), knots, degree)
        phi_in = cox_de_boor(torch.tensor([-1.0 + eps], device=device, dtype=dtype), knots, degree)
        slope = (phi_in - phi_end) / eps
        phi[left_mask] = phi_end + slope * (x_norm[left_mask] + 1.0).unsqueeze(-1)
    return phi
