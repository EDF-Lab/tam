# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the HiPPO State Space Model effect via continuous-time orthogonal polynomial memory.

Projects past input trajectories onto an explicit Primal basis of orthogonal Legendre polynomials
using continuous-time linear time-invariant (LTI) state space operators. Discretization is
performed via Bilinear (Tustin) or Zero-Order Hold (ZOH) transforms and evaluated in O(N T log T)
via causal FFT convolutions.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import from_btk, to_btd


def _build_hippo_matrices(
    k: int, measure: str = "legt", dt: float = 0.05, discretization: str = "bilinear"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs continuous-time HiPPO (A, B) matrices and discretizes them into (A_bar, B_bar)."""
    measure = str(measure).lower().strip()
    discretization = str(discretization).lower().strip()

    # Continuous A, B constructed in float64 for numerical precision
    a_cont = torch.zeros(k, k, dtype=torch.float64)
    b_cont = torch.zeros(k, dtype=torch.float64)

    if measure == "legt":
        # Translated Legendre on sliding window [t - theta, t]
        for n in range(k):
            c_n = np.sqrt(2.0 * n + 1.0)
            b_cont[n] = c_n
            for m in range(k):
                c_m = np.sqrt(2.0 * m + 1.0)
                if m <= n:
                    a_cont[n, m] = -c_n * c_m
                else:
                    a_cont[n, m] = -c_n * c_m * ((-1.0) ** (n - m))

    elif measure == "legs":
        # Scaled Legendre on [0, t]
        for n in range(k):
            c_n = np.sqrt(2.0 * n + 1.0)
            b_cont[n] = c_n
            for m in range(k):
                c_m = np.sqrt(2.0 * m + 1.0)
                if n > m:
                    a_cont[n, m] = -c_n * c_m
                elif n == m:
                    a_cont[n, m] = -(n + 1.0)
                else:
                    a_cont[n, m] = 0.0

    elif measure == "lagt":
        # Laguerre / exponentially-warped measure
        for n in range(k):
            b_cont[n] = 1.0
            for m in range(k):
                a_cont[n, m] = -1.0 if n >= m else 0.0
    else:
        raise ValueError(f"Unknown HiPPO measure '{measure}'. Use 'legt', 'legs', or 'lagt'.")

    # Discretization
    dt_val = float(dt)
    if discretization == "bilinear":
        eye = torch.eye(k, dtype=torch.float64)
        m_mat = eye - 0.5 * dt_val * a_cont
        rhs_a = eye + 0.5 * dt_val * a_cont
        rhs_b = (dt_val * b_cont).unsqueeze(-1)
        a_bar = torch.linalg.solve(m_mat, rhs_a)
        b_bar = torch.linalg.solve(m_mat, rhs_b).squeeze(-1)

    elif discretization == "zoh":
        # Van Loan's method: exact matrix exponential of augmented block
        augmented = torch.zeros(k + 1, k + 1, dtype=torch.float64)
        augmented[:k, :k] = a_cont
        augmented[:k, k:] = b_cont.unsqueeze(-1)
        exp_m = torch.matrix_exp(dt_val * augmented)
        a_bar = exp_m[:k, :k]
        b_bar = exp_m[:k, k:].squeeze(-1)
    else:
        raise ValueError(
            f"Unknown discretization '{discretization}'. Use 'bilinear' or 'zoh'."
        )

    return a_bar, b_bar


class HiPPOEffect(BaseEffect):
    r"""Implements the HiPPO Continuous-Time State Space Memory effect.

    Projects an input time-series u(t) onto an N-dimensional orthogonal polynomial
    memory basis via the continuous differential system ds/dt = A s(t) + B u(t).
    Unrolled via causal FFT convolutions in O(N T log T) and regularized with a
    Dirichlet-Legendre polynomial degree penalty.

    Attributes:
        k (int): Number of state dimensions (order of polynomial memory N).
        measure (str): Measure weighting the memory ('legt', 'legs', or 'lagt').
        dt (float): Discretization time step Delta.
        discretization (str): Discretization scheme ('bilinear' or 'zoh').
        degree_pen (float): Degree penalty exponent p in diag(1 + (n(n+1))^p).
        a_bar (torch.Tensor): Discretized state transition matrix of shape (k, k).
        b_bar (torch.Tensor): Discretized input projection vector of shape (k,).
    """

    stateful = True

    def __init__(
        self,
        feature_name: str,
        k: int = 16,
        measure: str = "legt",
        dt: float = 0.05,
        discretization: str = "bilinear",
        degree_pen: float = 1.0,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        """Initializes the HiPPOEffect."""
        if k <= 0:
            raise ValueError(f"State dimension k must be positive, got {k}.")
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}.")

        super().__init__(feature_name, "hippo", lambda_p, extrapolate)

        self.k = int(k)
        self.measure = str(measure).lower().strip()
        self.dt = float(dt)
        self.discretization = str(discretization).lower().strip()
        self.degree_pen = float(degree_pen)

        a_bar_64, b_bar_64 = _build_hippo_matrices(
            self.k, self.measure, self.dt, self.discretization
        )
        self.a_bar = a_bar_64.to(dtype=torch.get_default_dtype())
        self.b_bar = b_bar_64.to(dtype=torch.get_default_dtype())

        self._a64 = a_bar_64
        self._b64 = b_bar_64
        self._check_temporal_extrapolate()

    def initialize(self, x_col: torch.Tensor) -> None:
        """The memory operator is fixed by (k, measure, dt): nothing is learned from the data."""
        return None

    def get_n_coeffs(self) -> int:
        """Returns the number of state coefficients (k)."""
        return self.k

    def summary_type(self) -> str:
        """Returns the summary type."""
        return "HiPPO"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return (
            f"k={self.k}, measure='{self.measure}', dt={self.dt}, "
            f"disc='{self.discretization}', DegreePen(p={self.degree_pen})"
        )

    def _compute_ssm_states(
        self,
        x_2d: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        use_fft: bool = True,
    ) -> torch.Tensor:
        """Computes SSM states for an input batch (B, T) -> (B, T, k), from the state ``initial_state`` (B, k) (zero when None)."""
        b_dim, t_len = x_2d.shape
        k = self.k
        out_dtype = x_2d.dtype
        work = torch.float64                                   # the transition matrix is applied up to T times: float64
        x_w = x_2d.to(work)

        a_bar = self._a64.to(device=x_2d.device, dtype=work)
        b_bar = self._b64.to(device=x_2d.device, dtype=work)
        init_s = None if initial_state is None else initial_state.to(device=x_2d.device, dtype=work)

        # Sequential recurrence for short sequences
        if not use_fft or t_len < 64:
            s = torch.zeros(b_dim, t_len, k, dtype=work, device=x_2d.device)
            curr_s = init_s if init_s is not None else torch.zeros(b_dim, k, dtype=work, device=x_2d.device)
            b_vec = b_bar.unsqueeze(0)
            for t in range(t_len):
                curr_s = curr_s @ a_bar.T + x_w[:, t : t + 1] @ b_vec
                s[:, t, :] = curr_s
            return s.to(out_dtype)

        # Causal FFT convolution for T >= 64
        # 1. Compute impulse response kernel K of shape (T, k)
        k_mat = torch.zeros(t_len, k, dtype=work, device=x_2d.device)
        curr_k = b_bar.clone()
        k_mat[0] = curr_k
        for t in range(1, t_len):
            curr_k = curr_k @ a_bar.T
            k_mat[t] = curr_k

        # 2. Causal 2T zero-padded FFT
        u_fft = torch.fft.rfft(x_w, n=2 * t_len, dim=-1)  # (B, T + 1)
        k_fft = torch.fft.rfft(k_mat, n=2 * t_len, dim=0)  # (T + 1, k)

        prod = u_fft.unsqueeze(-1) * k_fft.unsqueeze(0)  # (B, T + 1, k)
        s_fft = torch.fft.irfft(prod, n=2 * t_len, dim=-2)[:, :t_len, :]

        # 3. Add the unforced response of the initial state: s_t += A^(t+1) s_0
        if init_s is not None:
            curr = init_s
            for t in range(t_len):
                curr = curr @ a_bar.T
                s_fft[:, t, :] = s_fft[:, t, :] + curr

        return s_fft.to(out_dtype)

#: <hippo_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Builds explicit SSM state basis matrix: ``(B, T)`` in, ``(B, T, k)`` out; each group has its own memory ``{"c": (B, k)}``."""
        x3, shape, layout = to_btd(x_col, False)
        if x3.shape[-1] != 1:
            raise ValueError(f"hippo reads one column, got {x3.shape[-1]}")
        x_2d = x3[..., 0]
        B = x_2d.shape[0]
        init_s = None
        if state_in is not None and "c" in state_in:
            init_s = state_in["c"].to(device=x_2d.device, dtype=x_2d.dtype)
            if tuple(init_s.shape) != (B, self.k):
                raise ValueError(f"state 'c' has shape {tuple(init_s.shape)}, expected {(B, self.k)}: continue_state needs the same groups and features as training")
        states = self._compute_ssm_states(x_2d, initial_state=init_s)
        self._state_out = {"c": states[:, -1, :].detach().clone()}
        return from_btk(states, shape, layout)
#: </hippo_feature_map>

#: <hippo_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds diagonal Legendre polynomial degree penalty matrix."""
        n_idx = torch.arange(self.k, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        dirichlet_energy = (n_idx * (n_idx + 1.0)) ** self.degree_pen
        diag = self.lambda_p * (1.0 + dirichlet_energy)
        return torch.diag(diag)
#: </hippo_penalty_matrix>
