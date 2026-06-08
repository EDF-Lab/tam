"""
Benchmark: torch.jit.script vs plain eager for the three sequential loops
in opera.py (_mlpol, _ewa) and kalman.py (_kalman_block).

Results are printed as median wall-clock time over N_RUNS iterations.
JIT functions include a warm-up pass to amortise compilation cost.
"""

import warnings
import time
import torch
import numpy as np
from typing import Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.get_default_dtype()
N_RUNS = 50       # timing iterations
N_WARMUP = 5      # warm-up passes (amortise JIT compilation)

print(f"Device : {DEVICE}")
print(f"dtype  : {DTYPE}")
print(f"Runs   : {N_WARMUP} warmup + {N_RUNS} timed\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _median_ms(fn, *args, n_warmup=N_WARMUP, n_runs=N_RUNS) -> float:
    for _ in range(n_warmup):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Opera loops, define both plain and JIT versions inline
# ---------------------------------------------------------------------------

def _mlpol_eager(experts_tensor, y_true, loss_type):
    B, T, K = experts_tensor.shape
    dtype, device = experts_tensor.dtype, experts_tensor.device
    scale_factor = torch.max(torch.abs(y_true), dim=1, keepdim=True)[0]
    scale_factor = torch.where(scale_factor == 0.0, torch.tensor(1.0, dtype=dtype, device=device), scale_factor)
    X_scaled = experts_tensor / scale_factor
    Y_scaled = y_true / scale_factor
    weights_history = torch.zeros((B, T, K), dtype=dtype, device=device)
    predictions = torch.zeros((B, T), dtype=dtype, device=device)
    w = torch.ones((B, K), dtype=dtype, device=device) / float(K)
    cum_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    max_sq_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    learning_rates = torch.ones((B, K), dtype=dtype, device=device) / (2.0**20)
    for t in range(T):
        xt_scaled = X_scaled[:, t, :]
        yt_scaled = Y_scaled[:, t, 0]
        weights_history[:, t, :] = w
        predictions[:, t] = torch.sum(w * experts_tensor[:, t, :], dim=1)
        y_hat_scaled = torch.sum(w * xt_scaled, dim=1)
        if loss_type == 'square':
            r = 2.0 * (y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
        else:
            r = torch.sign(y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
        r_square = r ** 2
        cum_regrets += r
        max_r_square = torch.max(r_square, dim=1, keepdim=True)[0]
        max_sq_regret_diff = torch.clamp(max_r_square - max_sq_regrets, min=0.0)
        learning_rates = 1.0 / (1.0 / learning_rates + r_square + max_sq_regret_diff)
        max_sq_regrets += max_sq_regret_diff
        relu_regrets = torch.clamp(cum_regrets, min=0.0)
        w_next = learning_rates * relu_regrets
        w_sum = torch.sum(w_next, dim=1, keepdim=True)
        mask = w_sum > 0.0
        w = torch.where(mask, w_next / w_sum, torch.ones((B, K), dtype=dtype, device=device) / float(K))
    return predictions, weights_history


@torch.jit.script
def _mlpol_jit(experts_tensor: torch.Tensor, y_true: torch.Tensor, loss_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, K = experts_tensor.shape
    dtype, device = experts_tensor.dtype, experts_tensor.device
    scale_factor = torch.max(torch.abs(y_true), dim=1, keepdim=True)[0]
    scale_factor = torch.where(scale_factor == 0.0, torch.tensor(1.0, dtype=dtype, device=device), scale_factor)
    X_scaled = experts_tensor / scale_factor
    Y_scaled = y_true / scale_factor
    weights_history = torch.zeros((B, T, K), dtype=dtype, device=device)
    predictions = torch.zeros((B, T), dtype=dtype, device=device)
    w = torch.ones((B, K), dtype=dtype, device=device) / float(K)
    cum_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    max_sq_regrets = torch.zeros((B, K), dtype=dtype, device=device)
    learning_rates = torch.ones((B, K), dtype=dtype, device=device) / (2.0**20)
    for t in range(T):
        xt_scaled = X_scaled[:, t, :]
        yt_scaled = Y_scaled[:, t, 0]
        weights_history[:, t, :] = w
        predictions[:, t] = torch.sum(w * experts_tensor[:, t, :], dim=1)
        y_hat_scaled = torch.sum(w * xt_scaled, dim=1)
        if loss_type == 'square':
            r = 2.0 * (y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
        else:
            r = torch.sign(y_hat_scaled.unsqueeze(1) - yt_scaled.unsqueeze(1)) * (y_hat_scaled.unsqueeze(1) - xt_scaled)
        r_square = r ** 2
        cum_regrets += r
        max_r_square = torch.max(r_square, dim=1, keepdim=True)[0]
        max_sq_regret_diff = torch.clamp(max_r_square - max_sq_regrets, min=0.0)
        learning_rates = 1.0 / (1.0 / learning_rates + r_square + max_sq_regret_diff)
        max_sq_regrets += max_sq_regret_diff
        relu_regrets = torch.clamp(cum_regrets, min=0.0)
        w_next = learning_rates * relu_regrets
        w_sum = torch.sum(w_next, dim=1, keepdim=True)
        mask = w_sum > 0.0
        w = torch.where(mask, w_next / w_sum, torch.ones((B, K), dtype=dtype, device=device) / float(K))
    return predictions, weights_history


# ---------------------------------------------------------------------------
# Kalman loop, define both versions inline
# ---------------------------------------------------------------------------

def _kalman_eager(phi_matrix, y_stacked, base_pred_stacked, B, P_init_diag, observation_noise_var, process_noise_var, eps, offset_boost):
    G, N, d = phi_matrix.shape
    dtype, device = phi_matrix.dtype, phi_matrix.device
    theta_t = torch.zeros((G, d, 1), device=device, dtype=dtype)
    P_t = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * P_init_diag
    Q_matrix = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * process_noise_var
    if d > 0:
        P_t[:, 0, 0] = P_t[:, 0, 0] * offset_boost
        Q_matrix[:, 0, 0] = Q_matrix[:, 0, 0] * offset_boost
    predictions = torch.zeros((G, N, 1), device=device, dtype=dtype)
    num_blocks = (N + B - 1) // B
    state_history_gpu = torch.zeros((num_blocks, G, d, 1), device=device, dtype=dtype)
    for i in range(num_blocks):
        t = i * B
        curr_B = B if t + B <= N else N - t
        X_B = phi_matrix[:, t:t+curr_B, :]
        Y_B = y_stacked[:, t:t+curr_B, :]
        Y_base_B = base_pred_stacked[:, t:t+curr_B, :]
        delta_B = torch.bmm(X_B, theta_t)
        y_hat_B = Y_base_B + delta_B
        predictions[:, t:t+curr_B, :] = y_hat_B
        state_history_gpu[i] = theta_t
        innovations = Y_B - y_hat_B
        nan_mask = torch.isnan(innovations)
        innovations = torch.where(nan_mask, torch.zeros_like(innovations), innovations)
        R_B = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * observation_noise_var
        P_X_T = torch.bmm(P_t, X_B.transpose(-2, -1))
        S = R_B + torch.bmm(X_B, P_X_T)
        jitter = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * eps
        L = torch.linalg.cholesky(S + jitter)
        S_inv = torch.cholesky_inverse(L)
        K_gain = torch.bmm(P_X_T, S_inv)
        theta_t = theta_t + torch.bmm(K_gain, innovations)
        P_t = P_t - torch.bmm(K_gain, torch.bmm(X_B, P_t))
        P_t = P_t + (Q_matrix * float(curr_B))
        P_t = (P_t + P_t.transpose(-2, -1)) * 0.5
    return predictions, state_history_gpu


@torch.jit.script
def _kalman_jit(phi_matrix: torch.Tensor, y_stacked: torch.Tensor, base_pred_stacked: torch.Tensor,
                B: int, P_init_diag: float, observation_noise_var: float, process_noise_var: float,
                eps: float, offset_boost: float) -> Tuple[torch.Tensor, torch.Tensor]:
    G, N, d = phi_matrix.shape
    dtype, device = phi_matrix.dtype, phi_matrix.device
    theta_t = torch.zeros((G, d, 1), device=device, dtype=dtype)
    P_t = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * P_init_diag
    Q_matrix = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * process_noise_var
    if d > 0:
        P_t[:, 0, 0] = P_t[:, 0, 0] * offset_boost
        Q_matrix[:, 0, 0] = Q_matrix[:, 0, 0] * offset_boost
    predictions = torch.zeros((G, N, 1), device=device, dtype=dtype)
    num_blocks = (N + B - 1) // B
    state_history_gpu = torch.zeros((num_blocks, G, d, 1), device=device, dtype=dtype)
    for i in range(num_blocks):
        t = i * B
        curr_B = B if t + B <= N else N - t
        X_B = phi_matrix[:, t:t+curr_B, :]
        Y_B = y_stacked[:, t:t+curr_B, :]
        Y_base_B = base_pred_stacked[:, t:t+curr_B, :]
        delta_B = torch.bmm(X_B, theta_t)
        y_hat_B = Y_base_B + delta_B
        predictions[:, t:t+curr_B, :] = y_hat_B
        state_history_gpu[i] = theta_t
        innovations = Y_B - y_hat_B
        nan_mask = torch.isnan(innovations)
        innovations = torch.where(nan_mask, torch.zeros_like(innovations), innovations)
        R_B = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * observation_noise_var
        P_X_T = torch.bmm(P_t, X_B.transpose(-2, -1))
        S = R_B + torch.bmm(X_B, P_X_T)
        jitter = torch.eye(curr_B, device=device, dtype=dtype).unsqueeze(0).repeat(G, 1, 1) * eps
        L = torch.linalg.cholesky(S + jitter)
        S_inv = torch.cholesky_inverse(L)
        K_gain = torch.bmm(P_X_T, S_inv)
        theta_t = theta_t + torch.bmm(K_gain, innovations)
        P_t = P_t - torch.bmm(K_gain, torch.bmm(X_B, P_t))
        P_t = P_t + (Q_matrix * float(curr_B))
        P_t = (P_t + P_t.transpose(-2, -1)) * 0.5
    return predictions, state_history_gpu


# ---------------------------------------------------------------------------
# Run benchmarks across representative sizes
# ---------------------------------------------------------------------------

def bench_opera():
    print("=" * 60)
    print("OPERA loops  (B=groups, T=time steps, K=experts)")
    print(f"{'Config':<28} {'Eager (ms)':>12} {'JIT (ms)':>12} {'JIT speedup':>12}")
    print("-" * 60)

    configs = [
        (2,   40,  2),
        (2,   200, 2),
        (10,  200, 5),
        (10,  500, 5),
        (50,  500, 5),
        (50,  2000, 10),
    ]
    for B, T, K in configs:
        experts = torch.randn(B, T, K, dtype=DTYPE, device=DEVICE)
        y_true  = torch.randn(B, T, 1,  dtype=DTYPE, device=DEVICE)

        t_eager = _median_ms(_mlpol_eager, experts, y_true, 'square')
        t_jit   = _median_ms(_mlpol_jit,   experts, y_true, 'square')
        speedup = t_eager / t_jit

        label = f"B={B:3d} T={T:5d} K={K:2d}"
        print(f"{label:<28} {t_eager:>12.3f} {t_jit:>12.3f} {speedup:>11.2f}x")

    print()


def bench_kalman():
    print("=" * 60)
    print("Kalman block loop  (G=groups, N=steps, d=features, B=block)")
    print(f"{'Config':<34} {'Eager (ms)':>12} {'JIT (ms)':>12} {'JIT speedup':>12}")
    print("-" * 60)

    configs = [
        (2,  100, 4,  16),
        (2,  500, 8,  32),
        (10, 500, 8,  32),
        (10, 500, 16, 64),
        (24, 500, 8,  32),
        (24, 2000, 16, 128),
    ]
    for G, N, d, B in configs:
        phi   = torch.randn(G, N, d, dtype=DTYPE, device=DEVICE)
        y     = torch.randn(G, N, 1, dtype=DTYPE, device=DEVICE)
        base  = torch.randn(G, N, 1, dtype=DTYPE, device=DEVICE)
        args  = (phi, y, base, B, 1.0, 1.0, 1e-4, 1e-6, 100.0)

        t_eager = _median_ms(_kalman_eager, *args)
        t_jit   = _median_ms(_kalman_jit,   *args)
        speedup = t_eager / t_jit

        label = f"G={G:3d} N={N:5d} d={d:3d} B={B:4d}"
        print(f"{label:<34} {t_eager:>12.3f} {t_jit:>12.3f} {speedup:>11.2f}x")

    print()


if __name__ == "__main__":
    bench_opera()
    bench_kalman()
    print("Done. If JIT speedup > 1.1x on your target platform,")
    print("consider keeping @torch.jit.script and suppressing the warning")
    print("with warnings.filterwarnings in the module header.")


"""
Device : cpu                                                                                                
dtype  : torch.float32
Runs   : 5 warmup + 50 timed

============================================================
OPERA loops  (B=groups, T=time steps, K=experts)
Config                         Eager (ms)     JIT (ms)  JIT speedup
------------------------------------------------------------
B=  2 T=   40 K= 2                  8.868        5.871        1.51x
B=  2 T=  200 K= 2                 44.613       21.771        2.05x
B= 10 T=  200 K= 5                 32.979       22.624        1.46x
B= 10 T=  500 K= 5                 80.351       56.382        1.43x
B= 50 T=  500 K= 5                 93.589       68.289        1.37x
B= 50 T= 2000 K=10                389.623      276.509        1.41x

============================================================
Kalman block loop  (G=groups, N=steps, d=features, B=block)
Config                               Eager (ms)     JIT (ms)  JIT speedup
------------------------------------------------------------
G=  2 N=  100 d=  4 B=  16                1.827        1.176        1.55x
G=  2 N=  500 d=  8 B=  32                4.256        2.960        1.44x
G= 10 N=  500 d=  8 B=  32                5.571        4.256        1.31x
G= 10 N=  500 d= 16 B=  64                5.566        4.783        1.16x
G= 24 N=  500 d=  8 B=  32                7.944        6.646        1.20x
G= 24 N= 2000 d= 16 B= 128               83.268       73.542        1.13x

Device : cuda
dtype  : torch.float32
Runs   : 5 warmup + 50 timed

============================================================
OPERA loops  (B=groups, T=time steps, K=experts)
Config                         Eager (ms)     JIT (ms)  JIT speedup
------------------------------------------------------------
B=  2 T=   40 K= 2                 21.653        8.599        2.52x
B=  2 T=  200 K= 2                120.854       43.023        2.81x
B= 10 T=  200 K= 5                105.200       43.904        2.40x
B= 10 T=  500 K= 5                266.124      121.670        2.19x
B= 50 T=  500 K= 5                267.711      133.273        2.01x
B= 50 T= 2000 K=10               1084.350      689.147        1.57x

============================================================
Kalman block loop  (G=groups, N=steps, d=features, B=block)
Config                               Eager (ms)     JIT (ms)  JIT speedup
------------------------------------------------------------
G=  2 N=  100 d=  4 B=  16                7.063        4.985        1.42x
G=  2 N=  500 d=  8 B=  32               42.363       42.804        0.99x
G= 10 N=  500 d=  8 B=  32               50.938       41.109        1.24x
G= 10 N=  500 d= 16 B=  64               25.274       22.843        1.11x
G= 24 N=  500 d=  8 B=  32               48.081       39.634        1.21x
G= 24 N= 2000 d= 16 B= 128               39.187       43.127        0.91x

"""