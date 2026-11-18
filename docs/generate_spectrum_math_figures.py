# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Draws the basis and penalty figure ``[Phi_l | P_l]`` of every spectrum effect of the release.

For each effect the script instantiates the live class from ``tam.model.spectrum`` on one deterministic cold-wave signal (a 10-day hourly
temperature series, no dataset file), calls ``initialize``, ``build_feature_map`` and ``build_penalty_matrix``, and draws

* left: a few representative columns of the design block ``Phi_l`` over the cold-wave trajectory (with its causal memory when the effect has one);
* right: the penalty ``P_l`` as a heat map of ``log10 |P_l|``, with the flat ridge ``lambda I`` of the same trace as the reference.

The figures are written to ``math/spectrum/assets/<effect>_geometry.png``. Run it from the repository root: ``python docs/generate_spectrum_math_figures.py``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

import importlib


class _Lazy:
    """A module of ``tam.model.spectrum`` imported when an effect of it is drawn: the script lists every effect and draws the ones that exist."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str):
        return getattr(importlib.import_module(f"tam.model.spectrum.{self._name}"), attr)


_bispectrum, _fno, _foundation, _hippo, _hurst, _inertia, _kan, _koopman, _patch_basis, _prior_spectral = (
    _Lazy(n) for n in ("_bispectrum", "_fno", "_foundation", "_hippo", "_hurst", "_inertia", "_kan", "_koopman", "_patch_basis", "_prior_spectral"))
_scattering, _signature, _single_index, _ssa, _temporal_operator, _tjepa, _transformer = (
    _Lazy(n) for n in ("_scattering", "_signature", "_single_index", "_ssa", "_temporal_operator", "_tjepa", "_transformer"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "math" / "spectrum" / "assets"
HOURS = 240  # ten days, hourly


def cold_wave() -> np.ndarray:
    """A 10-day hourly temperature: a diurnal cycle on a seasonal level, with a cold wave of about 12 degrees centred on day 5."""
    t = np.arange(HOURS, dtype=float)
    diurnal = 3.0 * np.sin(2.0 * math.pi * (t - 9.0) / 24.0)
    wave = -12.0 * np.exp(-(((t - 120.0) / 40.0) ** 2))
    return 6.0 + diurnal + wave


def _normalized(x: np.ndarray) -> torch.Tensor:
    """The signal as the effects read it: rescaled to [-1, 1] (the range the model gives to every feature)."""
    lo, hi = float(x.min()), float(x.max())
    return torch.tensor(2.0 * (x - lo) / (hi - lo) - 1.0, dtype=torch.float64)


# name -> (release, class factory, input kind, memory note). The input kind is "series" (the cold wave) or "phase" (hour of the day with a day-type regime).
EFFECTS = {
    "sig": ("1.6.0", lambda: _signature.SignatureEffect("x", w=9, N=3), "series", "window w = 9"),
    "si": ("1.6.0", lambda: _single_index.SingleIndexEffect("x", k=8, lag_window=24), "series", "lag window 24"),
    "kan": ("1.6.0", lambda: _kan.KANEffect("x", n_nodes=3, k_inner=5, k_outer=6), "series", "memoryless"),
    "inertia": ("1.6.0", lambda: _inertia.InertiaEffect("x", alphas=(0.2, 0.05, 0.01), k=6), "series", "exponential memory"),
    "dd": ("1.6.0", lambda: _inertia.DegreeDayEffect("x"), "series", "memoryless"),
    "wcal": ("1.6.0", lambda: _inertia.WarpedCalendarEffect("x", regime_feature="r", period=24.0, n_harmonics=3), "phase", "memoryless"),
    "hp_trend": ("1.6.0", lambda: _inertia.MacroTrendEffect("x", k=10), "series", "memoryless"),
    "temporal_operator": ("1.6.0", lambda: _temporal_operator.TemporalOperatorEffect("x", operator_type="attn", k_base=5, n_random_features=4), "series", "running sums"),
    "patch_basis": ("1.6.0", lambda: _patch_basis.PatchBasisEffect("x", patch_len=12, n_tokens=6), "series", "patch of 12 steps"),
    "foundation": ("1.6.0", lambda: _foundation.FoundationEffect("x", patch_len=12, n_tokens=4, n_bins=4, scales=(1, 4, 16)), "series", "patch, filters and histogram"),
    "koopman": ("1.6.0", lambda: _koopman.KoopmanEffect("x", k=6, n_lift=24, delay_embed=6), "series", "delay embedding of 6"),
    "pspec": ("1.6.1", lambda: _prior_spectral.PriorSpectralEffect("x", k=8), "series", "memoryless"),
    "hippo": ("1.6.1", lambda: _hippo.HiPPOEffect("x", k=6), "series", "continuous-time memory"),
    "attn": ("1.6.1", lambda: _transformer.TransformerEffect("x", m_features=4, n_heads=2), "series", "running sums"),
    "bispec": ("1.6.1", lambda: _bispectrum.BispectrumEffect("x", k=2, n_fft=32), "series", "window of 32"),
    "wst": ("1.6.1", lambda: _scattering.ScatteringEffect("x", J=3, Q=1, max_order=2), "series", "finite memory"),
    "tjepa": ("1.6.1", lambda: _tjepa.TJEPAEffect("x", latent_dim=3, window=12, horizon=3, readout="spline", k_spline=5), "series", "window of 12"),
    "fno": ("1.6.1", lambda: _fno.OperatorEffect("x", modes=4, width=3, window=16), "series", "window of 16"),
    "hurst": ("1.6.1", lambda: _hurst.HurstEffect("x", k=8, H=0.75), "series", "memoryless"),
    "ssa": ("1.6.1", lambda: _ssa.SSAEffect("x", window=12, k=5), "series", "window of 12"),
}


def _inputs(kind: str) -> torch.Tensor:
    if kind == "phase":
        t = np.arange(HOURS, dtype=float)
        phase = 2.0 * ((t % 24.0) / 24.0) - 1.0
        regime = ((t // 24.0) % 7 >= 5).astype(float)  # a weekend indicator
        return torch.tensor(np.stack([phase, regime], axis=-1), dtype=torch.float64)[None]  # (1, T, 2)
    return _normalized(cold_wave())[None, :, None]  # (1, T, 1)


def figure(name: str, release: str, factory, kind: str, memory: str) -> Path:
    # A memoryless effect reads the series as (B, T); a temporal one as (B, T, D). Take the layout whose feature map has one row per hour.
    for layout in ("btd", "bt"):
        torch.manual_seed(0)
        effect = factory()
        x = _inputs(kind)
        if layout == "bt" and x.dim() == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        effect.initialize(x)
        features = effect.build_feature_map(x).detach().cpu()
        if features.dim() == 3 and features.shape[0] == 1 and features.shape[1] == HOURS:
            phi = features[0].numpy()  # (T, K)
            break
        if features.dim() == 2 and features.shape[0] == HOURS:
            phi = features.numpy()
            break
    else:
        raise RuntimeError(f"{name}: no layout gives one feature row per hour")
    penalty = effect.build_penalty_matrix()
    if penalty.is_sparse:
        penalty = penalty.to_dense()
    penalty = penalty.detach().cpu().numpy()
    n_cols = phi.shape[1]
    picks = sorted(set(np.linspace(0, n_cols - 1, min(5, n_cols)).round().astype(int)))

    fig, (ax_phi, ax_pen) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=200, constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1.0]})
    hours = np.arange(HOURS)
    ax_t = ax_phi.twinx()
    ax_t.plot(hours, cold_wave(), color="0.55", lw=1.0, ls="--")
    ax_t.set_ylabel("temperature (°C)", color="0.45", fontsize=10)
    ax_t.tick_params(axis="y", labelcolor="0.45", labelsize=9)
    for rank, j in enumerate(picks):
        ax_phi.plot(hours, phi[:, j], lw=1.3, label=rf"$\phi_{{l,{j + 1}}}(x_{{1:t}})$", color=plt.cm.viridis(rank / max(1, len(picks) - 1) * 0.9))
    ax_phi.set_xlabel("hour of the 10-day cold wave", fontsize=10)
    ax_phi.set_ylabel(r"columns of $\Phi_l$", fontsize=10)
    ax_phi.set_title(rf"$\Phi_l$ ({n_cols} columns, {memory})", fontsize=10)
    ax_phi.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax_phi.tick_params(labelsize=9)

    magnitude = np.log10(np.abs(penalty) + 1e-12)
    # the entries below six decades under the largest one are shown as the floor: the structure, not the numerical zeros, is the point
    image = ax_pen.imshow(magnitude, cmap="magma", aspect="equal", vmin=max(float(magnitude.min()), float(magnitude.max()) - 6.0))
    ax_pen.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_pen.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ridge = float(np.trace(np.abs(penalty))) / max(1, n_cols)
    diag = np.diag(penalty)
    ax_pen.set_title(rf"$P_l$: $\log_{{10}}|P_l|$; diagonal {diag.min():.1e} to {diag.max():.1e}, flat ridge {ridge:.1e}", fontsize=8.5)
    ax_pen.set_xlabel("column", fontsize=10)
    ax_pen.set_ylabel("column", fontsize=10)
    ax_pen.tick_params(labelsize=9)
    cbar = fig.colorbar(image, ax=ax_pen, shrink=0.85)
    cbar.set_label(r"$\log_{10}|P_l|$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_geometry.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="all", choices=["all", "1.6.0", "1.6.1"], help="only the effects first shipped in this release")
    parser.add_argument("--only", default=None, help="comma-separated effect names")
    args = parser.parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    for name, (release, factory, kind, memory) in EFFECTS.items():
        if (args.release != "all" and release != args.release) or (wanted and name not in wanted):
            continue
        try:
            print(figure(name, release, factory, kind, memory))
        except ModuleNotFoundError:
            print(f"{name}: not in this version of the package, skipped")


if __name__ == "__main__":
    main()
