# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Feature Profiler for Automated TAM (AutoTAM).

Deterministic O(N) pre-search diagnostic that decides, per feature, whether a linear
term suffices and - when it does not - which basis family and what capacity the signal
actually requires. It runs once before the evolutionary search and writes its verdict
into the search space, so the engine spends its budget on structure (which feature x
which effect x which interaction) rather than rediscovering basis capacity by trial.

The diagnosis is computed on partial residuals from a single ridge solve, so a feature
is judged on the signal left for it once the other features are accounted for. Group
structure is removed by a within-group transform first, matching the block-diagonal
geometry the solver uses on panel data.

See math/meta/12_autotam_feature_profiling.md for the estimators and their justification.
"""

#: <feature_profiler_imports>
import re

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
#: </feature_profiler_imports>

_TINY = 1e-12


#: <feature_profiler_class>
class FeatureProfiler:
    """
    Diagnoses per-feature linearity, basis family and required capacity.

    Attributes:
        n_bins (int): Number of quantile bins used to build each feature's response profile.
        linearity_tol (float): Minimum excess of the correlation ratio over the squared
            Pearson correlation for a feature to be treated as non-linear.
        energy_target (float): Fraction of spectral energy the retained harmonics must reach.
        tv_threshold (float): Normalised total-variation above which a profile is treated
            as spiky and a localised (wavelet) basis is preferred.
        max_harmonics (int): Upper bound on the reported harmonic count.
        ridge_lambda (float): Ridge regularisation used by the baseline solve.
    """

#: <feature_profiler_init>
    def __init__(self, n_bins: int = 50, linearity_tol: float = 0.02,
                 energy_target: float = 0.95, tv_threshold: float = 8.0,
                 max_harmonics: int = 32, ridge_lambda: float = 1e-6):
        self.n_bins = int(n_bins)
        self.linearity_tol = float(linearity_tol)
        self.energy_target = float(energy_target)
        self.tv_threshold = float(tv_threshold)
        self.max_harmonics = int(max_harmonics)
        self.ridge_lambda = float(ridge_lambda)
        self.profiles_: Dict[str, Dict[str, Any]] = {}
#: </feature_profiler_init>

#: <feature_profiler_baseline>
    def _within_group(self, values: np.ndarray, codes: Optional[np.ndarray], n_groups: int) -> np.ndarray:
        """Removes per-group means from a column-stacked array. Shape: [N, P] -> [N, P]."""
        if codes is None:
            return values - values.mean(axis=0, keepdims=True)
        sums = np.zeros((n_groups, values.shape[1]), dtype=float)
        np.add.at(sums, codes, values)
        counts = np.bincount(codes, minlength=n_groups).astype(float)[:, None]
        return values - (sums / np.maximum(counts, 1.0))[codes]

    def _ridge_baseline(self, x_matrix: np.ndarray, y_vector: np.ndarray) -> np.ndarray:
        """Solves the regularised normal equations once. Shape: [N, P] -> [P]."""
        gram = x_matrix.T @ x_matrix
        scale = float(np.trace(gram)) / max(gram.shape[0], 1)
        gram.flat[:: gram.shape[0] + 1] += self.ridge_lambda * max(scale, 1.0)
        try:
            return np.linalg.solve(gram, x_matrix.T @ y_vector)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(x_matrix, y_vector, rcond=None)[0]
#: </feature_profiler_baseline>

#: <feature_profiler_statistics>
    def _binned_profile(self, feature: np.ndarray, response: np.ndarray):
        """Quantile-bins a feature and returns (bin means of response, bin weights)."""
        edges = np.unique(np.quantile(feature, np.linspace(0.0, 1.0, self.n_bins + 1)))
        if len(edges) < 4:
            return None, None
        idx = np.clip(np.digitize(feature, edges[1:-1], right=False), 0, len(edges) - 2)
        n_bins = len(edges) - 1
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        sums = np.bincount(idx, weights=response, minlength=n_bins)
        occupied = counts > 0
        if occupied.sum() < 4:
            return None, None
        return (sums[occupied] / counts[occupied]), counts[occupied]

    def _eta_squared(self, bin_means: np.ndarray, weights: np.ndarray, response: np.ndarray) -> float:
        """Correlation ratio: variance of the response explained by the bin means."""
        total_var = float(np.var(response))
        if total_var <= _TINY:
            return 0.0
        grand = float(np.average(bin_means, weights=weights))
        between = float(np.average((bin_means - grand) ** 2, weights=weights))
        return float(np.clip(between / total_var, 0.0, 1.0))

    def _harmonics_for_energy(self, profile: np.ndarray) -> int:
        """Minimum number of harmonics whose cumulative spectral energy clears the target."""
        centred = profile - profile.mean()
        if np.allclose(centred, 0.0):
            return 1
        power = np.abs(np.fft.rfft(centred)) ** 2
        power = power[1:]                       # drop the DC bin, already removed by centring
        if power.size == 0 or power.sum() <= _TINY:
            return 1
        cumulative = np.cumsum(power) / power.sum()
        needed = int(np.searchsorted(cumulative, self.energy_target) + 1)
        return int(np.clip(needed, 1, min(self.max_harmonics, power.size)))

    def _total_variation_ratio(self, profile: np.ndarray) -> float:
        """Total variation of the profile normalised by its range; ~1 for a monotone curve.

        The profile is smoothed with a 3-point moving average first: bin-level sampling
        noise otherwise accumulates into the total variation and misreports a smooth
        periodic curve as spiky.
        """
        if profile.size >= 5:
            kernel = np.ones(3) / 3.0
            profile = np.convolve(profile, kernel, mode="valid")
        span = float(profile.max() - profile.min())
        if span <= _TINY:
            return 0.0
        return float(np.abs(np.diff(profile)).sum() / span)

    @staticmethod
    def is_autoregressive_lag(name: str) -> bool:
        """True for a lagged copy of the target, recognised by name (*_lag_*, Load_d*, *_d1, *_d7)."""
        lowered = name.lower()
        return ("_lag_" in lowered or lowered.startswith("load_d")
                or lowered.endswith("_d1") or lowered.endswith("_d7"))

    @staticmethod
    def lag_order(name: str) -> Optional[int]:
        """Order of a raw autoregressive lag (Load_d7 -> 7, Load_lag_48 -> 48), else None.

        A derived column such as Load_d1_ewma_alpha30 is lag-like but not a raw lag, so it has no order.
        """
        if not FeatureProfiler.is_autoregressive_lag(name):
            return None
        match = re.search(r"(?:_lag_|_d)(\d+)$", name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def most_recent_lag(features: List[str]) -> Optional[str]:
        """The raw autoregressive lag with the smallest order (the first listed on a tie), if any."""
        orders = [(FeatureProfiler.lag_order(name), position, name) for position, name in enumerate(features)]
        orders = [entry for entry in orders if entry[0] is not None]
        return min(orders)[2] if orders else None

    @staticmethod
    def _is_index_like(name: str, values: np.ndarray) -> bool:
        """True for autoregressive lags and monotone time counters.

        Force monotone indices and autoregressive lags to linear terms to
        prevent high-capacity bases from chasing serial correlation noise.
        """
        if FeatureProfiler.is_autoregressive_lag(name) or "trend" in name.lower():
            return True
        finite = values[np.isfinite(values)]
        if finite.size > 2:
            steps = np.diff(finite)
            # Require strict monotonicity: tolerance would incorrectly flag cyclic seasonal coordinates (e.g., time-of-year) as monotone.
            if np.all(steps >= 0) or np.all(steps <= 0):
                return True           # a counter/index, not a response curve
        return False
#: </feature_profiler_statistics>

#: <feature_profiler_profile>
    def profile(self, df: pd.DataFrame, target_col: Optional[str], features: List[str],
                group_col: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Diagnoses every numeric candidate feature against its partial residual.

        Args:
            df (pd.DataFrame): Training frame holding the target and candidate features.
            target_col (str, optional): Name of the response column.
            features (List[str]): Candidate feature names to diagnose.
            group_col (str, optional): Panel key removed by a within-group transform.

        Returns:
            Dict[str, Dict[str, Any]]: Per-feature verdict with keys 'linear', 'family',
                'm', 'k', 'eta2', 'r2', 'gap', 'tv_ratio' and 'reason'. Features that
                cannot be diagnosed are omitted, leaving existing behaviour untouched.
        """
        self.profiles_ = {}
        if not target_col or target_col not in df.columns:
            return self.profiles_

        numeric = [c for c in features
                   if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                   and df[c].nunique(dropna=True) > 2]
        if not numeric:
            return self.profiles_

        frame = df[[target_col] + numeric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < max(100, 4 * self.n_bins):
            return self.profiles_

        y_raw = frame[target_col].to_numpy(dtype=float)
        x_raw = frame[numeric].to_numpy(dtype=float)

        codes, n_groups = None, 0
        if group_col and group_col in df.columns:
            codes = pd.Categorical(df.loc[frame.index, group_col]).codes.astype(np.intp)
            if (codes < 0).any():
                codes = None
            else:
                n_groups = int(codes.max()) + 1

        stacked = self._within_group(np.column_stack([y_raw, x_raw]), codes, n_groups)
        y_w, x_w = stacked[:, 0], stacked[:, 1:]

        spread = x_w.std(axis=0)
        usable = spread > _TINY
        if not usable.any():
            return self.profiles_
        x_scaled = np.where(usable, x_w / np.where(usable, spread, 1.0), 0.0)

        beta = self._ridge_baseline(x_scaled, y_w)
        residual_full = y_w - x_scaled @ beta

        for position, name in enumerate(numeric):
            if not usable[position]:
                continue
            contribution = x_scaled[:, position] * beta[position]
            partial = residual_full + contribution          # partial residual for this feature
            feature_values = x_raw[:, position]

            if self._is_index_like(name, feature_values):
                self.profiles_[name] = {"linear": True, "family": "l", "m": None, "k": None,
                                        "eta2": 0.0, "r2": 0.0, "gap": 0.0, "tv_ratio": 0.0,
                                        "reason": "autoregressive lag or monotone index"}
                continue

            bin_means, weights = self._binned_profile(feature_values, partial)
            if bin_means is None:
                continue

            eta2 = self._eta_squared(bin_means, weights, partial)
            corr = np.corrcoef(feature_values, partial)[0, 1]
            r2 = 0.0 if not np.isfinite(corr) else float(corr ** 2)
            gap = eta2 - r2

            entry: Dict[str, Any] = {"eta2": eta2, "r2": r2, "gap": gap,
                                     "tv_ratio": self._total_variation_ratio(bin_means)}
            if gap < self.linearity_tol:
                entry.update(linear=True, family="l", m=None, k=None,
                             reason=f"eta2-r2={gap:.4f} < {self.linearity_tol}")
            else:
                harmonics = self._harmonics_for_energy(bin_means)
                # Allocate roughly two spline knots per resolved oscillation to match Fourier resolution.
                knots = int(np.clip(2 * harmonics, 10, 40))
                family = "w" if entry["tv_ratio"] > self.tv_threshold else (
                    "f" if harmonics >= 3 else "s")
                entry.update(linear=False, family=family, m=int(harmonics), k=knots,
                             reason=f"eta2-r2={gap:.4f}, m={harmonics}, tv={entry['tv_ratio']:.2f}")
            self.profiles_[name] = entry

        return self.profiles_
#: </feature_profiler_profile>

#: <feature_profiler_penalty>
    @staticmethod
    def penalty_for(effect: str, feature: str, harmonics: Optional[int] = None,
                    knots: Optional[int] = None) -> float:
        """
        Basis-typed default penalty exponent (lambda = 10**ap).

        Scale penalties dynamically: near-zero for recursive lags/trends,
        light for categoricals, and heavy for high-capacity seasonal bases.

        """
        lowered = feature.lower()
        if effect == "l" and ("_lag_" in lowered or lowered.startswith("load_d")
                              or "_d1" in lowered or "_d7" in lowered or "trend" in lowered):
            return -30.0
        if effect == "c":
            return -8.0
        if effect in ("s", "f", "w", "p", "rbf"):
            if (harmonics is not None and harmonics >= 15) or (knots is not None and knots >= 30):
                return -3.0
            return -5.0
        return -9.0
#: </feature_profiler_penalty>
#: </feature_profiler_class>
