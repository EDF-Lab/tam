# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Negative binomial and Tweedie families (log link) and their quantiles.

Both are GLM families like ``poisson_family``: the P-WLS atom is reweighted with a variance function, here with a dispersion that
is estimated after each IRLS fit (negative binomial: the shape of the variance ``mu + alpha mu^2``; Tweedie: the scale ``phi`` of
``phi mu^p``). A predicted mean gives a whole law, so ``quantiles`` returns discrete quantiles for the counts (Poisson, negative
binomial) and the quantiles of the compound Poisson-gamma law for the Tweedie family, which carries a point mass at zero.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import optimize, special, stats

from ._glm import GLMFamily, LogLink, poisson_family

_TINY: float = 1e-12
# The dispersion is re-estimated until it moves by less than this relative amount.
_DISPERSION_TOL: float = 1e-3
# Poisson mass beyond which the number of jumps of a compound Poisson-gamma is cut.
_JUMP_TAIL: float = 1e-10
_MAX_JUMPS: int = 2000
_BISECTION_STEPS: int = 45
_CHUNK: int = 4000


def _check_levels(taus: Sequence[float]) -> list:
    levels = [float(tau) for tau in taus]
    if any(not 0.0 < tau < 1.0 for tau in levels):
        raise ValueError(f"taus must lie in (0, 1); got {list(taus)!r}")
    return levels


def _check_non_negative(y: torch.Tensor, family: str) -> None:
    n_bad = int(torch.count_nonzero(y < 0))
    if n_bad:
        raise ValueError(f"The {family} target must be >= 0; {n_bad} negative values found.")


#: <negative_binomial>
class NegativeBinomialFamily(GLMFamily):
    """Negative binomial (NB2) with log link: variance ``mu + alpha mu^2``, ``alpha = 1 / theta`` estimated by maximum likelihood."""

    def __init__(self, dispersion: Optional[float] = None):
        if dispersion is not None and not float(dispersion) > 0.0:
            raise ValueError(f"dispersion must be > 0; got {dispersion!r}")
        self.fixed_dispersion = None if dispersion is None else float(dispersion)
        self.dispersion = self.fixed_dispersion if dispersion is not None else 0.1
        super().__init__(LogLink(), self._variance, self._unit_deviance, "negative_binomial")

    def _variance(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.dispersion * mu * mu

    def _unit_deviance(self, y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        alpha = self.dispersion
        y_pos, mu_pos = y.clamp_min(_TINY), mu.clamp_min(_TINY)
        return 2.0 * (y * torch.log(y_pos / mu_pos) - (y + 1.0 / alpha) * torch.log((1.0 + alpha * y) / (1.0 + alpha * mu_pos)))

    def validate_target(self, y: torch.Tensor) -> None:
        _check_non_negative(y, "negative binomial")

    def estimate_dispersion(self, y: torch.Tensor, mu: torch.Tensor) -> bool:
        """Maximum-likelihood ``alpha`` given the fitted means; returns whether it moved enough to refit the mean."""
        if self.fixed_dispersion is not None:
            return False
        observed = y.detach().cpu().numpy().ravel().astype(float)
        mean = np.clip(mu.detach().cpu().numpy().ravel().astype(float), _TINY, None)

        def negative_log_likelihood(log_theta: float) -> float:
            theta = np.exp(log_theta)
            return -float(np.sum(
                special.gammaln(observed + theta) - special.gammaln(theta)
                + theta * np.log(theta / (theta + mean)) + observed * np.log(mean / (theta + mean))
            ))

        best = optimize.minimize_scalar(negative_log_likelihood, bounds=(-8.0, 12.0), method="bounded", options={"xatol": 1e-6})
        updated = float(np.exp(-best.x))
        moved = abs(updated - self.dispersion) / self.dispersion > _DISPERSION_TOL
        self.dispersion = updated
        return bool(moved)

    def quantiles(self, mu: np.ndarray, taus: Sequence[float]) -> np.ndarray:
        """Discrete quantiles (smallest ``k`` with ``F(k) >= tau``), shape ``(len(mu), len(taus))``."""
        theta = 1.0 / self.dispersion
        mean = np.clip(np.asarray(mu, dtype=float), _TINY, None)
        return np.stack([stats.nbinom.ppf(tau, theta, theta / (theta + mean)) for tau in _check_levels(taus)], axis=1)
#: </negative_binomial>


#: <tweedie>
class TweedieFamily(GLMFamily):
    """Tweedie with power ``1 < p < 2`` and log link: variance ``phi mu^p``, a point mass at zero and a positive continuous part.

    It is the compound Poisson-gamma law: ``N ~ Poisson(lambda)`` jumps, each ``Gamma(a, b)``, with
    ``lambda = mu^(2-p) / (phi (2-p))``, ``a = (2-p) / (p-1)`` and ``b = phi (p-1) mu^(p-1)``. ``phi`` is estimated from the Pearson
    residuals (it does not change the weights, so no refit).
    """

    def __init__(self, power: float = 1.5):
        if not 1.0 < float(power) < 2.0:
            raise ValueError(f"The Tweedie power must lie in (1, 2) (compound Poisson-gamma); got {power!r}")
        self.power = float(power)
        self.dispersion = 1.0
        super().__init__(LogLink(), lambda mu: mu.clamp_min(_TINY) ** self.power, self._unit_deviance, "tweedie")

    def _unit_deviance(self, y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        p = self.power
        y_pos, mu_pos = y.clamp_min(0.0), mu.clamp_min(_TINY)
        return 2.0 * (y_pos ** (2.0 - p) / ((1.0 - p) * (2.0 - p)) - y * mu_pos ** (1.0 - p) / (1.0 - p) + mu_pos ** (2.0 - p) / (2.0 - p))

    def validate_target(self, y: torch.Tensor) -> None:
        _check_non_negative(y, "Tweedie")

    def estimate_dispersion(self, y: torch.Tensor, mu: torch.Tensor) -> bool:
        """Pearson estimate of ``phi``; the mean is not refitted (a constant factor of the weights)."""
        observed = y.detach().cpu().numpy().ravel().astype(float)
        mean = np.clip(mu.detach().cpu().numpy().ravel().astype(float), _TINY, None)
        self.dispersion = float(max(np.mean((observed - mean) ** 2 / mean ** self.power), _TINY))
        return False

    def _jump_parameters(self, mu: np.ndarray):
        p, phi = self.power, self.dispersion
        rate = mu ** (2.0 - p) / (phi * (2.0 - p))
        shape = (2.0 - p) / (p - 1.0)
        scale = phi * (p - 1.0) * mu ** (p - 1.0)
        return rate, shape, scale

    def probability_of_zero(self, mu: np.ndarray) -> np.ndarray:
        """``P(y = 0) = exp(-lambda)``."""
        rate, _, _ = self._jump_parameters(np.clip(np.asarray(mu, dtype=float), _TINY, None))
        return np.exp(-rate)

    def cdf(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """``F(y) = P(0) + sum_n P(N = n) GammaCDF(y; n a, b)`` for ``y >= 0``, summed over the jumps that carry mass."""
        mu = np.clip(np.asarray(mu, dtype=float), _TINY, None)
        y = np.asarray(y, dtype=float)
        rate, shape, scale = self._jump_parameters(mu)
        top = int(min(max(float(np.max(stats.poisson.ppf(1.0 - _JUMP_TAIL, rate))), 1.0) + 1.0, _MAX_JUMPS))
        jumps = np.arange(1, top + 1)[None, :]
        weight = stats.poisson.pmf(jumps, rate[:, None])
        mass = special.gammainc(jumps * shape, np.maximum(y, 0.0)[:, None] / scale[:, None])
        return np.where(y < 0.0, 0.0, np.exp(-rate) + np.sum(weight * mass, axis=1))

    def quantiles(self, mu: np.ndarray, taus: Sequence[float]) -> np.ndarray:
        """Quantiles of the compound Poisson-gamma law by bisection on the CDF; ``0`` where ``tau <= P(y = 0)``."""
        mu = np.clip(np.asarray(mu, dtype=float), _TINY, None)
        levels = _check_levels(taus)
        zero = self.probability_of_zero(mu)
        out = np.zeros((len(mu), len(levels)))
        spread = np.sqrt(self.dispersion * mu ** self.power)
        for column, tau in enumerate(levels):
            rows = np.flatnonzero(tau > zero)
            for start in range(0, len(rows), _CHUNK):
                chunk = rows[start:start + _CHUNK]
                low, high = np.zeros(len(chunk)), mu[chunk] + 12.0 * spread[chunk] + 1e-9
                while True:
                    short = self.cdf(high, mu[chunk]) < tau
                    if not short.any():
                        break
                    high = np.where(short, high * 2.0, high)
                for _ in range(_BISECTION_STEPS):
                    middle = 0.5 * (low + high)
                    below = self.cdf(middle, mu[chunk]) < tau
                    low, high = np.where(below, middle, low), np.where(below, high, middle)
                out[chunk, column] = high
        return out
#: </tweedie>


def poisson_quantiles(mu: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """Discrete Poisson quantiles, shape ``(len(mu), len(taus))``."""
    mean = np.clip(np.asarray(mu, dtype=float), _TINY, None)
    return np.stack([stats.poisson.ppf(tau, mean) for tau in _check_levels(taus)], axis=1)


def negative_binomial_family(dispersion: Optional[float] = None) -> NegativeBinomialFamily:
    """Negative binomial family with log link; ``dispersion`` (alpha) is estimated when None."""
    return NegativeBinomialFamily(dispersion)


def tweedie_family(power: float = 1.5) -> TweedieFamily:
    """Tweedie family (``1 < power < 2``) with log link."""
    return TweedieFamily(power)


def predict_count_quantiles(model, data: pd.DataFrame, taus: Sequence[float]) -> pd.DataFrame:
    """Quantiles of a fitted Poisson, negative binomial or Tweedie ``StaticTAM`` on the rows of ``data``.

    Args:
        model: A fitted StaticTAM whose loss has a ``quantiles`` method.
        data: The rows to forecast.
        taus: Quantile levels in (0, 1).

    Returns:
        A DataFrame with one ``q<tau>`` column per level, non-negative, non-decreasing in the level.
    """
    levels = _check_levels(taus)
    predicted = model.predict(data)
    mean = predicted[f"Estimated{model.target_col_}"].to_numpy(dtype=float)
    strategy = model._reweighting_strategy_
    quantiles = strategy.quantiles(mean, levels)
    return pd.DataFrame({f"q{tau}": quantiles[:, j] for j, tau in enumerate(levels)}, index=predicted.index)


# Poisson gets its quantiles here, without changing poisson_family() in _glm.py.
def poisson_family_with_quantiles() -> GLMFamily:
    family = poisson_family()
    family.quantiles = poisson_quantiles
    return family
