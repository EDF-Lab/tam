# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Exponential-family GLMs (Gaussian, Gamma, Poisson, Binomial) and their links.

Each iteration builds the working response and weights from the current mean mu:
    z = eta + (y - mu) * g'(mu)
    w = 1 / (g'(mu)^2 * V(mu))
The links live here since only the GLM families use them; gaussian_family() is the
identity-link L2 path (one solve == ordinary least squares).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-12
_ETA_CLAMP: float = 30.0


class Link(ABC):
    """A GLM link g: eta = g(mu), mu = g^-1(eta), plus g'(mu) and a starting mu."""

    @abstractmethod
    def link(self, mu: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def inverse(self, eta: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def deriv(self, mu: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def starting_mu(self, y: torch.Tensor) -> torch.Tensor: ...


class IdentityLink(Link):
    """Identity link g(mu)=mu (Gaussian family and all M-estimators)."""

    def link(self, mu): return mu
    def inverse(self, eta): return eta
    def deriv(self, mu): return torch.ones_like(mu)
    def starting_mu(self, y): return y.clone()


class LogLink(Link):
    """Log link g(mu)=log mu; inverse clamps eta to keep mu=exp(eta) bounded."""

    def link(self, mu): return torch.log(mu.clamp_min(_TINY))
    def inverse(self, eta): return torch.exp(eta.clamp(-_ETA_CLAMP, _ETA_CLAMP))
    def deriv(self, mu): return 1.0 / mu.clamp_min(_TINY)
    def starting_mu(self, y): return ((y + y.mean()) * 0.5).clamp_min(_TINY)


class LogitLink(Link):
    """Logit link for the Binomial family."""

    def link(self, mu):
        mu = mu.clamp(_TINY, 1.0 - _TINY)
        return torch.log(mu / (1.0 - mu))

    def inverse(self, eta): return torch.sigmoid(eta.clamp(-_ETA_CLAMP, _ETA_CLAMP))

    def deriv(self, mu):
        mu = mu.clamp(_TINY, 1.0 - _TINY)
        return 1.0 / (mu * (1.0 - mu))

    def starting_mu(self, y): return ((y + 0.5) * 0.5).clamp(0.01, 0.99)


#: <glm_family>
class GLMFamily(ReweightingStrategy):
    """A GLM defined by a link, a variance V(mu) and a unit deviance."""

    is_glm = True

    def __init__(
        self,
        link: Link,
        variance: Callable[[torch.Tensor], torch.Tensor],
        unit_deviance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        name: str,
    ):
        self.link = link
        self.variance = variance
        self.unit_deviance = unit_deviance
        self.name = name

    def initial_eta(self, y: torch.Tensor) -> torch.Tensor:
        return self.link.link(self.link.starting_mu(y))

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        return self.link.inverse(eta)

    def working_response_and_weights(self, y, eta):
        mu = self.link.inverse(eta)
        g_prime = self.link.deriv(mu)
        working_response = eta + (y - mu) * g_prime
        weights = 1.0 / (g_prime * g_prime * self.variance(mu)).clamp_min(_TINY)
        return working_response, weights

    def mean_objective(self, y, eta):
        mu = self.link.inverse(eta)
        return self.unit_deviance(y, mu).mean()
#: </glm_family>


def _gaussian_unit_deviance(y, mu):
    return (y - mu) ** 2


def _gamma_unit_deviance(y, mu):
    y_pos = y.clamp_min(_TINY)
    mu_pos = mu.clamp_min(_TINY)
    return 2.0 * (-torch.log(y_pos / mu_pos) + (y - mu) / mu_pos)


def _poisson_unit_deviance(y, mu):
    y_pos = y.clamp_min(_TINY)
    mu_pos = mu.clamp_min(_TINY)
    return 2.0 * (y * torch.log(y_pos / mu_pos) - (y - mu))


def _binomial_unit_deviance(y, mu):
    mu_c = mu.clamp(_TINY, 1.0 - _TINY)
    term_one = y * torch.log(y.clamp_min(_TINY) / mu_c)
    term_zero = (1.0 - y) * torch.log((1.0 - y).clamp_min(_TINY) / (1.0 - mu_c))
    return 2.0 * (term_one + term_zero)


def gaussian_family() -> GLMFamily:
    """Gaussian identity family; a single P-WLS step equals ordinary least squares."""
    return GLMFamily(IdentityLink(), lambda mu: torch.ones_like(mu), _gaussian_unit_deviance, "gaussian")


def gamma_family() -> GLMFamily:
    """Gamma family with log link (variance V(mu) = mu^2)."""
    return GLMFamily(LogLink(), lambda mu: mu * mu, _gamma_unit_deviance, "gamma")


def poisson_family() -> GLMFamily:
    """Poisson family with log link (variance V(mu) = mu)."""
    return GLMFamily(LogLink(), lambda mu: mu, _poisson_unit_deviance, "poisson")


def binomial_family() -> GLMFamily:
    """Binomial family with logit link (variance V(mu) = mu(1-mu))."""
    return GLMFamily(LogitLink(), lambda mu: (mu * (1.0 - mu)).clamp_min(_TINY),
                     _binomial_unit_deviance, "binomial")
