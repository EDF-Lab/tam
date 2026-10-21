# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Monotone target transforms ``z = g(y)``: the one code path of the plain and the distributional ``StaticTAM``.

A transform is a name plus a dict of parameters estimated on the training target (``estimate_params``); ``forward`` maps the response
to the model scale, ``inverse`` maps back. The inverse is exact for quantiles because ``g`` is increasing: ``q_tau(y) = g^-1(q_tau(z))``.
``inverse_mean`` adds the second-order correction ``E[y] ~ g^-1(z) + (1/2) (g^-1)''(z) var`` (exact for the log) for a mean forecast.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import expit

TRANSFORMS = ("none", "log", "log1p", "asinh", "logit", "boxcox")

_TINY: float = 1e-12
# Offset from the bounds of a logit target, as a fraction of the interval: a value on a bound is nudged inside.
_LOGIT_EPS: float = 1e-6
# Margin added on each side of the observed range when the bounds of a logit target are estimated.
_LOGIT_MARGIN: float = 0.05
_LAMBDA_ZERO: float = 1e-8

_KWARGS = {
    "none": (), "log": ("shift",), "log1p": (), "asinh": ("centre", "scale"), "logit": ("bounds",),
    "boxcox": ("shift", "lam"),
}


def check_kwargs(name: str, kwargs: dict) -> None:
    """Reject a keyword that the transform ``name`` does not use (``mean_correction`` is read by the model, not here)."""
    unknown = sorted(set(kwargs) - set(_KWARGS[name]) - {"mean_correction"})
    if unknown:
        raise ValueError(f"transform_kwargs {unknown} do not apply to target_transform={name!r}; it takes {list(_KWARGS[name])}.")


def _finite(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    return values[np.isfinite(values)]


def _require_above(finite: np.ndarray, floor: float, name: str) -> None:
    n_bad = int(np.count_nonzero(finite <= floor))
    if n_bad:
        raise ValueError(
            f"{n_bad} of {finite.size} training target values are <= {floor:g}, outside the domain of target_transform={name!r}. "
            "Shift the target (transform_kwargs={'shift': c}), or use 'asinh' for a real-valued target."
        )


def estimate_params(name: str, y: np.ndarray, kwargs: Optional[dict] = None) -> dict:
    """Estimate the parameters of the transform on the training target and check its domain.

    ``log`` and ``boxcox`` take a ``shift`` c >= 0 (``g(y + c)``); ``boxcox`` also a ``lam`` (maximum likelihood when omitted);
    ``asinh`` takes ``centre`` and ``scale`` (the median and the normalised MAD, else the standard deviation, when omitted);
    ``logit`` takes ``bounds`` (the observed range widened by 5 % when omitted).

    Raises:
        ValueError: On an unknown name or keyword, or a target outside the domain of the transform.
    """
    if name not in TRANSFORMS:
        raise ValueError(f"target_transform must be one of {TRANSFORMS}; got {name!r}")
    kwargs = dict(kwargs or {})
    check_kwargs(name, kwargs)
    finite = _finite(y)
    if name == "none":
        return {}
    if name == "log":
        shift = float(kwargs.get("shift", 0.0))
        _require_above(finite, -shift, name)
        return {"shift": shift}
    if name == "log1p":
        _require_above(finite, -1.0, name)
        return {}
    if name == "asinh":
        centre = kwargs.get("centre")
        centre = float(np.median(finite)) if centre is None else float(centre)
        scale = kwargs.get("scale")
        if scale is None:
            scale = float(1.4826 * np.median(np.abs(finite - centre)))
            if not scale > 0.0:
                scale = float(np.std(finite))
        scale = float(scale)
        return {"centre": centre, "scale": scale if scale > 0.0 else 1.0}
    if name == "logit":
        bounds = kwargs.get("bounds")
        if bounds is None:
            low, high = float(finite.min()), float(finite.max())
            margin = _LOGIT_MARGIN * (high - low if high > low else 1.0)
            return {"bounds": (low - margin, high + margin)}
        low, high = float(bounds[0]), float(bounds[1])
        if not low < high:
            raise ValueError(f"bounds must be (low, high) with low < high; got {bounds!r}")
        if finite.min() < low or finite.max() > high:
            n_out = int(np.count_nonzero((finite < low) | (finite > high)))
            raise ValueError(f"{n_out} training target values lie outside the bounds {(low, high)} of the logit transform.")
        return {"bounds": (low, high)}
    # boxcox
    shift = float(kwargs.get("shift", 0.0))
    _require_above(finite, -shift, name)
    lam = kwargs.get("lam")
    lam = float(stats.boxcox_normmax(finite + shift, method="mle")) if lam is None else float(lam)
    return {"shift": shift, "lam": lam}


def forward(name: str, params: dict, y: np.ndarray) -> np.ndarray:
    """The response on the model scale."""
    values = np.asarray(y, dtype=float)
    if name == "log":
        return np.log(np.clip(values + params.get("shift", 0.0), _TINY, None))
    if name == "log1p":
        return np.log1p(np.clip(values, -1.0 + _TINY, None))
    if name == "asinh":
        return np.arcsinh((values - params["centre"]) / params["scale"])
    if name == "logit":
        low, high = params["bounds"]
        unit = np.clip((values - low) / (high - low), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
        return np.log(unit / (1.0 - unit))
    if name == "boxcox":
        shifted = np.clip(values + params["shift"], _TINY, None)
        lam = params["lam"]
        return np.log(shifted) if abs(lam) < _LAMBDA_ZERO else (shifted ** lam - 1.0) / lam
    return values


def inverse(name: str, params: dict, z: np.ndarray) -> np.ndarray:
    """The model scale back on the response scale (an overflow gives inf, never an exception)."""
    values = np.asarray(z, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if name == "log":
            return np.exp(values) - params.get("shift", 0.0)
        if name == "log1p":
            return np.expm1(values)
        if name == "asinh":
            return params["centre"] + params["scale"] * np.sinh(values)
        if name == "logit":
            low, high = params["bounds"]
            return low + (high - low) * expit(values)
        if name == "boxcox":
            lam = params["lam"]
            if abs(lam) < _LAMBDA_ZERO:
                return np.exp(values) - params["shift"]
            base = np.clip(lam * values + 1.0, 0.0, None)  # outside the range of g the inverse is 0
            return base ** (1.0 / lam) - params["shift"]
    return values


def _second_derivative(name: str, params: dict, z: np.ndarray) -> np.ndarray:
    """(g^-1)''(z)."""
    values = np.asarray(z, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if name in ("log", "log1p"):
            return np.exp(values)
        if name == "asinh":
            return params["scale"] * np.sinh(values)
        if name == "logit":
            low, high = params["bounds"]
            s = expit(values)
            return (high - low) * s * (1.0 - s) * (1.0 - 2.0 * s)
        if name == "boxcox":
            lam = params["lam"]
            if abs(lam) < _LAMBDA_ZERO:
                return np.exp(values)
            base = np.clip(lam * values + 1.0, _TINY, None)
            return (1.0 - lam) * base ** (1.0 / lam - 2.0)
    return np.zeros_like(values)


def inverse_mean(name: str, params: dict, z: np.ndarray, variance: float) -> np.ndarray:
    """A mean forecast on the response scale: ``g^-1(z) + (1/2) (g^-1)''(z) var`` (``exp(z + var/2)`` for the log)."""
    values = np.asarray(z, dtype=float)
    if name == "log":
        with np.errstate(over="ignore"):
            return np.exp(values + 0.5 * variance) - params.get("shift", 0.0)
    if name == "log1p":
        with np.errstate(over="ignore"):
            return np.exp(values + 0.5 * variance) - 1.0
    return inverse(name, params, values) + 0.5 * _second_derivative(name, params, values) * variance
