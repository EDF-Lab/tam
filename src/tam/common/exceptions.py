# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""Warning categories raised by TAM, so that callers can filter or escalate them."""


class TAMExtrapolationWarning(UserWarning):
    r"""
    A forecast evaluates an effect outside the range it was trained on.

    Raised once per model and feature when, after normalisation, a feature read by a non-linear effect
    (``s``, ``f``, ``p``, ``w``, ``rbf``, ``n``, ``t``, ``phys``, or a ``te`` margin) leaves [-1, 1], and when a categorical
    level was not seen in training. ``l()`` never warns. Use ``extrapolate='constant'`` to hold the effect at the edge of the
    trained range, or escalate it in tests with ``warnings.simplefilter("error", ta.TAMExtrapolationWarning)``.
    """


import warnings
from contextlib import contextmanager

_collectors: list = []


def warn_extrapolation(feature: str, message: str, stacklevel: int = 3) -> None:
    r"""Raises a ``TAMExtrapolationWarning``, unless a ``collect_extrapolation()`` block is open: it then only records the feature."""
    if _collectors:
        for features in _collectors:
            features.setdefault(feature, message)
        return
    warnings.warn(message, TAMExtrapolationWarning, stacklevel=stacklevel + 1)


@contextmanager
def collect_extrapolation():
    r"""Silences ``TAMExtrapolationWarning`` and collects ``{feature: message}`` of the ones that would have been raised (AutoTAM search)."""
    features: dict = {}
    _collectors.append(features)
    try:
        yield features
    finally:
        _collectors.remove(features)
