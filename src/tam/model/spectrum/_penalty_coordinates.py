# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Several smoothing parameters for one effect.

GCV and REML tune one smoothing weight per *penalty coordinate*. A spline has one; a tensor product has one per margin
(``TensorProductEffect``). The mixin below gives the same four members to an effect made of blocks that deserve their own weight
(for example one weight per decay constant, or one for a base block and one for the block added to it):

* ``n_penalty_coordinates`` (from ``BaseEffect``) is the number of coordinates;
* ``coordinate_labels()`` names them, in a fixed order;
* ``penalty_coordinates()`` returns the smoothing weights (lambda, not log10);
* ``set_penalty_coordinates(values)`` sets them; a list of length 1 sets coordinate ``j`` to ``values[0] * multiplier_j``.

Assigning ``lambda_p`` (the ``ap`` of a formula) sets every coordinate to ``lambda_p * multiplier_j``. With every coordinate at its
initial value the penalty equals the one the effect had before it was split into coordinates.
"""

from typing import List, Optional, Sequence


class CoordinatePenaltyMixin:
    """Put it before ``BaseEffect`` in the bases of the effect; call ``_init_coordinates`` in ``__init__``."""

    _coord_mult: List[float] = [1.0]
    _coord_labels: List[str] = ["lambda"]
    _coords: Optional[List[float]] = None
    _lambda_p: float = 1.0

    @property
    def lambda_p(self) -> float:
        """The smoothing weight(s) of the effect."""
        return self._lambda_p

    @lambda_p.setter
    def lambda_p(self, value: float) -> None:
        """The smoothing weight(s) of the effect."""
        self._lambda_p = float(value)
        self._coords = None          # every coordinate follows lambda_p again

    def _init_coordinates(self, labels: List[str], multipliers: List[float]) -> None:
        """Declares the coordinates: their labels and their ratios to ``lambda_p`` at the start."""
        self._coord_labels = list(labels)
        self._coord_mult = [float(m) for m in multipliers]
        self._coords = None

    def coordinate_labels(self) -> List[str]:
        """Names of the penalty coordinates."""
        return list(self._coord_labels)

    def penalty_coordinates(self) -> list:
        """Current weights of the penalty coordinates."""
        if self._coords is not None:
            return list(self._coords)
        return [self._lambda_p * m for m in self._coord_mult]

    def set_penalty_coordinates(self, values: Sequence[float]) -> None:
        """Sets the weights of the penalty coordinates (one value sets all of them)."""
        values = [float(v) for v in values]
        n = len(self._coord_mult)
        if len(values) == 1:
            self._coords = [values[0] * m for m in self._coord_mult]
        elif len(values) == n:
            self._coords = values
        else:
            raise ValueError(f"{self.__class__.__name__} has {n} penalty coordinates, got {len(values)}")
