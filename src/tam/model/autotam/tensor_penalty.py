# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The smoothing weight of the margins of a tensor product, as the search chose it.

A ``te(...)`` term without an ``ap=`` of its own carries the engine default (1e-9) as the weight of the product itself, and the engine folds that
weight into the penalty of every margin: ``te(s(x1, ap=-3), s(x2, ap=-3))`` is fitted with margin penalties of 1e-12, not the 1e-3 the search
wrote. ``with_tensor_outer_penalty`` writes ``ap=0.0`` on the product (a weight of one), so each margin keeps its own ``ap``.
"""

import re

from tam.common.utils import split_args_respecting_parentheses

#: <tensor_penalty>
_TENSOR = re.compile(r"\bte\s*\(")


def _closing(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        depth += text[index] == "("
        depth -= text[index] == ")"
        if depth == 0:
            return index
    return -1


def with_tensor_outer_penalty(formula: str) -> str:
    """``formula`` with ``ap=0.0`` added to every ``te(...)`` that has no ``ap=`` of its own (at its own level, not in a margin)."""
    out, position = [], 0
    for match in _TENSOR.finditer(formula):
        if match.start() < position:
            continue
        opening = match.end() - 1
        closing = _closing(formula, opening)
        if closing < 0:
            break
        arguments = split_args_respecting_parentheses(formula[opening + 1:closing])
        has_ap = any(re.match(r"^\s*ap\s*=", argument) for argument in arguments)
        out.append(formula[position:closing])
        out.append("" if has_ap else ", ap=0.0")
        position = closing
    out.append(formula[position:])
    return "".join(out)
#: </tensor_penalty>
