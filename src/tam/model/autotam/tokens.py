# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The column names a formula uses, read the same way everywhere in AutoTAM.

A column of a frame loaded from R or from a CSV can be called ``day.of.year``. Cutting a formula at every character that is not a letter, a digit
or an underscore turns it into ``day``, ``of`` and ``year``, none of which is a column: the candidate that uses it is fitted without its column,
fails, and scores as infinite, so the search quietly drops the feature. ``identifier_tokens`` keeps the dots inside a name.
"""

import logging
import re
from typing import Any, Dict, Iterable, List

from tam.common.utils import parse_formula_to_terms

logger = logging.getLogger(__name__)

#: <identifier_tokens>
IDENTIFIER = r"[A-Za-z0-9_][A-Za-z0-9_.]*"
FIRST_ARGUMENT = r"\(\s*([A-Za-z_][A-Za-z0-9_.]*)"


def identifier_tokens(text: str) -> List[str]:
    """Every name in ``text`` (columns, functions, numbers), dots kept inside a name: ``"s(day.of.year, k=10)"`` gives ``day.of.year``."""
    return re.findall(IDENTIFIER, text)


def first_arguments(text: str) -> List[str]:
    """The first argument of every call in ``text``: ``"te(s(x.one), s(x.two))"`` gives ``x.one`` and ``x.two`` (and ``s``, ``s``)."""
    return re.findall(FIRST_ARGUMENT, text)


def term_columns(terms: List[Dict[str, Any]]) -> List[str]:
    """The columns the parsed terms of a formula use, in order of appearance: the feature of each term, the columns named by ``others`` (a
    ``|`` separated list) and ``slope``, and the margins of a tensor product (read recursively). It reads what ``StaticTAM`` reads, so a column a
    term takes by keyword is not missed."""
    found: List[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name.strip() and name.strip() not in found:
            found.append(name.strip())

    for term in terms:
        params = term.get("params", {}) or {}
        if term.get("type") != "te":
            add(term.get("feature"))
        if isinstance(params.get("others"), str):
            for extra in params["others"].split("|"):
                add(extra)
        if isinstance(params.get("slope"), str):
            add(params["slope"])
        if term.get("type") == "te":
            for margin in params:
                if margin and "(" in str(margin):
                    try:
                        _, sub_terms = parse_formula_to_terms(f"DUMMY ~ {str(margin).strip()}")
                    except Exception as exc:
                        logger.debug("margin %r of a tensor term could not be parsed: %s", margin, exc)
                        continue
                    for name in term_columns(sub_terms):
                        add(name)
    return found


def formula_columns(formula: str, available: Iterable[str]) -> List[str]:
    """The columns of ``available`` that a formula uses: its target and the columns of its terms, in order of appearance.

    A word of the formula that is a function or a keyword of a term (``s``, ``l``, ``deg``, ``nominal``) is not a column even when the frame
    has a column of that name; a formula that cannot be parsed falls back on its words.
    """
    try:
        target, terms = parse_formula_to_terms(formula)
        names = ([target] if target else []) + term_columns(terms)
    except Exception as exc:
        logger.debug("formula %r could not be parsed, its words are used: %s", formula, exc)
        names = identifier_tokens(formula)
    present = set(available)
    return [n for n in dict.fromkeys(names) if n in present]
#: </identifier_tokens>
