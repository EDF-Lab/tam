# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The column names a formula uses, read the same way everywhere in AutoTAM.

A column of a frame loaded from R or from a CSV can be called ``day.of.year``. Cutting a formula at every character that is not a letter, a digit
or an underscore turns it into ``day``, ``of`` and ``year``, none of which is a column: the candidate that uses it is fitted without its column,
fails, and scores as infinite, so the search quietly drops the feature. ``identifier_tokens`` keeps the dots inside a name.
"""

import re
from typing import List

#: <identifier_tokens>
IDENTIFIER = r"[A-Za-z0-9_][A-Za-z0-9_.]*"
FIRST_ARGUMENT = r"\(\s*([A-Za-z_][A-Za-z0-9_.]*)"


def identifier_tokens(text: str) -> List[str]:
    """Every name in ``text`` (columns, functions, numbers), dots kept inside a name: ``"s(day.of.year, k=10)"`` gives ``day.of.year``."""
    return re.findall(IDENTIFIER, text)


def first_arguments(text: str) -> List[str]:
    """The first argument of every call in ``text``: ``"te(s(x.one), s(x.two))"`` gives ``x.one`` and ``x.two`` (and ``s``, ``s``)."""
    return re.findall(FIRST_ARGUMENT, text)
#: </identifier_tokens>
