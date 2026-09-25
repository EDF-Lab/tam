# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Automated TAM (AutoTAM) module.
"""

from .parser import (
    canonicalize_term,
    terms_are_equivalent,
    canonicalize_formula,
    FormulaParser,
    parse_formula_to_terms,
)

__all__ = [
    "canonicalize_term",
    "terms_are_equivalent",
    "canonicalize_formula",
    "FormulaParser",
    "parse_formula_to_terms",
]
