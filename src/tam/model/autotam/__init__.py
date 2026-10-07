# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Automated TAM (AutoTAM) module.
"""

from .parser import (
    canonicalize_term,
    terms_are_equivalent,
    term_subsumes,
    canonicalize_formula,
    FormulaParser,
    parse_formula_to_terms,
)

__all__ = [
    "canonicalize_term",
    "terms_are_equivalent",
    "term_subsumes",
    "canonicalize_formula",
    "FormulaParser",
    "parse_formula_to_terms",
]
