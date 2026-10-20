# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The ``search="fast"`` preset of AutoTAM: a documented bundle of values of existing settings, not a second code path.

``search="full"`` (default) is the search as it always was. ``search="fast"`` keeps four effect families, a smaller population and fewer
generations, three Kalman noise rates and two adaptive windows, and switches the seasonal anchor on. An argument given by the caller
(``pop_size``, ``n_experts``, ``seasonal_anchor``) wins over the preset.
"""

#: <search_preset>
SEARCH_MODES = ("full", "fast")

FAST_ISLANDS = ("LinearIsland", "SplineIsland", "FourierIsland", "ChebyshevIsland")
FAST_POP_SIZE = 16
FAST_N_EXPERTS = 8
FAST_GENERATIONS = 8
FULL_GENERATIONS = 20
FAST_KALMAN_RATE_COUNT = 3
FAST_ADAPTIVE_WINDOW_COUNT = 2


def check_search_mode(search: str) -> str:
    """Returns ``search`` when it is ``"full"`` or ``"fast"``, else raises a ``ValueError`` that names the two."""
    if search not in SEARCH_MODES:
        raise ValueError(f"search must be one of {SEARCH_MODES}, got {search!r}.")
    return search
#: </search_preset>
