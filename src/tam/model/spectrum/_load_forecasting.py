# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Deprecated module name; use _inertia."""

from ._inertia import InertiaEffect, DegreeDayEffect, SmoothHingeEffect, WarpedCalendarEffect, MacroTrendEffect, ThermalInertiaEffect

__all__ = ["InertiaEffect", "DegreeDayEffect", "SmoothHingeEffect", "WarpedCalendarEffect", "MacroTrendEffect", "ThermalInertiaEffect"]
