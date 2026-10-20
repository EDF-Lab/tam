# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Stable identifiers for the experts of a fitted ``AutoTAM``.

The name of an expert (``Adaptive_AR_Only_168p_ChebyshevIsland_R1``) says where the search found it: the island, the round, the rank. The same
formula found on another window, or in another order, gets another name, so a caller that follows an expert from one window to the next
cannot use it. The identifier is a short hash of what the expert is: its formula, its kind of expansion and the settings of that expansion.
Two fits that reach the same formula and the same expansion give the same identifier.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: <expert_ids>
ID_PREFIX = "ex_"
ID_LENGTH = 10
_STATIC_TAIL = re.compile(r"^(?:Static|Native)_.+?_R\d+(?:_(.*))?$")


def normalise_formula(formula: Optional[str]) -> str:
    """The formula without blanks, its top-level terms sorted (the order of the terms does not change the model)."""
    text = re.sub(r"\s+", "", str(formula or ""))
    if "~" not in text:
        return text
    left, right = text.split("~", 1)
    terms, depth, current = [], 0, ""
    for char in right:
        depth += char == "("
        depth -= char == ")"
        if char == "+" and depth == 0:
            terms.append(current)
            current = ""
        else:
            current += char
    terms.append(current)
    return left + "~" + "+".join(sorted(terms))


def _significant(value: float) -> str:
    return f"{float(value):.2g}"


def expansion_of(record: Dict[str, Any]) -> str:
    """What was done to the formula: ``static:<kind>`` (prior, auto, grid, family loss, native), ``kalman:<noise>`` (process noise relative to
    the observation noise, so that it does not follow the scale of the data) or ``adaptive:<architecture>:<window>p``."""
    kind, params, name = record.get("type"), record.get("params") or {}, str(record.get("name", ""))
    if kind == "external":
        return "external"
    if kind == "kalman":
        observation = float(params.get("observation_noise_var") or 0.0)
        rate = float(params.get("process_noise_var") or 0.0)
        return "kalman:" + _significant(rate / observation if observation > 0 else rate)
    if kind == "adaptive":
        return f"adaptive:{params.get('architecture', '')}:{params.get('training_window_periods', '')}p"
    match = _STATIC_TAIL.match(name)
    tail = (match.group(1) if match else None) or ("native" if name.startswith("Native_") else "plain")
    return f"static:{tail}"


def base_formula_of(record: Dict[str, Any]) -> str:
    """The formula of the static model under the expert."""
    model = record.get("model")
    base = model if record.get("type") == "static" else getattr(model, "_saved_base_model", getattr(model, "base_model_", None))
    return str(getattr(base, "formula_", "") or "")


def identifier_of(record: Dict[str, Any]) -> str:
    """The identifier of a trained expert record: ``ex_`` and ten hexadecimal digits of the hash of formula, expansion and dynamic formula.
    An external expert (forecast made by the caller) is known by its given name."""
    if record.get("type") == "external":
        return str(record["name"])
    parts = [normalise_formula(base_formula_of(record)), expansion_of(record), normalise_formula(record.get("dynamic_formula") if record.get("type") != "static" else "")]
    return ID_PREFIX + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:ID_LENGTH]


def identifiers(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """``{expert name: identifier}``. Two experts with the same formula and expansion (found by two islands) get the same hash: the later ones
    are numbered (``_2``, ``_3``) so that a column named by identifier is never ambiguous."""
    out: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for record in records:
        base = identifier_of(record)
        seen[base] = seen.get(base, 0) + 1
        out[str(record["name"])] = base if seen[base] == 1 else f"{base}_{seen[base]}"
    return out


def experts_table(records: List[Dict[str, Any]], weights: Dict[str, float], complete: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
    """One row per trained expert: identifier, name, island, formula, expansion, validation score, Apex weight (0 outside the Apex; the
    weights of the Apex members sum to 1), and whether the forecast of the expert was complete on the rows of the last prediction (``None``
    before one)."""
    ids = identifiers(records)
    total = float(sum(w for w in weights.values() if np.isfinite(w) and w > 0))
    rows = []
    for record in records:
        name = str(record["name"])
        weight = float(weights.get(name, 0.0))
        rows.append({
            "identifier": ids[name], "name": name, "island": record.get("island"), "formula": base_formula_of(record),
            "expansion": expansion_of(record), "dynamic_formula": record.get("dynamic_formula") if record.get("type") != "static" else None,
            "validation_score": record.get("val_rmse"), "apex_weight": weight / total if total > 0 and weight > 0 else 0.0,
            "complete": None if complete is None else complete.get(name),
        })
    return pd.DataFrame(rows, columns=["identifier", "name", "island", "formula", "expansion", "dynamic_formula", "validation_score", "apex_weight", "complete"])
#: </expert_ids>
