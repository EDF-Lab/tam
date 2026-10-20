# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Distribution families offered to the static experts, chosen from the profile of the target.

The Gaussian expert (identity link, L2 loss) is always in the pool. With ``family="auto"`` each champion formula is also fitted under the
GLM losses the data allow, so the family competes on the validation rows, scored in the response scale: a log link for a positive skewed
target (Gamma), compound Poisson-gamma for a non-negative target with zeros (Tweedie), Poisson and negative binomial for counts, and a
robust loss (Huber) for heavy tails. A family the data forbid is never proposed: a log link needs a positive target.
"""

from typing import Any, Dict, List

#: <families>
FAMILY_MODES = ("gaussian", "auto")
HEAVY_TAIL_KURTOSIS = 6.0


def check_family_mode(family: str) -> str:
    """Returns ``family`` when it is ``"gaussian"`` (default) or ``"auto"``, else raises a ``ValueError`` that names the two."""
    if family not in FAMILY_MODES:
        raise ValueError(f"family must be one of {FAMILY_MODES}, got {family!r}.")
    return family


def candidate_losses(profile: Dict[str, Any]) -> List[str]:
    """The non-Gaussian losses the profile of the training target allows, most likely first (empty when the Gaussian one is enough).

    Args:
        profile: A target profile (``zero_share``, ``negative_share``, ``integer_share``, ``skewness``, ``kurtosis``, ``hints``).

    Returns:
        Loss names of ``StaticTAM(loss=...)``: ``gamma`` (positive target), ``tweedie`` (non-negative with zeros), ``poisson`` and
        ``negative_binomial`` (counts), ``huber`` (heavy tails).
    """
    if not profile or profile.get("n", 0) < 20 or profile.get("sentinels"):
        return []
    hints = profile.get("hints", ["gaussian"])
    losses: List[str] = []
    if "count" in hints:
        losses += ["poisson", "negative_binomial"]
    elif "tweedie" in hints:
        losses.append("tweedie")
    elif "log" in hints:
        losses.append("gamma")    # positive and skewed: the profile's hint is a log link
    if profile.get("kurtosis", 0.0) > HEAVY_TAIL_KURTOSIS and "cumulative" not in hints:
        losses.append("huber")
    return losses
#: </families>
