# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Logging of the failures the AutoTAM search survives.

A candidate that cannot be fitted is dropped and the search goes on, but the reason must be visible.
"""

import logging
from typing import Optional

#: <failures_log>
def log_failure(logger: logging.Logger, what: str, error: BaseException, formula: Optional[str] = None,
                level: int = logging.WARNING) -> None:
    """Logs a failure that the search survives, with what failed, its formula and the error.

    Args:
        logger: The logger of the calling module.
        what: What failed, for example ``"KalmanTAM expert of Island_R1"``.
        error: The exception that was caught.
        formula: Formula of the failed candidate, when there is one.
        level: Logging level; ``logging.DEBUG`` for failures that are expected in large numbers.
    """
    suffix = f" | formula: {formula}" if formula else ""
    logger.log(level, "%s failed (%s: %s)%s", what, type(error).__name__, error, suffix)
#: </failures_log>
