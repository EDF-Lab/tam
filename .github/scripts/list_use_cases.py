"""Prints the use-case scripts of a folder as a JSON list (the matrix of the use-case workflow).

    python .github/scripts/list_use_cases.py use_cases

Every ``.py`` file is a use case, except the shared helpers (``utils_cases.py``, ``__init__.py``) and the ``temp/`` and ``tests/`` folders.
"""
import json
import sys
from pathlib import Path

HELPERS = {"utils_cases.py", "__init__.py"}
SKIPPED_FOLDERS = {"temp", "tests", "__pycache__"}


def use_case_scripts(folder) -> list:
    folder = Path(folder)
    found = []
    for path in folder.rglob("*.py"):
        relative = path.relative_to(folder)
        if path.name in HELPERS or SKIPPED_FOLDERS & set(relative.parts[:-1]):
            continue
        found.append(relative.as_posix())
    return sorted(found)


if __name__ == "__main__":
    print(json.dumps(use_case_scripts(sys.argv[1] if len(sys.argv) > 1 else "use_cases")))
