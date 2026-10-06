# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runs every check and writes the results file.

    python tests/regression/run.py                       # rewrites tests/regression/RESULTS.txt
    python tests/regression/run.py --out /tmp/other.txt  # somewhere else (to compare two versions of tam)
    python tests/regression/run.py --only spline fourier

Each check is a module of ``checks/`` with ``run(res)``; one that covers a beta or experimental module sets ``EXPERIMENTAL = True`` and reports into
``RESULTS_EXP.txt`` (a change there is shown but does not fail CI). The results file holds one ``<check>.<name> = <value>`` line per number, sorted, so that
``git diff tests/regression/RESULTS.txt`` shows exactly what a change moved. It carries no version and no timing: it only changes when a result does.
The same file is the contract in CI (``.github/workflows/regression.yml``): a pull request that moves a number commits the new file, and the diff is
reviewed like any other change.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402


def load_check(path: Path):
    spec = importlib.util.spec_from_file_location(f"check_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(only: list[str] | None = None, folder: Path = HERE / "checks") -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    """(stable lines, experimental lines, seconds per check). A check that sets ``EXPERIMENTAL = True`` reports into the experimental file."""
    lines: dict[str, str] = {}
    experimental: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for path in sorted(folder.glob("[0-9]*.py")):
        if only and not any(word in path.stem for word in only):
            continue
        res = common.Results(path.stem)
        common.seed()
        start = time.perf_counter()
        flagged = False
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            try:
                module = load_check(path)
                flagged = bool(getattr(module, "EXPERIMENTAL", False))
                module.run(res)
            except Exception as error:                      # noqa: BLE001 - a broken check is a result, not a crash of the run
                res.lines[f"{path.stem}.SCRIPT"] = common.text(f"ERROR {type(error).__name__}: {str(error)[:90]}")
        seconds[path.stem] = time.perf_counter() - start
        (experimental if flagged else lines).update(res.lines)
    return lines, experimental, seconds


def render(lines: dict[str, str]) -> str:
    return "".join(f"{key} = {lines[key]}\n" for key in sorted(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "RESULTS.txt", help="the results file; the experimental results go next to it, as <name>_EXP.txt")
    ap.add_argument("--only", nargs="*", help="run only the checks whose name contains one of these words")
    ap.add_argument("--checks", type=Path, default=HERE / "checks", help="the folder of checks")
    a = ap.parse_args(argv)
    stable, experimental, seconds = run_checks(a.only, a.checks)
    a.out.write_text(render(stable), encoding="utf-8", newline="\n")
    experimental_out = a.out.with_name(f"{a.out.stem}_EXP{a.out.suffix}")
    experimental_out.write_text(render(experimental), encoding="utf-8", newline="\n")
    errors = sum(1 for v in {**stable, **experimental}.values() if v.startswith("ERROR"))
    print(f"{len(stable)} stable and {len(experimental)} experimental lines from {len(seconds)} checks in {sum(seconds.values()):.0f} s, "
          f"{errors} ERROR line(s) -> {a.out}, {experimental_out.name}")
    for name, took in sorted(seconds.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  slowest: {name} {took:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
