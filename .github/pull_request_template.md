## What and why
<!-- The problem solved, the approach taken, the performance implications. -->

## Checklist
Fill one row per line you add to `CHANGELOG.md`. "Not needed" is an answer only with its reason.

| CHANGELOG entry | Test (name; fails before the change?) | `math/` page | `architecture/` page | User docs (`README`, `THEORY`, docstrings) | Callers checked | Regression |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

- [ ] Tests pass locally (`pytest`), and the new tests were written first and fail without the change.
- [ ] The documentation pair is updated (`math/` for theory, `architecture/` for the implementation) and the docs build without warnings.
- [ ] One `CHANGELOG.md` line per user-visible change, under `[Unreleased]`.
- [ ] No frozen public API change: existing parameters keep their name, order and default (`tests/test_api_contracts.py`); new ones are keyword arguments with a default.
- [ ] No `print()` in new code.
- [ ] Data used by tests and examples is public.
- [ ] Every added file keeps its `SPDX-FileCopyrightText` and `SPDX-License-Identifier` header; its author is named as `SPDX-FileContributor`.

## Pre-push verdict
<!-- Paste the verdict of the regression check if you ran it (READY TO PUSH or the failing step). -->
