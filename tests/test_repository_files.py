# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Repository files that contributors and users meet first: the review rules, the templates, and a citation file without placeholders.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not (ROOT / "CITATION.cff").exists(), reason="not running from a repository checkout")


def test_every_path_is_owned_by_the_maintainer():
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    rules = [line.split() for line in owners.splitlines() if line.strip() and not line.startswith("#")]
    assert rules and rules[0][0] == "*" and all(owner.startswith("@") for owner in rules[0][1:])


def test_the_pull_request_template_asks_for_the_checks():
    text = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8").lower()
    for needle in ("tests", "changelog", "math/", "architecture/", "public api", "print()", "public"):
        assert needle in text, needle


def test_the_issue_templates_exist_with_their_front_matter():
    for name in ("bug_report", "feature_request", "research"):
        text = (ROOT / ".github" / "ISSUE_TEMPLATE" / f"{name}.md").read_text(encoding="utf-8")
        assert text.startswith("---\nname:") and "about:" in text.split("---")[1], name


def test_the_citation_file_has_no_placeholder_identifier():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "XXXX" not in text and "joss" not in text.lower()
    assert re.search(r"value:\s*10\.\d{4,9}/\S+", text)


def test_the_public_pages_do_not_present_the_package_as_evaluated_by_a_journal_that_did_not_accept_it():
    for name in ("README.md", "AUTHORS.md", "README_DOC.md", "references.bib"):
        assert "joss" not in (ROOT / name).read_text(encoding="utf-8").lower(), name


def test_contributing_describes_shared_feature_branches_and_the_print_rule():
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Shared Feature Branches" in text and "feat/*" in text and "three working days" in text and "No `print()` in new code" in text
    assert "student" not in text.lower()
