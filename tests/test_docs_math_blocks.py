# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
A display equation of a documentation page must be separated from the paragraph text above and below it by a blank line.

Written right after a paragraph line, ``$$ ... $$`` is read by MyST as two inline-math delimiters: the published page shows literal dollar
signs and shifted ``\( ... \)`` spans, and Markdown eats the backslash of ``\,``. The same holds inside a list item, where the equation
needs a blank line before it too. Fenced code is not checked.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(
    [p for folder in ("math", "architecture") if (ROOT / folder).exists() for p in (ROOT / folder).rglob("*.md")]
    + [ROOT / name for name in ("THEORY.md", "README.md", "EXAMPLES.md") if (ROOT / name).exists()]
)
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s")


def misplaced_display_math(text: str) -> list:
    """Return ``(line, reason)`` for each display equation that touches paragraph text (1-based line numbers)."""
    lines = text.splitlines()
    findings = []
    in_fence = in_block = False
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if in_block:
            if stripped.endswith("$$"):
                in_block = False
                if index + 1 < len(lines) and _is_text(lines[index + 1]):
                    findings.append((index + 1, "text directly below the display equation"))
            continue
        if not stripped.startswith("$$"):
            continue
        single = stripped.endswith("$$") and len(stripped) > 2
        if index > 0 and _is_paragraph_line(lines[index - 1]):
            findings.append((index + 1, "text directly above the display equation"))
        if single:
            if index + 1 < len(lines) and _is_text(lines[index + 1]):
                findings.append((index + 1, "text directly below the display equation"))
        else:
            in_block = True
    return findings


def _is_paragraph_line(line: str) -> bool:
    """A non-blank line that is paragraph text; the first line of a list item is one."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("#", "$$", "```", "~~~", ">", "|", ":::"))


def _is_text(line: str) -> bool:
    """Paragraph text that would continue the paragraph of an equation above it: not the marker line of a new list item."""
    return _is_paragraph_line(line) and not LIST_ITEM.match(line)


def test_a_display_equation_right_after_a_paragraph_line_is_reported():
    page = "Intro text\n$$ a = b $$\n\nMore text\n"
    assert misplaced_display_math(page) == [(2, "text directly above the display equation")]


def test_a_display_equation_right_before_a_paragraph_line_is_reported():
    page = "Intro text\n\n$$ a = b $$\ntext below\n"
    assert misplaced_display_math(page) == [(3, "text directly below the display equation")]


def test_both_sides_are_reported_for_a_multi_line_block():
    page = "Intro text\n$$\na = b\n$$\ntext below\n"
    assert misplaced_display_math(page) == [(2, "text directly above the display equation"),
                                            (4, "text directly below the display equation")]


def test_separated_blocks_and_the_closing_line_of_a_block_are_not_reported():
    page = "Intro text\n\n$$\n\\text{E: } r = 1, \\qquad\n\\text{M: } s = 2\n$$\n\nMore text\n\n$$ c = d $$\n"
    assert misplaced_display_math(page) == []


def test_an_equation_in_a_list_item_needs_a_blank_line_before_it():
    page = "1. item\n   $$ a = b $$\n2. next\n"
    assert misplaced_display_math(page) == [(2, "text directly above the display equation")]
    assert misplaced_display_math("1. item\n\n   $$ a = b $$\n\n2. next\n") == []


def test_headings_fences_and_the_next_list_item_are_not_reported():
    page = "## Heading\n$$ a = b $$\n\n```\ntext\n$$ x $$\ntext\n```\n\n1. item\n\n   $$ c = d $$\n2. next\n"
    assert misplaced_display_math(page) == []


@pytest.mark.skipif(not PAGES, reason="not running from a repository checkout")
@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_display_equation_of_the_documentation_is_set_apart(path):
    assert misplaced_display_math(path.read_text(encoding="utf-8")) == []
