# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Every source and test file carries the same header shape: ``SPDX-FileCopyrightText`` lines, ``SPDX-FileContributor`` lines (a name, no
explanation), then ``SPDX-License-Identifier``. Who did what is told in ``AUTHORS.md`` and the CHANGELOG, not in each file.
Files that hold no code and no header (empty ``__init__.py``) are annotated in ``REUSE.toml``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for folder in ("src", "tests") if (ROOT / folder).exists() for p in (ROOT / folder).rglob("*.py"))

pytestmark = pytest.mark.skipif(not FILES, reason="not running from a repository checkout")

COPYRIGHT = re.compile(r"^# SPDX-FileCopyrightText: \d{4}(-\d{4})? \S.*$")
CONTRIBUTOR = re.compile(r"^# SPDX-FileContributor: [^()@<>]+$")
LICENSE = "# SPDX-" + "License-Identifier: LGPL-3.0-or-later"


def _header(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = []
    for line in lines:
        if not line.startswith("# SPDX-"):
            break
        header.append(line)
    return lines, header


def _has_content(path):
    return bool(path.read_text(encoding="utf-8").strip())


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_the_header_has_the_one_shape(path):
    if not _has_content(path):
        pytest.skip("empty file: annotated in REUSE.toml")
    _, header = _header(path)
    assert header, "no SPDX header"
    copyrights = [l for l in header if l.startswith("# SPDX-FileCopyrightText:")]
    contributors = [l for l in header if l.startswith("# SPDX-FileContributor:")]
    assert header == copyrights + contributors + [LICENSE], "order: copyright lines, contributor lines, then the licence"
    assert copyrights and all(COPYRIGHT.match(l) for l in copyrights), copyrights
    assert contributors and all(CONTRIBUTOR.match(l) for l in contributors), contributors
    assert len(set(contributors)) == len(contributors)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_free_text_author_line_remains(path):
    top = "\n".join(path.read_text(encoding="utf-8").splitlines()[:15])
    assert not re.search(r"^# Authors? *:", top, re.M)


def test_the_sorbonne_files_name_the_prototype_authors():
    """The files derived from the prototype and the research code keep the Sorbonne copyright and name their authors."""
    expected = {"src/tam/common/utils.py", "src/tam/model/_data.py", "src/tam/model/_math.py", "src/tam/model/adaptative.py",
                "src/tam/model/additive.py", "src/tam/model/hierarchical.py", "src/tam/model/spectrum/_categorical.py",
                "src/tam/model/spectrum/_fourier.py", "src/tam/model/spectrum/_linear.py", "src/tam/model/spectrum/_physics.py"}
    found = {p.relative_to(ROOT).as_posix() for p in FILES if "Sorbonne" in "\n".join(_header(p)[1])}
    assert found == expected
    for name in expected:
        assert "# SPDX-FileContributor: Nathan Doumèche" in _header(ROOT / name)[1], name


def test_every_empty_python_file_is_annotated_in_reuse_toml():
    reuse = (ROOT / "REUSE.toml").read_text(encoding="utf-8")
    empty = [p.relative_to(ROOT).as_posix() for p in FILES if not _has_content(p)]
    missing = [name for name in empty if f'"{name}"' not in reuse]
    assert not missing, missing


def test_every_licence_used_has_its_text_in_the_licenses_folder():
    used = set(re.findall(r'SPDX-License-Identifier = "([^"]+)"', (ROOT / "REUSE.toml").read_text(encoding="utf-8")))
    used |= {m for p in FILES for m in re.findall(r"SPDX-License-Identifier: (\S+)", "\n".join(_header(p)[1]))}
    assert {"LGPL-3.0-or-later", "CC-BY-4.0", "CC0-1.0"} <= used
    missing = sorted(name for name in used if not (ROOT / "LICENSES" / f"{name}.txt").exists())
    assert not missing, missing
    unused = sorted(p.stem for p in (ROOT / "LICENSES").glob("*.txt") if p.stem not in used)
    assert not unused, unused


def test_every_data_file_is_annotated_in_reuse_toml():
    reuse = (ROOT / "REUSE.toml").read_text(encoding="utf-8")
    data = sorted(p.relative_to(ROOT).as_posix() for folder in ("src/tam/data", "tests/regression/data") for p in (ROOT / folder).glob("*.csv"))
    assert data, "no data file found"
    missing = [name for name in data if f'"{name}"' not in reuse]
    assert not missing, missing
