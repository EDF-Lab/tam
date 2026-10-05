# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
The use-case workflow: every script of ``use_cases/`` runs headless on each push and pull request, one job per script, and one required check
says whether they all passed.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "use_cases.yml"
LISTER = ROOT / ".github" / "scripts" / "list_use_cases.py"

pytestmark = pytest.mark.skipif(not WORKFLOW.parent.exists(), reason="not running from a repository checkout")


def _workflow():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _lister():
    spec = importlib.util.spec_from_file_location("list_use_cases", LISTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_use_case_script_is_listed_and_the_helpers_are_not():
    scripts = _lister().use_case_scripts(ROOT / "use_cases")
    on_disk = {p.relative_to(ROOT / "use_cases").as_posix() for p in (ROOT / "use_cases").rglob("*.py")}
    assert scripts, "no use case found"
    assert {"cheatsheet.py", "air_passengers.py", "readme.py"} <= set(scripts)
    assert set(scripts) <= on_disk
    assert "utils_cases.py" not in scripts and not any(s.startswith(("temp/", "tests/")) for s in scripts)
    assert scripts == sorted(scripts)


def test_a_new_use_case_is_picked_up_without_editing_the_workflow(tmp_path):
    (tmp_path / "paper_reproductions").mkdir()
    (tmp_path / "temp").mkdir()
    for name in ("a.py", "paper_reproductions/b.py", "utils_cases.py", "temp/scratch.py", "__init__.py"):
        (tmp_path / name).write_text("")
    assert _lister().use_case_scripts(tmp_path) == ["a.py", "paper_reproductions/b.py"]


def test_the_lister_prints_the_matrix_json_the_workflow_reads():
    out = subprocess.run([sys.executable, str(LISTER), str(ROOT / "use_cases")], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == _lister().use_case_scripts(ROOT / "use_cases")


def test_the_workflow_runs_on_pushes_and_pull_requests_of_the_protected_and_release_branches():
    triggers = _workflow()[True]                                   # PyYAML reads the key `on` as the boolean True
    for event in ("push", "pull_request"):
        branches = triggers[event]["branches"]
        assert {"main", "automl", "feat/**"} <= set(branches), event
    assert "release/**" in triggers["push"]["branches"]


def test_one_job_per_script_runs_headless_from_a_scratch_folder():
    workflow = _workflow()
    job = workflow["jobs"]["run"]
    assert "fromJSON" in str(job["strategy"]["matrix"]["script"]) and job["strategy"]["fail-fast"] is False
    assert job["timeout-minutes"] <= 30
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "MPLBACKEND: Agg" in text and "RUNNER_TEMP" in text
    assert "pip install -e . -r use_cases/requirements.txt" in text


def test_one_aggregating_check_fails_when_any_script_fails():
    job = _workflow()["jobs"]["use-cases"]
    assert job["name"] == "Use cases" and job["needs"] == ["list", "run"] and "always()" in job["if"]
    assert "needs.run.result" in WORKFLOW.read_text(encoding="utf-8")


def test_every_package_a_use_case_imports_is_installed_by_tam_or_by_the_use_case_requirements():
    import ast

    installed = {"tam", "torch", "numpy", "pandas", "scipy", "matplotlib", "psutil"}
    listed = {re.split(r"[\[<>=]", line.strip())[0].lower() for line in (ROOT / "use_cases" / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")}
    names = {"ipython": "IPython"}
    installed |= {names.get(package, package) for package in listed}
    local = {p.stem for p in (ROOT / "use_cases").rglob("*.py")}
    missing = set()
    for script in (ROOT / "use_cases").rglob("*.py"):
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            for module in modules:
                top = module.split(".")[0]
                if top not in installed and top not in local and top not in sys.stdlib_module_names:
                    missing.add((script.name, top))
    assert not missing, f"imported by a use case but installed by nothing: {sorted(missing)}"


def test_the_packages_the_use_cases_compare_with_are_listed_once_in_the_use_cases_folder_not_in_the_package():
    packages = ("statsmodels", "tabicl", "pygam")
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not re.search(r"^bench\s*=", metadata, re.M)
    listed = (ROOT / "use_cases" / "requirements.txt").read_text(encoding="utf-8")
    root = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for package in packages:
        assert package in listed, package
        assert package not in metadata, package
        assert not re.search(rf"^{package}", root, re.M), package
        assert f"pip install \"{package}" not in workflow, package
