# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
The regression runner (``tests/regression``): how a number becomes a line of text, how a failing check shows up, the frozen dataset, and what the results
file and the workflow must look like. The numbers themselves are not asserted here: they are the contract of the results file, checked by CI.
"""

import ast
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "tests" / "regression"
CHECKS = sorted((FOLDER / "checks").glob("[0-9]*.py"))

pytestmark = pytest.mark.skipif(not FOLDER.exists(), reason="not running from a repository checkout")


def _load(name):
    path = FOLDER / f"{name}.py"
    sys.path.insert(0, str(FOLDER))
    try:
        spec = importlib.util.spec_from_file_location(f"regression_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(FOLDER))
    return module


# ------------------------------------------------------------------ numbers become lines of text
def test_floats_keep_six_significant_digits_and_nothing_spans_two_lines():
    common = _load("common")
    assert common.text(1234.56789) == "1234.57" and common.text(0.000123456789) == "0.000123457" and common.text(-0.0) == "0"
    assert common.text(float("nan")) == "nan" and common.text(float("inf")) == "inf"
    assert common.text(3) == "3" and common.text(np.int64(7)) == "7" and common.text(True) == "True" and common.text(np.bool_(False)) == "False"
    assert "\n" not in common.text("a\nb  c") and common.text("a\nb  c") == "a b c"


def test_a_forecast_is_recorded_as_error_bias_spread_and_quantiles():
    common = _load("common")
    res = common.Results("x")
    res.forecast("f", [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    keys = {k.split(".", 1)[1] for k in res.lines}
    assert keys == {f"f.{s}" for s in ("n", "missing", "rmse", "bias", "mean", "std", "min", "q25", "q50", "q75", "max")}
    assert res.lines["x.f.rmse"] == "1" and res.lines["x.f.missing"] == "0" and res.lines["x.f.max"] == "6"


def test_a_missing_forecast_value_is_counted_not_hidden():
    common = _load("common")
    res = common.Results("x")
    res.forecast("f", [1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
    assert res.lines["x.f.missing"] == "1" and res.lines["x.f.n"] == "3" and res.lines["x.f.rmse"] == "0"


def test_a_difference_that_should_be_zero_is_recorded_as_below_its_tolerance_or_as_itself():
    common = _load("common")
    res = common.Results("x")
    res.gap("tiny", 3e-11)
    res.gap("large", 0.5)
    assert res.lines["x.tiny"] == "<= 1e-06" and res.lines["x.large"] == "0.5"


def test_a_failing_step_becomes_an_error_line_and_the_others_still_run():
    common = _load("common")
    res = common.Results("x")
    res.attempt("bad", lambda: (_ for _ in ()).throw(ValueError("line one\nline two")))
    res.attempt("good", lambda: res.value("good", 1))
    assert res.lines["x.bad"] == "ERROR ValueError: line one line two" and res.lines["x.good"] == "1"


# ------------------------------------------------------------------ the runner
def _toy_checks(folder: Path):
    folder.mkdir()
    (folder / "01_fine.py").write_text("def run(res):\n    res.value('b', 2)\n    res.value('a', 1.5)\n", encoding="utf-8")
    (folder / "02_broken.py").write_text("def run(res):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    (folder / "03_seeded.py").write_text("import numpy as np\n\ndef run(res):\n    res.value('draw', float(np.random.rand()))\n", encoding="utf-8")
    (folder / "04_experimental.py").write_text("EXPERIMENTAL = True\n\ndef run(res):\n    res.value('x', 7)\n", encoding="utf-8")


def test_the_runner_collects_every_check_sorts_the_lines_and_records_a_broken_check(tmp_path):
    run = _load("run")
    _toy_checks(tmp_path / "checks")
    lines, experimental, seconds = run.run_checks(folder=tmp_path / "checks")
    assert set(seconds) == {"01_fine", "02_broken", "03_seeded", "04_experimental"}
    assert experimental == {"04_experimental.x": "7"} and not any(k.startswith("04_") for k in lines)          # the flag moves a check to the experimental file
    assert lines["01_fine.a"] == "1.5" and lines["02_broken.SCRIPT"] == "ERROR RuntimeError: boom"
    assert run.render(lines).splitlines() == sorted(run.render(lines).splitlines())
    assert run.render(lines).startswith("01_fine.a = 1.5\n01_fine.b = 2\n02_broken.SCRIPT = ERROR RuntimeError: boom\n03_seeded.draw = ")


def test_two_runs_give_the_same_text_whatever_ran_in_between(tmp_path):
    run = _load("run")
    _toy_checks(tmp_path / "checks")
    first, first_experimental, _ = run.run_checks(folder=tmp_path / "checks")
    np.random.rand(1000)
    second, second_experimental, _ = run.run_checks(folder=tmp_path / "checks")
    assert run.render(first) == run.render(second) and first_experimental == second_experimental


def test_only_runs_the_checks_whose_name_contains_a_word(tmp_path):
    run = _load("run")
    _toy_checks(tmp_path / "checks")
    lines, _, seconds = run.run_checks(only=["fine"], folder=tmp_path / "checks")
    assert set(seconds) == {"01_fine"} and set(lines) == {"01_fine.a", "01_fine.b"}


def test_main_writes_the_file_it_is_asked_to_with_unix_line_endings(tmp_path, capsys):
    run = _load("run")
    _toy_checks(tmp_path / "checks")
    assert run.main(["--out", str(tmp_path / "out.txt"), "--checks", str(tmp_path / "checks")]) == 0
    data = (tmp_path / "out.txt").read_bytes()
    assert b"\r" not in data and data.endswith(b"\n") and b"ERROR RuntimeError: boom" in data
    assert (tmp_path / "out_EXP.txt").read_text(encoding="utf-8") == "04_experimental.x = 7\n"          # the experimental results go next to the file
    assert "ERROR line(s)" in capsys.readouterr().out


# ------------------------------------------------------------------ the frozen dataset
def test_the_frozen_dataset_is_one_year_of_24_hourly_groups_without_a_hole():
    df = pd.read_csv(FOLDER / "data" / "force_2023.csv", parse_dates=["date"])
    assert list(df.columns) == ["date", "hour", "toy", "day_type_week", "day_type_jf", "temperature", "load", "enedis_total", "enedis_residential",
                                "enedis_professional", "enedis_entreprise", "enedis_industrial"]
    assert len(df) == 8736 and df.isna().sum().sum() == 0 and df["date"].dt.year.unique().tolist() == [2023]
    assert df.groupby("hour").size().unique().tolist() == [364] and df["date"].is_monotonic_increasing and df["date"].is_unique
    segments = df[["enedis_residential", "enedis_professional", "enedis_entreprise", "enedis_industrial"]].sum(axis=1)
    assert float(((df["enedis_total"] - segments).abs() / df["enedis_total"]).max()) < 1e-3          # the hierarchy is (almost exactly) additive


def test_the_frozen_dataset_has_not_been_touched():
    digest = hashlib.sha256((FOLDER / "data" / "force_2023.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    pinned = (FOLDER / "data" / "force_2023.sha256").read_text(encoding="utf-8").split()[0]
    assert digest == pinned, "force_2023.csv changed: rebuild force_2023.sha256 and RESULTS.txt together, on purpose"


def test_the_helpers_split_the_year_without_overlap_and_across_the_weekdays():
    common = _load("common")
    train, test = common.split(common.frame())
    assert set(train["date"]).isdisjoint(set(test["date"])) and len(test) > 1000
    assert set(test["day_type_week"]) == set(range(7)) and set(train["day_type_week"]) == set(range(7))
    first, second = common.chrono_split(common.frame())
    assert first["date"].max() < second["date"].min()


# ------------------------------------------------------------------ the checks
def test_there_is_one_script_per_functionality_numbered_documented_and_runnable():
    assert len(CHECKS) >= 30
    numbers = [p.name.split("_", 1)[0] for p in CHECKS]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)
    for path in CHECKS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"{path.name}: no docstring saying what it checks"
        assert any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body), f"{path.name}: no run(res)"


def test_the_checks_import_only_tam_and_the_scientific_stack_and_download_nothing():
    allowed = {"numpy", "pandas", "torch", "matplotlib", "tam", "common"} | set(sys.stdlib_module_names)
    for path in CHECKS + [FOLDER / "common.py"]:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= allowed, (path.name, sorted(imported - allowed))
        for needle in ("urllib", "requests", "http://", "https://", "zenodo"):
            assert needle not in text, (path.name, needle)


def test_every_family_of_effects_and_every_model_has_a_check():          # noqa: D103
    names = " ".join(p.stem for p in CHECKS)
    for word in ("linear", "spline", "fourier", "chebyshev", "wavelet", "rbf", "tree", "neural", "tensor", "physics", "pid", "fixed_penalty", "auto_fit",
                 "losses", "distributional", "conformal", "extremes", "prediction_api", "adaptive", "kalman", "opera", "hierarchical", "errors", "warnings",
                 "grid_search", "autotam", "plotting", "metrics", "extrapolation", "mixed_decomposition", "interactions"):
        assert word in names, word


# ------------------------------------------------------------------ the results file
def _is_experimental(path: Path) -> bool:
    return "EXPERIMENTAL = True" in path.read_text(encoding="utf-8")


def _file(name):
    return (FOLDER / name).read_text(encoding="utf-8")


def test_the_results_files_are_sorted_one_number_per_line_and_have_no_error():
    for name in ("RESULTS.txt", "RESULTS_EXP.txt"):
        text = _file(name)
        lines = text.splitlines()
        assert lines and lines == sorted(lines) and "\r" not in text and text.endswith("\n"), name
        for line in lines:
            assert re.fullmatch(r"\d\d_[a-z0-9_]+\.[^=\n]+ = .+", line), line
        assert not [l for l in lines if " = ERROR" in l], f"{name}: a committed result must not be an error"


def test_each_results_file_has_the_lines_of_its_checks_and_no_line_of_a_check_that_is_gone():
    stable = {p.stem for p in CHECKS if not _is_experimental(p)}
    experimental = {p.stem for p in CHECKS if _is_experimental(p)}
    assert {line.split(".", 1)[0] for line in _file("RESULTS.txt").splitlines()} == stable
    assert {line.split(".", 1)[0] for line in _file("RESULTS_EXP.txt").splitlines()} == experimental


def test_the_beta_and_experimental_modules_report_apart_from_the_stable_ones():
    experimental = {p.stem.split("_", 1)[1] for p in CHECKS if _is_experimental(p)}
    assert {"kalman", "kalman_per_term_noise", "hierarchical", "neural_tam", "autotam_small", "effect_tree", "effect_physics", "interactions_neural"} <= experimental
    stable = {p.stem.split("_", 1)[1] for p in CHECKS if not _is_experimental(p)}
    assert {"effect_spline", "effect_rbf", "fixed_penalty", "auto_fit_gcv", "opera", "adaptive_windows", "conformal", "extrapolation", "mixed_decomposition", "interactions"} <= stable


def test_the_sequential_checks_build_on_a_base_model_that_does_not_extrapolate():
    text = (FOLDER / "common.py").read_text(encoding="utf-8")
    base = text[text.index("def base_model"):text.index("def with_residual")]
    assert "cyclic=True" in base and "l(temperature)" in base and "s(temperature" not in base and "s(toy" not in base


def test_the_extrapolation_check_fits_winter_and_forecasts_summer():
    text = (FOLDER / "checks" / "35_extrapolation.py").read_text(encoding="utf-8")
    assert "month <= 3" in text and "between(6, 8)" in text
    values = dict(line.split(" = ") for line in _file("RESULTS.txt").splitlines() if line.startswith("35_extrapolation."))
    assert float(values["35_extrapolation.test.share_beyond_training_range"]) > 0.5          # most of the summer is outside the training range


# ------------------------------------------------------------------ the workflow
def test_the_workflow_reruns_every_check_and_fails_when_the_results_file_differs():
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github" / "workflows" / "regression.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"main", "automl", "feat/**"} <= set(workflow[True]["pull_request"]["branches"])
    job = workflow["jobs"]["regression"]
    steps = "\n".join(str(s) for s in job["steps"])
    assert job["name"] == "Regression" and job["timeout-minutes"] <= 30
    assert "tests/regression/run.py" in steps and "git diff --exit-code" in steps and "RESULTS.txt" in steps and "upload-artifact" in steps
    assert "RESULTS_EXP.txt" in steps and "warning" in steps            # an experimental change is shown, it does not fail the job
    assert "pip install -e ." in steps and "requirements" not in steps


# ------------------------------------------------------------------ every argument of every effect and every loss is exercised by some check
ARGUMENTS = {
    "01_effect_linear_categorical": ["n_cat", "topo='nominal'", "topo='ordinal'", "topo='fourier'", "p_order", "extrapolate", "ap="],
    "02_effect_spline": ["k=", "deg=", "p=", "extrapolate='continue'", "extrapolate='constant'", "extrapolate='linear'", "extrapolate='saturation'", "ap="],
    "03_effect_fourier": ["m=", "s=", "cyclic=True", "cyclic=False", "scaled=", "extrapolate=", "ap="],
    "04_effect_chebyshev": ["deg=", "s=", "extrapolate=", "ap="],
    "05_effect_wavelet": ["n_scales", "n_locations", "extrapolate=", "ap="],
    "06_effect_rbf": ["n_centers", "gamma=", "nu=", "others=", "seed=", "extrapolate=", "ap="],
    "07_effect_tree": ["n_trees", "max_depth", "max_leaves", "seed=", "sp_alpha", "split_strategy='quantile'", "split_strategy='uniform'", "others=", "extrapolate=", "ap="],
    "08_effect_linear_tree": ["max_leaves", "max_depth", "slope=", "sp_alpha", "split_strategy", "seed=", "extrapolate=", "ap="],
    "09_effect_neural": ["n_neurons", "act='relu'", "act='tanh'", "act='cos'", "n_hidden_layers", "seed=", "others=", "extrapolate=", "ap="],
    "10_effect_tensor": ["te(s(", "te(l(", "te(c(", "te(p(", "extrapolate=", "ap="],
    "11_effect_physics": ["basis='spline'", "basis='fourier'", "D1=", "D2=", "D3=", "D0=", "n_coeffs", "k=", "extrapolate=", "ap="],
    "12_effect_pid": ["w=", "d_pen=0", "d_pen=", "extrapolate=", "ap="],
    "15_losses": ["loss=\"gamma\"", "loss=\"poisson\"", "loss=\"huber\"", "loss=\"student_t\"", "loss=\"expectile\"", "loss=\"binomial\"", "\"tau\"", "\"delta\"", "\"nu\""],
}


@pytest.mark.parametrize("check", sorted(ARGUMENTS))
def test_every_argument_of_the_effect_is_exercised_by_its_check(check):
    text = (FOLDER / "checks" / f"{check}.py").read_text(encoding="utf-8")
    missing = [argument for argument in ARGUMENTS[check] if argument not in text]
    assert not missing, f"{check}: not exercised: {missing}"

