
# 🤝 Contributing to Time series Additive Model (TAM)

[`⬅️ README`](README.md) | [`📚 THEORY`](THEORY.md) | [`📘 DOCS GUIDE`](README_DOC.md) | [`🆕 CHANGELOG`](CHANGELOG.md)

Thank you for your interest in contributing to TAM! We welcome contributions from the community, whether you are a mathematician adding a new effect or meta-model, or a software engineer optimizing our PyTorch tensor operations.

To ensure the framework remains mathematically rigorous and computationally stable, we ask all contributors to strictly follow the guidelines below.

---

## 1. Where to Start

1. **Check Existing Issues:** Before starting to code, please check our [Issues tracker] to see if someone else is already working on the feature or bug.
2. **Open a New Issue:** If you want to add a new **Meta-Model** or **Effect**, please open an issue first to discuss the mathematical validity and implementation strategy with the maintainers.

## 2. The "Mirror Architecture" (Documentation)

Because TAM serves two distinct audiences (mathematicians and software engineers), we enforce a strict **Mirror Architecture** for our documentation. You must separate mathematical proofs from Python engineering.

When you add a new feature, you must write a pair of Markdown files:

* 🧠 **`math/` (The Brain):** Explains *why* the formula is exact. Contains the theory, LaTeX equations (`$$...$$`), theorems, and academic citations `` {cite:p}`key` ``. Do not put code here.
* 💻 **`architecture/` (The Hands):** Explains *how* the Python script calculates the formula without crashing. Contains PyTorch implementation details, OOP structure, and VRAM management. 

### 🏛️ Documentation Golden Rules
* **Read the full guide:** Please read our detailed [**Documentation Generation Guide**](README_DOC.md) before writing documentation.
* **No Redundancy:** The `architecture/` files must *never* re-demonstrate the math. Link back to the `math/` files.
* **Code Extraction:** Do not hard-copy Python code into Markdown. Use the Sphinx `{literalinclude}` directive to pull code dynamically from the source files.

## 3. Coding Standards

Our core engine is built for industrial-grade performance on massive datasets. If you are modifying the Python source code in `src/tam/`:

* **OOM Safety:** Ensure your PyTorch implementations strictly respect Out-Of-Memory (OOM) safety protocols. Matrix operations across the group dimension must support dynamic chunking (`try/except` memory routing).
* **Vectorization:** Avoid standard Python `for` loops. Exploit N-dimensional broadcasting and PyTorch tensorization.
* **Docstrings:** We use **Google-style docstrings**. Ensure your functions and classes are fully documented, as Sphinx uses `autodoc` and `napoleon` to generate the API reference automatically.
* **Extend the public API, never change it:** `tests/test_api_contracts.py` pins the signatures of the main models (`StaticTAM`, `AdaptiveTAM`, `KalmanTAM`, `OperaTAM`, `HierarchicalTAM`) and of the formula parser, and the naming of `decompose_prediction` columns and `summary()` keys. Do not rename, remove, reorder or change the default of an existing parameter; add a keyword argument with a default after the existing ones. A deliberate API change updates the pin in the same commit and says so in the pull request.
* **No `print()` in new code:** the library does not print; use warnings for something the user must act on, and exceptions for errors. A logging refactor of the existing code comes later.
* **Headers and credit:** every `.py` file of `src/` and `tests/` starts with the same SPDX header (`SPDX-FileCopyrightText`, then one `SPDX-FileContributor: Name` line per author, then `SPDX-License-Identifier`; copy the header of a neighbouring file). Add an `SPDX-FileContributor` line, a name and nothing else, when you write a substantial part of a file; credit smaller contributions in the CHANGELOG. `AUTHORS.md` tells who did what; `tests/test_spdx_headers.py` fails on a file with another shape.
* **Type Hinting:** Use strict Python type hints (`typing`) for all function arguments and return types.
* **Math in Code: Notation Only:** Keep inline Python comments focused on software engineering (e.g., tensor shapes `# Shape: [B, T, D]`, VRAM allocation, PyTorch workarounds). Docstrings in `src/tam/` may use *minimal* LaTeX notation when one expression states what a function computes more precisely than prose, e.g. a raw docstring containing ``:math:`\hat{\theta} = (\Phi^\top \Phi + nP)^{-1} \Phi^\top Y` ``. Keep it to a defining expression; derivations, proofs and citations stay in the `math/` Markdown files.
* **Chronological Tensor Alignment:** When projecting Pandas DataFrames into PyTorch tensors for time-series formulas, row-adjacency must equal chronological adjacency. Always apply `.sort_values(date_col)` before tensor stacking or truncating (e.g., before `.groupby().head()`), and re-sort the target Pandas indices chronologically before mapping tensor predictions back, to prevent silent row-scrambling on unsorted input.

## 4. Shared Feature Branches

A body of work too large for one pull request (a new model family, a research collaboration, a team or course project) lives on a long-lived `feat/*` branch of the main repository, kept by the maintainer.

1. **Fork** the repository and create your working branch from the `feat/*` branch of the project you join.
2. **Open your pull request into that `feat/*` branch**, never straight into `main`; the `feat/*` branch joins `main` through its own pull request, at a release.
3. **Review rhythm:** the maintainer reviews pull requests in two fixed slots a week and answers within three working days; keep pull requests small so a review fits a slot.
4. Follow the rules of the project's integration guide, shared with its collaborators: branch naming, what is merged when, and how the `feat/*` branch joins `main`.

## 5. Submitting a Pull Request (PR)

1. **Fork the repository** and create your branch from `main`.
2. **Implement your feature** and ensure your code follows the coding standards.
3. **Write the documentation pair** in the `math/` and `architecture/` folders.
4. **Compile the docs locally:** Run `build_docs.bat` (Windows) or `build_docs.sh` (Linux/Mac) to ensure the HTML and PDFs compile perfectly without Sphinx or LaTeX warnings.
5. **Run the regression checks:** `python tests/regression/run.py` rewrites `tests/regression/RESULTS.txt` and `RESULTS_EXP.txt`; `git diff tests/regression/` shows what your change moved. If that is what you intended, commit the new file in your pull request and say why the numbers moved; the `Regression` workflow fails when `RESULTS.txt` in the pull request is not the one the checks produce (a different `RESULTS_EXP.txt` is only a warning) (see `tests/regression/README.md`).
6. **Submit your PR:** Provide a clear description of the problem solved, the mathematical approach taken, and the performance implications.

---

## 6. Licence of Contributions and Sign-off (DCO)

By contributing, you agree that your contribution is licensed under the licence of the file it changes (LGPL-3.0-or-later for the code, see `LICENSES/` and `REUSE.toml` for the data and documents).

Every commit must carry a sign-off line, which certifies the [Developer Certificate of Origin 1.1](DCO):

```
Signed-off-by: Your Name <your.email@example.com>
```

`git commit -s -m "short phrase"` adds it. The `DCO` check of a pull request fails when a commit lacks it. How to fix a missing sign-off: [docs/contributing/dco_howto.md](docs/contributing/dco_howto.md).
