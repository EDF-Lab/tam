
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
* **Type Hinting:** Use strict Python type hints (`typing`) for all function arguments and return types.
* **No Math in Code Comments:** Keep inline Python comments focused strictly on software engineering (e.g., tensor shapes `# Shape: [B, T, D]`, VRAM allocation, PyTorch workarounds). Leave the mathematical proofs to the `math/` Markdown files.
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
5. **Submit your PR:** Provide a clear description of the problem solved, the mathematical approach taken, and the performance implications.

---
*By contributing to TAM, you agree that your contributions will be licensed under the project's LGPL license.*