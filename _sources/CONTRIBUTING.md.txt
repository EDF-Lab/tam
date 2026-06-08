
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
* **Type Hinting:** Use strict Python type hints (`typing`) for all function arguments and return types.
* **No Math in Code Comments:** Keep inline Python comments focused strictly on software engineering (e.g., tensor shapes `# Shape: [B, T, D]`, VRAM allocation, PyTorch workarounds). Leave the mathematical proofs to the `math/` Markdown files.

## 4. Submitting a Pull Request (PR)

1. **Fork the repository** and create your branch from `main`.
2. **Implement your feature** and ensure your code follows the coding standards.
3. **Write the documentation pair** in the `math/` and `architecture/` folders.
4. **Compile the docs locally:** Run `build_docs.bat` (Windows) or `build_docs.sh` (Linux/Mac) to ensure the HTML and PDFs compile perfectly without Sphinx or LaTeX warnings.
5. **Submit your PR:** Provide a clear description of the problem solved, the mathematical approach taken, and the performance implications.

---
*By contributing to TAM, you agree that your contributions will be licensed under the project's LGPL license.*