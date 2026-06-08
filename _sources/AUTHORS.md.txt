# 👥 Authors and Contributions

[`⬅️ README`](README.md) | [`📚 THEORY`](THEORY.md) | [`📝 ACKNOWLEDGEMENTS`](ACKNOWLEDGEMENTS.md)

The **Time series Additive Model (TAM)** project combines foundational theoretical research with modern software engineering to deliver a scalable, interpretable forecasting framework.

⚠️ **Authorship Note (JOSS Compliance):**
The current TAM package is a full re-engineering and scientific extension of the original `weakl` research prototype. The contributions below explicitly distinguish between foundational theory, early prototypes, new theoretical extensions, and the production-grade software architecture.

---

## 🧑‍💻 Core Contributors

### **Yann Allioux**

**Lead Architect, Maintainer & Scientific Co-Author (TAM)**
Primary author of the TAM software framework and co-author of its theoretical extensions.

* **Software & Architecture:** Designed the modular `StaticTAM` OOP system, the Formula API, and the OOM-safe hardware dispatcher (matrix-free Sparse CG solvers, group-chunking).
* **Modeling:** Extended the framework to meta concepts (`AdaptiveTAM`, `OperaTAM`, `KalmanTAM` (BETA), `HierarchicalTAM` (BETA), `NeuralTAM` (EXP) and `AutoTAM` (EXP)).
* **Theoretical Extensions (TAM):** Formalized the "Spectrum" abstraction unifying heterogeneous bases (neural, wavelets, tensors with Kronecker, splines, Chebyshev, trees, radial basis functions, categorical, extended physics, Fourier and linear). Designed Neural Explicit Primal Tensorization (NEPT) and integrated control-theoretic structures (PID) into the Primal RKHS space.

### **Nathan Doumèche**

**Original Researcher & Foundational Theory (WeaKL / PIKL)**
Primary author of the foundational theory on which TAM builds.

* **Theory:** Developed the primal formulation for Kernel Ridge Regression (WeaKL) (Fourier, linear bases), Physics-Informed constraints (PIKL), the Online WeaKL tensorial formulation, and the hierarchical constraints.
* **Prototyping:** Authored the original research prototypes (the [`weakl` package](https://github.com/NathanDoumeche/weakl-package)).
* **Research code:** Co-authored the original research codes (the [`research code`](https://github.com/NathanDoumeche/WeaKL)), branched the model out into specific domains (Tourism, Hierarchical forecasting) and handling the Python/Jupyter side of those experiments.

### **Éloi Bedek**

**Research Engineer (Prototype Stage)**

* **Implementation:** Contributed to real-world dataset validation for the initial `weakl` prototype.
* **Research code:** Co-authored the original research codes (the [`research code`](https://github.com/NathanDoumeche/WeaKL)), heavy lifting on the repository's architecture, the dataset implementations, and the statistical/bootstrap validations.

---

## 🎓 Acknowledgments

### **Yannig Goude**

**Scientific Mentor & Co-Author (TAM Paper)**

* Acknowledged for foundational contributions to time series forecasting, GAMs, and expert aggregation **(including the foundational `opera` R package with Pierre Gaillard, whose theoretical framework is natively re-implemented in Python for `OperaTAM`)**. Serves as scientific mentor and co-author of the TAM scientific manuscript.

---

## 📊 Contribution Summary

To strictly comply with JOSS guidelines, the specific domains of contribution are mapped below:

| Area | Primary Contributors |
| --- | --- |
| **Software Architecture & Engineering** | Yann Allioux |
| **Performance & Optimization on GPU & CPU** | Yann Allioux |
| **TAM Theoretical Extensions** | Yann Allioux |
| **Foundational Theory (WeaKL / PIKL)** | Nathan Doumèche |
| **Early WeaKL Prototype Engineering** | Nathan Doumèche, Éloi Bedek |
| **Original opera R package** | Pierre Gaillard, Yannig Goude |
| **Mentorship & Paper Co-Authorship** | Yannig Goude |

**Authorship Transparency:** This repository (TAM) is primarily authored and maintained by Yann Allioux, expanding upon the foundational WeaKL/PIKL research by Nathan Doumèche and early prototypes assisted by Éloi Bedek. Yannig Goude provides scientific mentorship and paper co-authorship. No honorary authorship is included.

---

*TAM is distributed under the **LGPL-3.0 License**.*