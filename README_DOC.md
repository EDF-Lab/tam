# 📘 Documentation Generation Guide (TAM)

[⬅️ `README`](README.md) 

This guide explains how to build the Time series Additive Model (TAM) technical documentation from scratch on a local machine.

Our "Docs as Code" pipeline generates three distinct products from a single source of truth:
1. 🌐 **A Comprehensive HTML Website** (Includes everything: API, Theory, and Code).
2. 🧠 **A Mathematical Theory PDF** (Tailored for researchers and academic publications).
3. 💻 **An Architecture & Code PDF** (Tailored for software engineers and auditors).

---

## Prerequisites

Before generating the documentation, ensure you have the following installed:

1. **Python 3.10+**

2. **A LaTeX Distribution (Required for PDFs):**

    * **Windows:** Download and install [MiKTeX for Windows](https://miktex.org/download).
        * *Important:* During MiKTeX installation, ensure the option **"Install missing packages on-the-fly"** is set to **"Yes"**. Sphinx will automatically attempt to download the LaTeX packages it needs during the build.
   * **Mac/Linux:** Install [MacTeX](https://tug.org/mactex/) (macOS) or [TeX Live](https://tug.org/texlive/) (Linux).

---

## Creating the Workspace

To avoid conflicts with other Python projects, we will create an isolated virtual environment.
Open a terminal (Command Prompt `cmd`, PowerShell, or your OS equivalent) at the root of the project and run:

```bash
python -m venv .venv
```
*(Note: If `python` is not in your PATH, provide the full path to your Python executable, e.g., `C:/Users/USER/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv`)*

This creates a `.venv/` folder containing a clean Python installation.

---

## Activating the Environment

Activate the environment to work inside it.

**On Windows (CMD):**
```bat
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**On Mac/Linux:**
```bash
source .venv/bin/activate
```

> ✅ Once activated, you should see `(.venv)` displayed at the beginning of your command line prompt.

---

## Installing Dependencies

We need to install **Sphinx** (the documentation generator), the **MyST Parser** (for Markdown support), **BibTeX** (for academic citations), and the TAM package itself (so Sphinx can read the Python docstrings).

Run this command at the root of the project:

```bash
pip install -r requirements.txt
```
*(Note: The repository's `requirements.txt` is already configured with `sphinx`, `myst-parser`, `sphinxcontrib-bibtex`, `sphinx-autodoc-typehints` and all necessary TAM dependencies.)*

---

## Building the Documentation

Once the installation is complete, run the automation script located in your working directory:

**On Windows:**
```bash
.\build_docs.bat
```

**On Mac/Linux:** *(Assuming you have an equivalent shell script)*
```bash
./build_docs.sh
```

This script will automatically:
1. **Clean** previous builds to prevent caching issues.
2. **Scan** the Python source code (`src/tam`) to auto-generate the API reference.
3. **Build** the global HTML website.
4. **Build** the Mathematical Theory LaTeX files and compile them into a PDF.
5. **Build** the Architecture & Code LaTeX files and compile them into a PDF.

---

## Viewing the Results

If the generation is successful, you will find your two distinct documentation formats in the `docs/build/` directory. You can open them directly:

* HTML 🌐 **Interactive Website:** `docs/build/html/index.html`
* PDF 🧠 **Theory (Researchers)** & 💻 **Architecture (Engineers):** `docs/build/latex_theory/TAM_documentation.pdf`

---

## 6. Troubleshooting PDF Generation (Corporate Environments)

Building the interactive HTML website is entirely handled by Python and Sphinx. However, generating the academic and architectural PDFs involves a more complex pipeline relying on a LaTeX engine (like MiKTeX). 

In a corporate IT environment, strict network firewalls and restricted user privileges can disrupt this pipeline. To successfully generate the PDFs, your system requires three key elements:

**1. System PATH Recognition**
The terminal running the build script must be able to locate the LaTeX compiler. If you just installed your LaTeX distribution, you must completely restart your terminal or IDE (like VSCode) so it can recognize commands like `pdflatex`.

**2. Dynamic Package Downloading ("On-the-fly")**
To format the PDFs correctly, Sphinx relies on specific LaTeX styling packages (such as `cmap.sty`, `times`, `fncychap`, `tabulary`, etc.). Your LaTeX distribution must be configured to download and install missing packages automatically in the background during the build process.

**3. Network Access and User Permissions**
This is the most common bottleneck in enterprise environments. Automatic background downloads often fail because:
* **Admin Rights:** Background installations might be blocked if you lack administrator privileges. You can bypass this by ensuring your LaTeX package manager (e.g., MiKTeX Console) is set to operate strictly in **"User mode"**, which installs packages locally in your `AppData` folder.
* **Firewalls & Proxies:** Corporate firewalls may block connections to default LaTeX servers, resulting in "Timeout" errors. If this happens, you will need to open your LaTeX console, change the remote package repository to an alternative mirror (e.g., an HTTPS server in your country), or configure your company's proxy settings directly within the LaTeX console to allow manual package installations.

---

## 🏗️ Quick Guide: How to Contribute

To maintain crystalline clarity and respond to two very different audiences (researchers/mathematicians vs. software engineers), TAM uses a strict **Mirror Architecture** for its documentation. 

We divide the documentation into two sealed worlds. When you add a new feature, you must write a pair of files:
1. 🧠 **`math/` (The Brain):** Explains *why* the formula is exact. Contains theory, LaTeX equations, theorems, and academic citations. 
2. 💻 **`architecture/` (The Hands):** Explains *how* the Python script calculates the formula without crashing. Contains PyTorch implementation details, OOP structure, VRAM management, and `{literalinclude}` code extraction.

### 🏛️ The Golden Rules of Writing (Sphinx/MyST)

To ensure the pipelines compile our PDFs and HTML flawlessly, all contributors must adhere to these strict writing rules:

* **Separation of Code Comments vs. Markdown Theory:** Because the `architecture/` Markdown files dynamically pull source code via `{literalinclude}`, your Python docstrings (`r"""..."""`) and inline comments must focus *strictly* on software engineering (e.g., tensor shapes, VRAM allocation, OOM prevention, PyTorch workarounds). Do **not** write LaTeX mathematical proofs or academic citations inside the `.py` files. Let the `math/` Markdown files carry that burden.
* **Zero Redundancy:** The `architecture/` files must *never* re-demonstrate the math. Instead, use clean relative links to point to the theory (e.g., `[See the theory](../../math/core/01_primal_model.md)`).
* **Code Extraction:** **Do not** hard-copy and paste PyTorch code into Markdown files. Exclusively use the Sphinx `{literalinclude}` directive with exact relative paths (e.g., `../../../../src/tam/...`) and Python comment tags (`#: <tag>` and `#: </tag>`) to pull code dynamically.
* **Academic Citations:** Any bibliographic reference to justify scientific work must use the MyST formalism `{cite:p}\`bibtex_key\``. Ensure you add the corresponding entry to `references.bib` at the root of the project so it compiles in the PDFs.
* **No Emojis in PDFs:** Ban emojis in titles and the body of text intended for the PDFs (`math/` and `architecture/` folders) to avoid fatal `pdflatex` compilation errors. Emojis are only permitted in the global `README.md`.

### 📂 Annotated Directory Structure (Mapping `.md` ↔ `.py`)

```text
TAM/
│
├── README.md                # Home (Auto-copied by Sphinx)
├── paper.md                 # JOSS Paper (Independent from Sphinx)
│
├── src/tam/                 # 🐍 SOURCE CODE 
│
└───                         # 📚 DOCUMENTATION (Sphinx)
    │
    ├── math/                # 🧠 THE "WHY" (Theory & Equations)
    │   │
    │   ├── core/            # -> Fundamental equations of the solver
    │   │   ├── 01_primal_model.md       # Representer Theorem, Aronszajn. 
    │   │   │                            # Scope scripts: _base.py, additive.py
    │   │   ├── 02_tensorization.md      # N-Dim Broadcasting, temporal/group independence. 
    │   │   │                            # Scope scripts: _data.py, _math.py
    │   │   ├── 03_linear_system.md      # Linear algebra (Cholesky vs Conjugate Gradient). 
    │   │   │                            # Scope scripts: _math.py, _dispatcher.py
    │   │   ├── 04_complexity.md         # Proof of O(N D²) vs O(N³) complexity. 
    │   │   │                            # Scope scripts: _math.py, _dispatcher.py
    │   │   └── 05_gcv_theory.md         # Golub's trace, Tikhonov regularization. 
    │   │                                # Scope scripts: _dispatcher_gcv.py
    │   │
    │   ├── spectrum/        # -> Mathematical definition of Bases (Formulas for Φ and P)
    │   │   ├── LINEAR.md                # Scope scripts: _linear.py
    │   │   ├── SPLINES.md               # Scope scripts: _spline.py
    │   │   ├── FOURIER.md               # Scope scripts: _fourier.py
    │   │   ├── WAVELETS.md              # Scope scripts: _wavelet.py
    │   │   ├── NEURAL.md                # Scope scripts: _neural.py
    │   │   ├── PHYSICS_PIKL.md          # Scope scripts: _physics.py
    │   │   ├── RBF.md                   # Scope scripts: _rbf.py
    │   │   ├── CATEGORICAL.md           # Scope scripts: _categorical.py
    │   │   ├── CHEBYSHEV.md             # Scope scripts: _chebyshev.py
    │   │   ├── TREE.md                  # Scope scripts: _tree.py
    │   │   ├── LINEAR_TREE.md           # Scope scripts: _linear_tree.py
    │   │   ├── PID.md                   # Scope scripts: _pid.py and model/bode.py  
    │   │   └── CROSS_TENSOR.md          # Scope scripts: _tensor.py 
    │   │
    │   └── meta/            # -> Theory of Meta-Learning algorithms
    │       ├── 01_adaptive_online.md         # Sliding windows theory and concept drift. 
    │       │                                 # Scope scripts: adaptive.py
    │       ├── 02_kalman_filter.md           # Extended Kalman Filtering, Woodbury matrix identity. 
    │       │                                 # Scope scripts: kalman.py
    │       ├── 03_hierarchical_joint.md      # Joint optimization under constraints (Parent = Sum). 
    │       │                                 # Scope scripts: hierarchical.py
    │       ├── 04_conformal_safety.md        # Conformal prediction (Split, ACI by Gibbs & Candès). 
    │       │                                 # Scope scripts: safety.py
    │       ├── 05_opera_aggregation.md       # Expert aggregation, regret bounds, Cesa-Bianchi. 
    │       │                                 # Scope scripts: opera.py
    │       ├── 06_deep_gam_backfitting.md    # Orthogonal backfitting per group (Hybridization). 
    │       │                                 # Scope scripts: neural.py (DeepGAM)
    │       ├── 07_statistical_diagnostics.md # T-tests, Bootstrap, Degrees of freedom. 
    │       │                                 # Scope scripts: diagnostics.py
    │       ├── 08_auto_orchestrator.md       # Evolutionary AutoML, EDA, Hub-and-Spoke, Parsimony.
    │       │                                   # Scope: auto_tam.py, drag_tam.py, knowledge_graph.py, population_nodes.py, evolution_reporter.py
    │       │                                   # Pipeline Scope: context.py, data_manager.py, base_discoverer.py, expert_expander.py, ensemble_selector.py
    │       ├── 09_auto_data_topology.md      # Data topology, Krylov stability, Covariate Lock, Panel Data bounds.
    │       │                                   # Scope: data_profiler.py, feature_engineer.py, effect_selector.py, parser.py
    │       └── 10_mlops_evaluation.md          # Theory of empirical metrics, SMAPE, and Temporal Degradation.
    │                                           # Scope scripts: metrics.py, performance_analyzer.py
    │
    └── architecture/        # 💻 THE "HOW" (Code, PyTorch & API)
        │
        ├── core/            # -> Implementation of the core engine
        │   ├── 01_additive_api.md       # Main class and OOP construction.
        │   │                            # Scope scripts: additive.py, _base.py, _factory.py, _base_effects.py, utils.py
        │   ├── 02_data_pipeline.md      # Normalization, Padding, transformations. 
        │   │                            # Scope scripts: _data.py
        │   ├── 03_math_dispatcher.md    # PyTorch routing (Direct Cholesky vs Sparse CG). 
        │   │                            # Scope scripts: _dispatcher.py, _math.py
        │   ├── 04_hardware_memory.md    # Anti-OOM systems, RAM/VRAM estimation. 
        │   │                            # Scope scripts: hardware.py, _memory.py, _dispatcher.py (catch OOM), _tree.py (sparse COO), utils.py
        │   ├── 05_gcv_implementation.md # Discrete coordinate descent and block matrices.
        │   │                            # Scope scripts: _dispatcher_gcv.py
        │   └── 06_the_spectrum_api.md   # Spectrum of core mathematic projection basis. 
        │                                # Scope scripts contained in model/spectrum folder
        │
        └── meta/            # -> Implementation of Wrappers / Meta-Models
            ├── 01_adaptive_code.md      # Vectorized sliding windows. 
            │                            # Scope scripts: adaptive.py, _data.py (_transform_data_adaptive)
            ├── 02_kalman_torchscript.md # @torch.jit.script optimization and block updates. 
            │                            # Scope scripts: kalman.py
            ├── 03_hierarchical_code.md  # Creation of global sparse L^T L loss matrices. 
            │                            # Scope scripts: hierarchical.py
            ├── 04_safety_code.md        # Tensor calculation of quantiles and residual tracker. 
            │                            # Scope scripts: safety.py
            ├── 05_opera_gpu.md          # 3D Tensor Batching (Groups, Time, Experts) on GPU. 
            │                            # Scope scripts: opera.py
            ├── 06_neural_hybrid.md      # Integration of nn.Sequential in the backfitting loop. 
            │                            # Scope scripts: neural.py (DeepGAM)
            ├── 07_diagnostics_utils.md  # Effect plots, statistical tests, and Pandas formatting. 
            │                            # Scope scripts: diagnostics.py, plotting.py, utils.py
            ├── 08_auto_orchestrator_code.md  # 7-Step Pipeline, Knowledge Graph, Dynamic Annealing, OPERA.
            │                                   # Scope: auto_tam.py, drag_tam.py, knowledge_graph.py, population_nodes.py, evolution_reporter.py, autotam_report_generator.py
            │                                   # Pipeline Scope: context.py, data_manager.py, base_discoverer.py, expert_expander.py, ensemble_selector.py
            ├── 09_auto_data_topology_code.md # Stateful Bounds, Collinearity Filter, Regex Parser.
            │                                   # Scope: data_profiler.py, feature_engineer.py, effect_selector.py, parser.py
            └── 10_mlops_tracking_code.md       # BenchmarkTracker OOP, NaN-safe metrics, Matplotlib dashboards.
                                                # Scope scripts: tracker.py, metrics.py, plotting.py (evaluation)

```                       

### Practical Example: Adding the Kalman Filter

If you are tasked with adding documentation for the Extended Kalman Filter meta-learner, your contribution must be split exactly like this:

**1. The Math File (`docs/source/math/meta/02_kalman_filter.md`)**
* **Target Audience:** Researchers.
* **Scope:** Focus entirely on the Markov equations and the Woodbury matrix identity.
* **Requirements:** Use standard LaTeX blocks for formulas (`$$...$$`). Cite academic papers justifying the Extended Kalman Filter approach using `{cite:p}`. 

**2. The Architecture File (`docs/source/architecture/meta/02_kalman_torchscript.md`)**
* **Target Audience:** Engineers.
* **Scope:** Focus entirely on GPU optimization and block updates.
* **Requirements:** Explain how the `@torch.jit.script` decorator is used to compile the inference loop into native C++ to avoid Python GIL bottlenecks. Use `{literalinclude}` to extract the specific decorated function from `src/tam/model/kalman.py` (which should only contain engineering-focused comments). Link back to the Math file for the theory.
