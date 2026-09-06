=======================================================
Time series Additive Model (TAM) Official Documentation
=======================================================

.. only:: html

   .. include:: README.md
      :parser: myst_parser.sphinx_

.. only:: latex

   Welcome to the official documentation for the **Time series Additive Model (TAM)** framework.

   This unified documentation portal bridges the gap between theoretical research and software engineering. It brings together the rigorous mathematical foundations of our primal solver and detailed implementation guides, complete with dynamically extracted PyTorch source code.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 Getting Started

   README
   THEORY

.. raw:: latex

   \part{Theory: The Core Engine}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🧠 Theory: The Core Engine

   math/core/01_primal_model
   math/core/02_tensorization
   math/core/03_linear_system
   math/core/04_complexity
   math/core/05_gcv_theory
   math/core/07_the_statistics_api

.. raw:: latex

   \part{Theory: Mathematical Representations}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🌈 Theory: The Spectrum Library

   math/spectrum/LINEAR
   math/spectrum/SPLINES
   math/spectrum/FOURIER
   math/spectrum/WAVELETS
   math/spectrum/CHEBYSHEV
   math/spectrum/TREE
   math/spectrum/NEURAL
   math/spectrum/RBF
   math/spectrum/CATEGORICAL
   math/spectrum/CROSS_TENSOR
   math/spectrum/PHYSICS_PIKL
   math/spectrum/PID
   math/spectrum/LINEAR_TREE

.. raw:: latex

   \part{Theory: The Statistics Engine}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📊 Theory: The Statistics Engine

   math/statistics/01_reweighted_estimation
   math/statistics/02_distributional
   math/statistics/03_mixture_copula
   math/statistics/04_risk_conformal_evt

.. raw:: latex

   \part{Theory: Meta-Models}
 
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: ⚙️ Theory: Meta-Learners & Inference

   math/meta/01_adaptive_online
   math/meta/02_kalman_filter
   math/meta/03_hierarchical_joint
   math/meta/05_opera_aggregation
   math/meta/06_deep_gam_backfitting   
   math/meta/07_statistical_diagnostics
   math/meta/08_auto_orchestrator
   math/meta/09_auto_data_topology
   math/meta/10_mlops_evaluation

.. raw:: latex

   \part{Architecture: Core Implementation}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 💻 Architecture: Core Implementation

   architecture/core/01_additive_api
   architecture/core/02_data_pipeline
   architecture/core/03_math_dispatcher
   architecture/core/04_hardware_memory
   architecture/core/05_gcv_implementation
   architecture/core/06_the_spectrum_api
   architecture/core/07_the_statistics_api

.. raw:: latex

   \part{Architecture: The Statistics Engine}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🔬 Architecture: The Statistics Engine

   architecture/statistics/01_reweighted_estimation_code
   architecture/statistics/02_distributional_code
   architecture/statistics/03_mixture_copula_code
   architecture/statistics/04_risk_conformal_evt_code

.. raw:: latex

   \part{Architecture: Meta Implementation}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🛠️ Architecture: Advanced Orchestration

   architecture/meta/01_adaptive_code
   architecture/meta/02_kalman_torchscript
   architecture/meta/03_hierarchical_code
   architecture/meta/05_opera_gpu
   architecture/meta/06_neural_hybrid
   architecture/meta/07_diagnostics_utils
   architecture/meta/08_auto_orchestrator_code
   architecture/meta/09_auto_data_topology_code
   architecture/meta/10_mlops_tracking_code

.. raw:: latex

   \part{Others}

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🧪 Gallery of Examples & Use Cases

   EXAMPLES

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 References & Changelog

   api/modules
   AUTHORS
   ACKNOWLEDGEMENTS
   CHANGELOG
   CONTRIBUTING
   README_DOC
   bibliography