r"""
Public API for the Spectrum Effect Library.

This module exposes the heterogeneous collection of functional bases ("Effects")
used to construct Additive TAM models. Each effect class implements
a mapping from input features to a Reproducing Kernel Hilbert Space (RKHS),
accompanied by a specific regularization operator.

Available Effects:

- Classical: Offset, Linear, Fourier (Spectral), Spline (Local).
- Discrete: Categorical (One-Hot with Nominal/Ordinal topology).
- Global Approximation: Chebyshev (Polynomials), Wavelet (Time-Frequency).
- Kernel & Neural: RBF (Gaussian/Matérn), Neural.
- Physics-Informed: UniversalPhysicsEffect (PDE-constrained).
- Interactions: TensorProductEffect (Multivariate Kronecker products).

Factory Utilities:

- create_effects_from_parsed_terms: Parsing engine transforming formula strings into objects.
- build_phi_from_effects: Assembler for the global Design Matrix (Phi).
- build_penalty_from_effects: Assembler for the global Penalty Matrix (P).
"""

from ._base_effects import BaseEffect
from ._linear import OffsetEffect, LinearEffect
from ._fourier import FourierEffect
from ._spline import SplineEffect
from ._categorical import CategoricalEffect
from ._chebyshev import ChebyshevEffect
from ._wavelet import WaveletEffect
from ._neural import NeuralEffect
from ._rbf import RBFEffect
from ._physics import UniversalPhysicsEffect
from ._tensor import TensorProductEffect
from ._tree import TreeEffect
from ._linear_tree import LinearTreeEffect
from ._pid import PIDEffect

# Utility functions exposed for the StaticTAM model
from ._factory import (
    create_effects_from_parsed_terms,
    build_phi_from_effects,
    build_penalty_from_effects
)

__all__ = [
    # Abstract Base
    "BaseEffect",
    
    # Concrete Effects
    "OffsetEffect", 
    "LinearEffect",
    "FourierEffect",
    "SplineEffect",
    "CategoricalEffect",
    "ChebyshevEffect",
    "WaveletEffect",
    "NeuralEffect",
    "RBFEffect",
    "UniversalPhysicsEffect",
    "TensorProductEffect",
    "TreeEffect",
    "LinearTreeEffect",
    "PIDEffect",

    # Factory / Assembly
    "create_effects_from_parsed_terms",
    "build_phi_from_effects",
    "build_penalty_from_effects",
]