# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Implements the NEPT (Neural Explicit Primal Tensorization) effect
"""

import torch
import numpy as np
from typing import List, Optional
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_neural>
class NeuralEffect(BaseEffect):
    r"""
    Implements the NEPT concept: A globally exact, backpropagation-free 
    generalized additive module via Explicit Primal Tensorization (EPT).

    This module projects input data through a randomized, frozen multi-layer 
    feed-forward neural network. By extracting the final hidden layer as a finite-dimensional 
    Primal block and scaling by $1/\sqrt{N_L}$, it translates deep kernel theories 
    (like NNGP and Compositional Kernels) into computationally tractable forms. 
    This allows deep neural representations to be hybridized simultaneously with 
    classical topological effects (e.g., splines) within a globally convex solver.

    Attributes:
        n_neurons (int): Width of the random hidden layers (dimension of the RKHS projection).
        n_hidden_layers (int): Depth of the random network. Defaults to 1.
        activation (str): Activation function ('relu', 'cos', 'tanh').
        scale (float): Scaling factor (1 / sqrt(N_L)) to normalize final output variance.
        input_features (List[str]): List of all input feature names consumed by this effect.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_neurons: int, 
        activation: str, 
        lambda_p: float,
        additional_features: Optional[List[str]],
        seed: int,
        n_hidden_layers: int,
        extrapolate: str
    ):
        r"""
        Initializes the NEPT Effect.

        Args:
            feature_name (str): Name of the primary input feature.
            n_neurons (int): Number of random features (hidden units) per layer.
            activation (str): Activation function ('relu', 'cos', 'tanh').
            lambda_p (float): Regularization strength (optimal isotropic Ridge penalty).
            additional_features (List[str], optional): List of additional feature names.
            seed (int): Random seed for reproducible frozen weights.
            n_hidden_layers (int): Number of hidden layers ($L$) for deep representations.
        """
        
        super().__init__(feature_name, "neural", lambda_p, extrapolate)
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation.lower()
        self.scale = 1.0 / np.sqrt(n_neurons)
        self.seed = seed

        # Aggregate all input feature names for the data loader (Factory)
        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)
            
        # Weights are initialized lazily upon seeing the data to match input dimensions
        self.weights_list = None
        self.bias_list = None
        
        # Legacy pointers for UniversalPhysicsEffect compatibility
        self.weights = None
        self.bias = None

    def get_n_coeffs(self) -> int:
        r"""Returns the dimension of the finite Primal feature space (number of neurons)."""
        return self.n_neurons
#: </init_neural>

#: <init_weights>
    def _init_random_weights(self, input_dim: int):
        r"""
        Initializes the frozen random weights $W^{(l)}$ and biases $b^{(l)}$ using a 
        deterministic generator based on predefined Gaussian prior distributions.

        Args:
            input_dim (int): Dimensionality of the input features.
        """
        rng = torch.Generator(device=TORCH_DEVICE)
        rng.manual_seed(self.seed)

        self.weights_list = []
        self.bias_list = []
        
        curr_dim = input_dim
        for i in range(self.n_hidden_layers):
            # NNGP/Compositional Kernel Theory: The first layer maps from data variance. 
            # Subsequent hidden layers MUST scale by 1/sqrt(N) to prevent variance explosion.
            std_scaling = 1.0 if i == 0 else 1.0 / np.sqrt(self.n_neurons)
            
            w = torch.randn(curr_dim, self.n_neurons, device=TORCH_DEVICE, generator=rng) * std_scaling
            
            # Bias initialization
            if self.activation == 'cos':
                # RFF (Gaussian Kernel) requires Uniform(0, 2*pi) biases
                b = torch.rand(self.n_neurons, device=TORCH_DEVICE, generator=rng) * 2 * np.pi
            else:
                # ReLU/Tanh use standard normal biases to spread kinks across the domain
                b = torch.randn(self.n_neurons, device=TORCH_DEVICE, generator=rng)
                
            self.weights_list.append(w)
            self.bias_list.append(b)
            curr_dim = self.n_neurons
            
        # Maintain backward compatibility for _physics.py penalty matrix constraints
        self.weights = self.weights_list[0]
        self.bias = self.bias_list[0]
#: </init_weights>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Executes the Explicit Primal Tensorization (EPT).
        
        Projects input data through the frozen multi-layer network, iteratively applying 
        the activation function $\sigma(z^{(l-1)} W^{(l)} + b^{(l)})$. Returns the final 
        hidden layer scaled by $1/\sqrt{N_L}$ to normalize the final output variance.

        This method natively resolves tensor broadcasting ambiguities when called by 
        independent sub-systems.

        Args:
            x_col (torch.Tensor): Input tensor of varying dimensionality.

        Returns:
            torch.Tensor: The finite-dimensional Primal block $\phi_{neural}(x)$.
        """
        # 1. Architectural Shape Normalization
        if x_col.dim() == 1:
            # 1D: From OOD Wrapper (Univariate) -> [N_ood]
            x_in = x_col.unsqueeze(-1)
        elif x_col.dim() == 2:
            # 2D: Resolve ambiguity between OOD Wrapper [N_ood, Features] and te() [Batch, Time]
            if x_col.shape[-1] == len(self.input_features):
                # Safeguard for the framework's (1, 1) VRAM memory probe
                if self.weights_list is None and x_col.shape == (1, 1) and len(self.input_features) == 1:
                    x_in = x_col.unsqueeze(-1)
                else:
                    x_in = x_col  # Features are perfectly intact
            else:
                x_in = x_col.unsqueeze(-1)  # Feature dimension was stripped by te()
        else:
            # 3D+: Natively structured from the _factory.py matrix builder
            x_in = x_col

        # 2. Lazy Initialization
        if self.weights_list is None:
            input_dim = x_in.shape[-1]
            self._init_random_weights(input_dim)

        # 3. Deep Linear Projection & Activation Loop (Explicit Primal Tensorization)
        phi = x_in
        for w, b in zip(self.weights_list, self.bias_list):
            w_aligned, b_aligned = self._align_device(x_in, w, b)
            projection = phi @ w_aligned + b_aligned
            
            if self.activation == 'relu':
                phi = torch.relu(projection)
            elif self.activation == 'cos':
                phi = torch.cos(projection)
            elif self.activation == 'tanh':
                phi = torch.tanh(projection)
            else:
                raise ValueError(f"Unknown activation function: {self.activation}")
            
        # Return final scaled features (1 / sqrt(N_L))
        return (phi * self.scale)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Constructs the mathematically optimal penalization for the Primal block.
        
        Because learning is restricted exclusively to the final readout layer $\theta$,
        the optimization problem collapses into a convex quadratic form. The penalty 
        is an isotropic Ridge penalty ($P_{neural} = \lambda_p I$) which, combined with 
        the variance scaling, guarantees operation within a valid RKHS.
        """
        return torch.eye(
            self.n_neurons, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
        ) * self.lambda_p
#: </penalty_matrix>