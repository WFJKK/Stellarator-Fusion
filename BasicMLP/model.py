"""
stellarator_net.py

Defines a simple feedforward neural network (MLP) model for predicting
stellarator metrics such as maximum elongation from numerical features. Output is a positive real number, hence the final softplus.
"""

import torch.nn as nn
import torch


class StellaratorNet(nn.Module):
    """
    Fully connected neural network .

    Architecture:
        input_dim → 128 → 64 → 1 (Softplus activation on output)

    Args:
        input_dim (int): Number of input features.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)


