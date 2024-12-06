# q_nl.py
# -*- coding: utf-8 -*-
"""
This module defines the DQNetwork class, which implements a deep Q-network 
for reinforcement learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DQNetwork']


class DQNetwork(nn.Module):
    """
    A deep Q-network (DQN) for approximating Q-values in reinforcement learning.

    Attributes:
        state_size (int): The size of the input state vector.
        action_size (int): The number of possible actions.
    """

    def __init__(self, state_size, action_size):
        """
        Initializes the network layers.

        Parameters:
            state_size (int): The size of the input state vector.
            action_size (int): The number of possible actions.
        """
        super(DQNetwork, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, action_size)  # Output layer (one output per action)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters:
            x (Tensor): Input tensor representing the state.

        Returns:
            Tensor: Output tensor representing Q-values for each action.
        """
        # Apply ReLU activation to the first and second hidden layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output layer (no activation function)
        x = self.fc3(x)
        return x
