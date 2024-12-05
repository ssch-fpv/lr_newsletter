# q_nl.py
# -*- coding utf-8 -*-

from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


__all__ = ['DQNetwork','DuelingDQNetwork']



class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)  # One output per action


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature_layer = nn.Linear(state_size, 128)
        self.value_stream = nn.Linear(128, 1)  # Wertschätzung
        self.advantage_stream = nn.Linear(128, action_size)  # Vorteil

    def forward(self, state):
        features = F.relu(self.feature_layer(state))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean())