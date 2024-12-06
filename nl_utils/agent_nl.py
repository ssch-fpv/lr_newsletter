# agent_nl.py
# -*- coding: utf-8 -*-
"""
This module implements the Agent_NL class, which uses Deep Q-Learning to 
train an agent to interact with the Env_NL environment and optimize actions.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np

from nl_utils.env_nl import Env_NL
from nl_utils.q_nl import DQNetwork

__all__ = ['Agent_NL']


class Agent_NL:
    """
    A Deep Q-Learning agent for decision-making in the NL environment.

    Attributes:
        data_df (DataFrame): Input dataset for training.
        state_attributes (list): Attributes used to define the agent's state.
        actions (list): Possible actions the agent can take.
        learning_rate (float): Learning rate for the optimizer.
        target_update_freq (int): Frequency (in steps) to update the target network.
        gamma (float): Discount factor for Q-learning.
        eps_decay (float): Decay rate for epsilon (exploration rate).
    """

    def __init__(self, data_df, state_attributes, actions, learning_rate=0.0001, 
                 target_update_freq=10, gamma=0.99, eps_decay=0.99):
        # Initialization of key parameters
        self.data_df = data_df
        self.state_size = len(state_attributes)
        self.action_size = len(actions)
        self.state_attributes = state_attributes
        self.actions = actions

        # Exploration-exploitation parameters
        self.eps = 1.0  # Initial exploration rate
        self.eps_min = 0.1  # Minimum exploration rate
        self.eps_decay = eps_decay  # Epsilon decay rate
        self.gamma = gamma  # Discount factor for future rewards

        # Target network update and step tracking
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Reward normalization parameters
        self.reward_mean = 0
        self.reward_var = 1
        self.num_rewards = 0

        # Track action counts for exploration diversity
        self.action_counts = defaultdict(int)

        # Initialize policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not updated during backpropagation

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _extract_state(self, nl):
        """Extract the current state from the environment."""
        return nl.get_state(*self.state_attributes)

    def normalize_reward(self, reward):
        """
        Normalize the reward using running mean and variance.

        Parameters:
            reward (float): The reward to normalize.

        Returns:
            float: The normalized reward.
        """
        self.num_rewards += 1
        old_mean = self.reward_mean
        self.reward_mean += (reward - self.reward_mean) / self.num_rewards
        self.reward_var += (reward - old_mean) * (reward - self.reward_mean)

        std = max(1e-6, (self.reward_var / self.num_rewards) ** 0.5)  # Avoid divide-by-zero
        return (reward - self.reward_mean) / std

    def decay_epsilon(self):
        """Decay the exploration rate epsilon."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def _get_action(self, state, valid_actions):
        """
        Select an action using the epsilon-greedy strategy.

        Parameters:
            state (list): Current state of the environment.
            valid_actions (list): List of valid actions.

        Returns:
            int: The selected action.
        """
        if random.random() < self.eps:
            # Exploration: Choose a random action
            return random.choice(valid_actions)

        # Exploitation: Use the policy network to predict the best action
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze().numpy()

        # Proportional sampling using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Stability trick
        probs = exp_q / exp_q.sum()
        return np.random.choice(valid_actions, p=probs)

    def update_network(self, state, action, reward, next_state, done):
        """
        Update the Q-network using one step of Q-learning.

        Parameters:
            state (list): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (list): Next state.
            done (bool): Whether the episode is done.
        """
        reward = self.normalize_reward(reward)

        # Convert inputs to tensors
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)

        # Compute the predicted Q-value for the current state-action pair
        q_values = self.policy_net(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # Compute the target Q-value using the target network
        with torch.no_grad():
            best_action = self.policy_net(next_state_tensor).argmax(1).unsqueeze(1)
            next_q_value = self.target_net(next_state_tensor).gather(1, best_action).squeeze(1)
            target_q_value = reward_tensor + (self.gamma * next_q_value * (1 - done))

        # Compute loss and update the network
        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self, n=30_000, max_steps=50):
        """
        Main training loop for the agent.

        Parameters:
            n (int): Number of episodes to train.
            max_steps (int): Maximum steps per episode.
        """
        for self.episode, (_, row) in enumerate(self.data_df.head(n).iterrows(), start=1):
            self._learn_one_game(row.to_dict(), self.episode, max_steps)
            
            if self.episode % 500 == 0:
                self.decay_epsilon()
                print(f"Episode {self.episode}: Epsilon = {self.eps:.3f}")

    def _learn_one_game(self, data_dict, episode, max_steps=50):
        """
        Train the agent for one game episode.

        Parameters:
            data_dict (dict): Data for the episode.
            episode (int): Current episode number.
            max_steps (int): Maximum steps allowed in the episode.
        """
        nl = Env_NL(data=data_dict, topics=self.actions)
        state = self._extract_state(nl)
        cumulative_reward = 0
        steps = 0
        done = False

        while steps < max_steps and not done:
            valid_actions = nl.get_valid_actions()
            action = self._get_action(state, valid_actions)

            # Simulate the environment's response
            reward = nl.get_reward(chosen_topic=action, col='groundtruth')
            cumulative_reward += reward

            self.action_counts[action] += 1

            # Update the Q-network
            action_index = valid_actions.index(action)
            self.update_network(state, action_index, reward, state, done=False)

            if reward == 2:  # Early termination condition
                done = True

            steps += 1

        self.action_counts = defaultdict(int)  # Reset action counts

        if episode % 1000 == 0:
            avg_reward = cumulative_reward / max(steps, 1)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Cumulative Reward = {cumulative_reward}")
