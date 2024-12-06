# agent_nl.py
# -*- coding utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np

from nl_utils_simple.env_nl_simple import Env_NL
from nl_utils_simple.q_nl_simple import DQNetwork, DuelingDQNetwork

__all__ = ['Agent_NL']



class Agent_NL:
    def __init__(self, data_df, state_attributes, actions, learning_rate=0.0001, target_update_freq=10, gamma=0.99, eps_decay=9.99):
        self.data_df = data_df
        self.state_size = len(state_attributes)
        self.action_size = len(actions) 
        self.state_attributes = state_attributes
        self.actions = actions
        self.eps = 1.0  # Initial exploration rate
        self.eps_min = 0.1  # Minimum exploration rate
        self.eps_decay = eps_decay # Decay rate for epsilon
        self.gamma = gamma  # Discount factor
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Initialize running stats for reward normalization
        self.reward_mean = 0
        self.reward_var = 1
        self.num_rewards = 0

        # Track action counts for diversity
        self.action_counts = defaultdict(int)

        # Initialize policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        #self.policy_net = DuelingDQNetwork(self.state_size, self.action_size)
        #self.target_net = DuelingDQNetwork(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def normalize_reward(self, reward):
        """Normalize reward using running mean and variance."""
        self.num_rewards += 1
        old_mean = self.reward_mean
        self.reward_mean += (reward - self.reward_mean) / self.num_rewards
        self.reward_var += (reward - old_mean) * (reward - self.reward_mean)

        std = max(1e-6, (self.reward_var / self.num_rewards) ** 0.5)  # Avoid divide-by-zero
        return (reward - self.reward_mean) / std

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        
        #if self.episode > 0 and self.episode % 5000 == 0:
        #    print("Resetting epsilon to 1.0 for exploration boost.")
        #    self.eps = 1.0

    #def normalize_reward(self, reward):
    #    """Normalize reward using exponential moving averages."""
    #    alpha = 0.1  # Smooth update factor
    #    self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
    #    self.reward_var = (1 - alpha) * self.reward_var + alpha * (reward - self.reward_mean) ** 2
    #    std = max(1e-6, self.reward_var ** 0.5)  # Avoid divide-by-zero
    #    return (reward - self.reward_mean) / std
        

    def _get_action(self, state, valid_actions):
        """Select action using epsilon-greedy strategy."""
        if random.random() < self.eps:  # Exploration
            return random.choice(valid_actions)

        # Exploitation: Use the policy network
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze().numpy()

        # Softmax exploration for proportional sampling
        exp_q = np.exp(q_values - np.max(q_values))  # Prevent overflow
        probs = exp_q / exp_q.sum()
        return np.random.choice(valid_actions, p=probs)

    def update_network(self, state, action, reward, next_state, done):
        """Update Q-network with one step of Q-learning."""
        # Normalize the reward
        reward = self.normalize_reward(reward)

        # Convert data to tensors
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)

        # Predict Q-values for the current state
        q_values = self.policy_net(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # Target Q-value using the target network
        #with torch.no_grad():
        #    next_q_values = self.target_net(next_state_tensor)
        #    next_q_value = next_q_values.max(1)[0]
        #    target_q_value = reward_tensor + (self.gamma * next_q_value * (1 - done))
#
          # Double Q-learning: Use policy network to select action, target network to evaluate
        with torch.no_grad():
            # Select the best action using the policy network
            best_action = self.policy_net(next_state_tensor).argmax(1).unsqueeze(1)
            # Evaluate the Q-value of the best action using the target network
            next_q_value = self.target_net(next_state_tensor).gather(1, best_action).squeeze(1)
            # Compute the target Q-value
            target_q_value = reward_tensor + (self.gamma * next_q_value * (1 - done))


        # Compute loss and update the network
        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

            # Increment steps

        #print(f"Steps: {self.steps}")

            

    def learn(self, n=30_000, max_steps=50):
        """Main training loop."""
        for self.episode, (_, row) in enumerate(self.data_df.head(n).iterrows(), start=1):
            #print(f'learning : {episode}, {row.to_dict()=}')
            self._learn_one_game(row.to_dict(), self.episode, max_steps)
            
            if self.episode % 500 == 0:
                self.decay_epsilon()
                print(f"Episode {self.episode}: Epsilon = {self.eps:.3f}")

    def learn_batch(self, n=30_000, max_steps=50, batch_size=2000):
        """
        Main training loop with shuffled data and batch processing.
        
        Parameters:
        - n: Total number of episodes (rows to use for training).
        - max_steps: Maximum number of steps per episode.
        - batch_size: Number of rows (customers) processed in each batch.
        """
        # Shuffle the data
        shuffled_data = self.data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Iterate through episodes in batches
        num_batches = n // batch_size  # Total number of batches
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, n)
            batch_data = shuffled_data.iloc[batch_start:batch_end]
            
            # Process each row in the batch
            for _, row in batch_data.iterrows():
                self._learn_one_game(row.to_dict(), episode=batch_num + 1, max_steps=max_steps)
            
            # Decay epsilon after processing each batch
            self.decay_epsilon()
            
            # Log progress every 10 batches
            if batch_num % 10 == 0:
                print(f"Batch {batch_num + 1}/{num_batches}: Epsilon = {self.eps:.3f}")



    def _learn_one_game(self, data_dict, episode, max_steps=50):
        """Train the agent for one game."""
        nl = Env_NL(data=data_dict, topics=self.actions)
        state = self._extract_state(nl)
        cumulative_reward = 0
        steps = 0
        done = False

        while steps < max_steps and not done:
            valid_actions = nl.get_valid_actions()
            action = self._get_action(state, valid_actions)

            # Simulate reward and s
            reward = nl.get_reward(chosen_topic=action, col='groundtruth')
            #reward = nl.get_reward(chosen_topic=action, col='groundtruth', time_step=steps)
            cumulative_reward += reward

            # Penalize repeated actions
            #if self.action_counts[action] > len(valid_actions) // 2:
            #    reward -= 2

            self.action_counts[action] += 1

            # Update the Q-network
            action_index = valid_actions.index(action)
            self.update_network(state, action_index, reward, state, done=False)

            # Terminate early if the correct choice was made
            if reward == 2:
                done = True
                #print(f"Episode {episode}: Correct choice made at step {steps + 1}.")

            # Increment steps
            steps += 1

        # Reset action counts after each episode
        self.action_counts = defaultdict(int)

        # Log cumulative reward
        if episode % 1000 == 0:
            avg_reward = cumulative_reward / max(steps, 1)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Cumulative Reward = {cumulative_reward}")

    def _extract_state(self, nl):
        """Extract the state dynamically based on state attributes."""
        return nl.get_state(*self.state_attributes)