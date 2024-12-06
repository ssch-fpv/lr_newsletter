# testing.py
# -*- coding: utf-8 -*-
"""
This module defines the Testing class, which evaluates the performance of a trained agent 
against a random strategy on a given test dataset.
"""

import random
from nl_utils.env_nl import Env_NL

__all__ = ['Testing']


class Testing:
    """
    A class for evaluating the performance of a trained agent and comparing it 
    to a random action strategy on test data.

    Attributes:
        state_attributes (list): Attributes used for state representation.
        actions (list): List of possible actions (topics) in the environment.
        agent (Agent_NL): The trained agent to be tested.
    """

    def __init__(self, state_attributes, actions, agent):
        """
        Initializes the Testing class.

        Parameters:
            state_attributes (list): Attributes to extract for state representation.
            actions (list): List of possible actions.
            agent (Agent_NL): The trained agent to be evaluated.
        """
        self.state_attributes = state_attributes
        self.actions = actions
        self.agent = agent

    def test_model(self, test_data):
        """
        Tests the agent's performance on the test dataset.

        Parameters:
            test_data (DataFrame): Pandas DataFrame containing test customer data.

        Returns:
            tuple: 
                - avg_reward (float): Average reward achieved by the agent.
                - success_rate (float): Percentage of correct actions (reward = 10).
        """
        total_rewards = 0
        success_count = 0

        for _, row in test_data.iterrows():
            # Convert row to a dictionary
            data_dict = row.to_dict()

            # Initialize the environment
            nl = Env_NL(data=data_dict, topics=self.actions)

            # Extract the current state
            state = nl.get_state(*self.state_attributes)

            # Predict the action using the agent
            action_campaing_target = self.agent._get_action(state, nl.get_valid_actions())

            # Calculate the reward for the chosen action
            reward = nl.get_reward(chosen_topic=action_campaing_target, col='groundtruth')

            # Update cumulative rewards and success count
            total_rewards += reward
            if reward == 10:  # Reward threshold for a successful action
                success_count += 1

        # Calculate metrics
        avg_reward = total_rewards / len(test_data)
        success_rate = success_count / len(test_data)

        return avg_reward, success_rate

    def test_random_strategy(self, test_data):
        """
        Tests a random action strategy on the test dataset.

        Parameters:
            test_data (DataFrame): Pandas DataFrame containing test customer data.

        Returns:
            tuple:
                - avg_reward (float): Average reward achieved by the random strategy.
                - success_rate (float): Percentage of correct actions (reward = 10).
        """
        total_rewards = 0
        success_count = 0

        for _, row in test_data.iterrows():
            # Convert row to a dictionary
            data_dict = row.to_dict()

            # Initialize the environment
            nl = Env_NL(data=data_dict, topics=self.actions)

            # Randomly select an action
            action_campaing_target = random.choice(nl.get_valid_actions())

            # Calculate the reward for the chosen action
            reward = nl.get_reward(chosen_topic=action_campaing_target, col='groundtruth')

            # Update cumulative rewards and success count
            total_rewards += reward
            if reward == 10:  # Reward threshold for a successful action
                success_count += 1

        # Calculate metrics
        avg_reward = total_rewards / len(test_data)
        success_rate = success_count / len(test_data)

        return avg_reward, success_rate

    def compare_strategies(self, test_data):
        """
        Compares the agent's performance against a random action strategy.

        Parameters:
            test_data (DataFrame): Pandas DataFrame containing test customer data.

        Returns:
            dict: A dictionary containing average rewards and success rates 
                  for both the agent and the random strategy.
        """
        # Evaluate the agent's performance
        avg_reward_agent, success_rate_agent = self.test_model(test_data)

        # Evaluate the random strategy's performance
        avg_reward_random, success_rate_random = self.test_random_strategy(test_data)

        # Compile the results into a dictionary
        results = {
            'agent': {
                'avg_reward': avg_reward_agent,
                'success_rate': success_rate_agent
            },
            'random': {
                'avg_reward': avg_reward_random,
                'success_rate': success_rate_random
            }
        }

        return results
