# testing.py
# -*- coding utf-8 -*-

import random
from nl_utils.env_nl import Env_NL

__all__ = ['Testing']

class Testing:
    def __init__(self, state_attributes, actions, agent):
        """
        Initialize the Testing class.

        Parameters:
        - state_attributes: List of attributes to extract from the customer data for state representation.
        """
        self.state_attributes = state_attributes
        self.actions = actions
        self.agent = agent

    def test_model(self, test_data):
        """
        Test the agent's performance on the test dataset.

        Parameters:
        - agent: The trained agent to evaluate.
        - test_data: Pandas DataFrame containing test customer data.

        Returns:
        - avg_reward: Average reward achieved by the agent on the test dataset.
        - success_rate: Percentage of correct actions (reward = 10) by the agent.
        """
        total_rewards = 0
        success_count = 0

        for _, row in test_data.iterrows():
            # Convert row to dictionary
            data_dict = row.to_dict()

            # Initialize the environment
            nl = Env_NL(data=data_dict, topics=self.actions)

            # Extract state and interests
            state = nl.get_state(*self.state_attributes)

            # Predict the best action
            action_campaing_target = self.agent._get_action(state, nl.get_valid_actions())

            # Simulate open rate and calculate reward
            reward = nl.get_reward(chosen_topic=action_campaing_target, col='groundtruth')

            # Update totals
            total_rewards += reward
            if reward == 10:
                success_count += 1

        # Calculate metrics
        avg_reward = total_rewards / len(test_data)
        success_rate = success_count / len(test_data)

        return avg_reward, success_rate

    def test_random_strategy(self, test_data):
        """
        Test a random action strategy on the test dataset.

        Parameters:
        - test_data: Pandas DataFrame containing test customer data.

        Returns:
        - avg_reward: Average reward achieved by the random strategy on the test dataset.
        - success_rate: Percentage of correct actions (reward = 10) by the random strategy.
        """
        total_rewards = 0
        success_count = 0

        for _, row in test_data.iterrows():
            # Convert row to dictionary
            data_dict = row.to_dict()

            # Initialize the environment
            nl = Env_NL(data=data_dict, topics=self.actions)

            # Extract past engagement and interests

            # Randomly select an action
            action_campaing_target = random.choice(nl.get_valid_actions())

            # Simulate open rate and calculate reward
            
            reward = reward = nl.get_reward(chosen_topic=action_campaing_target, col='groundtruth')

            # Update totals
            total_rewards += reward
            if reward == 10:
                success_count += 1

        # Calculate metrics
        avg_reward = total_rewards / len(test_data)
        success_rate = success_count / len(test_data)

        return avg_reward, success_rate

    def compare_strategies(self, test_data):
        """
        Compare the agent's performance against a random strategy on the test dataset.

        Parameters:
        - agent: The trained agent to evaluate.
        - test_data: Pandas DataFrame containing test customer data.

        Returns:
        - A dictionary containing average rewards and success rates for both strategies.
        """
        avg_reward_agent, success_rate_agent = self.test_model( test_data)
        avg_reward_random, success_rate_random = self.test_random_strategy(test_data)

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
