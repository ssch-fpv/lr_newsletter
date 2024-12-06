# nl.py
# -*- coding: utf-8 -*-
"""
This module defines the Env_NL class, which simulates an environment 
for customer behavior and preferences in a decision-making process.
"""

import numpy as np
import pandas as pd

__all__ = ['Env_NL']


class Env_NL:
    """
    A simulation environment for customer behavior and topic preference.

    Attributes:
        data (dict): A dictionary representing customer attributes and preferences.
        topics (list): A list of valid topics (actions) for the environment.
    """

    def __init__(self, data, topics):
        """
        Initializes the environment with customer data and topics.

        Parameters:
            data (dict): A dictionary representing customer data and preferences.
            topics (list): A list of valid topics (actions).
        """
        self.data = data
        self.topics = topics

    def get_state(self, *args):
        """
        Retrieves the state of the environment based on specified attributes.

        Parameters:
            *args: Column names to include in the state.

        Returns:
            tuple: A tuple containing the values of the specified attributes.
        """
        return tuple(self.data[col] for col in args)

    def get_valid_actions(self):
        """
        Retrieves the list of valid actions (topics) in the environment.

        Returns:
            list: A list of valid topics.
        """
        return self.topics

    def get_preferred_topic(self, col):
        """
        Retrieves the preferred topic for the customer.

        Parameters:
            col (str): The column name containing the preferred topic.

        Returns:
            str: The preferred topic, or an empty string if the column does not exist.
        """
        return self.data.get(col, '')

    def get_reward(self, chosen_topic, col):
        """
        Calculates the reward based on the chosen topic and the preferred topic.

        Parameters:
            chosen_topic (str): The topic chosen by the agent.
            col (str): The column name containing the preferred topic.

        Returns:
            int: A reward of 3 if the chosen topic matches the preferred topic, 
                 otherwise -1.
        """
        preferred_topic = self.get_preferred_topic(col)
        return 3 if chosen_topic == preferred_topic else -1
