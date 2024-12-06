# predict_nl.py
# -*- coding: utf-8 -*-
"""
This module defines the Predict class, which provides functionality for 
making predictions using a trained agent and appending these predictions to a dataset.
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from nl_utils.env_nl import Env_NL

__all__ = ['Predict']


class Predict:
    """
    A class for generating predictions using a trained agent in the NL environment.

    Attributes:
        agent (Agent_NL): A trained agent capable of making predictions.
        state_attributes (list): A list of attributes defining the environment's state.
    """

    def __init__(self, agent, state_attributes):
        """
        Initializes the prediction class with a trained agent and state attributes.

        Parameters:
            agent (Agent_NL): A trained agent for making predictions.
            state_attributes (list): List of attributes defining the state of the environment.
        """
        self.agent = agent
        self.state_attributes = state_attributes

    def predict(self, data):
        """
        Generates predictions for a given dataset.

        Parameters:
            data (DataFrame): Input data containing customer attributes.

        Returns:
            list: A list of predicted actions for each row in the dataset.
        """
        predictions = []
        for _, row in data.iterrows():
            # Convert the current row to a dictionary
            data_dict = row.to_dict()

            # Initialize the environment with the current data
            nl = Env_NL(data=data_dict, topics=self.agent.actions)

            # Extract the current state
            state = nl.get_state(*self.state_attributes)

            # Get the predicted action using the agent
            predicted_action = self.agent._get_action(state, nl.get_valid_actions())

            # Append the prediction to the list
            predictions.append(predicted_action)

        return predictions

    def add_predictions_to_dataset(self, test_data):
        """
        Adds predicted actions as a new column to the test dataset.

        Parameters:
            test_data (DataFrame): Pandas DataFrame containing test customer data.

        Returns:
            DataFrame: Updated DataFrame with a new column 'predicted_action' containing the predictions.
        """
        # Generate predictions and append them to the DataFrame
        test_data['predicted_action'] = self.predict(test_data)
        return test_data
