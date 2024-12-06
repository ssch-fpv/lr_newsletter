# predict_nl.py
# -*- coding utf-8 -*-

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from nl_utils.env_nl import Env_NL

__all__ = ['Predict']

class Predict:
    def __init__(self, agent, state_attributes) -> None:
        self.agent = agent
        self.state_attributes = state_attributes

    def predict(self, data):
        predictions = []
        for _, row in data.iterrows():
            data_dict = row.to_dict()
            nl = Env_NL(data=data_dict, topics=self.agent.actions)
            state = nl.get_state(*self.state_attributes)
            
            predicted_action = self.agent._get_action(state, nl.get_valid_actions())
            predictions.append(predicted_action)
        return predictions
        
    def add_predictions_to_dataset(self, test_data):
        """
        Add predicted actions as a new column to the test dataset.

        Parameters:
        - test_data: Pandas DataFrame containing test customer data.

        Returns:
        - Updated DataFrame with a new column 'predicted_action' containing the predictions.
        """
        test_data['predicted_action'] = self.predict(test_data)
        return test_data