# nl.py
# -*- coding utf-8 -*-


import numpy as np
import pandas as pd

__all__ = ['Env_NL']

class Env_NL():
    """

    """
    def __init__(self, data, topics) -> None:
        self.data = data
        self.topics = topics

    def get_state(self, *args):
        """state are the customer attributs"""
        return tuple(self.data[col] for col in args)

    def get_valid_actions(self):
        return self.topics
    
    def get_preferred_topic(self, col):
        return self.data.get(col, '')
    
    def get_reward(self, chosen_topic, col):
        preferred_topic = self.get_preferred_topic(col)
        return 3 if chosen_topic == preferred_topic else -1 
    
    #def get_reward(self, chosen_topic, col, time_step):
    #    preferred_topic = self.get_preferred_topic(col)
    #    if chosen_topic == preferred_topic:
    #        return 1 + (0.1 / (time_step + 1))  # Higher reward for faster decisions
    #    return -1