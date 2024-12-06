# nl_main_simple.py
# -*- coding utf-8 -*-

from nl_utils_simple.helpers_nl_simple import visualize_pca_with_topics, generate_dataset
from nl_utils_simple.agent_nl_simple import Agent_NL
from nl_utils_simple.env_nl_simple import Env_NL
from nl_utils_simple.testing_simple import Testing
from nl_utils_simple.predict_nl_simple import Predict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# Get train and test datasets
train_data, test_data = generate_dataset(n_samples=30_000, split=0.6)
train_data.columns
print(f'{train_data.shape =}')


state_attributes =['age', 'interest_anlegen', 'interest_finanzieren', 'interest_vorsorge' ]
state_size = len(state_attributes)

actions = ['interest_anlegen', 'interest_finanzieren', 'interest_vorsorge']
action_size = len(actions)  # Total = 3

scaler = StandardScaler()
train_data[state_attributes] = scaler.fit_transform(train_data[state_attributes])
test_data[state_attributes] = scaler.transform(test_data[state_attributes])


visualize_pca_with_topics(train_data, state_attributes, 'groundtruth', pca=True )

train_dict = train_data.iloc[0].to_dict()
n=Env_NL(data=train_dict, topics=actions)
n.get_state(*state_attributes)
n.get_valid_actions()
n.get_preferred_topic(col='groundtruth')

n.get_reward(chosen_topic='interest_anlegen', col='groundtruth')

# Create the agent
q_agent = Agent_NL(data_df=train_data,
                   state_attributes=state_attributes,
                   actions=actions,
                   learning_rate=0.001, # 
                   target_update_freq=10,
                   gamma=0.95,
                   eps_decay=0.9) 

q_agent._extract_state(n)
q_agent._get_action(n.get_state(*state_attributes),n.get_valid_actions() )

print("learning...")
q_agent.learn(n=train_data.shape[0])
print("done")


predictor = Predict(agent=q_agent, state_attributes=state_attributes)
tester= Testing(agent=q_agent, state_attributes=state_attributes, actions=actions)


results = tester.compare_strategies( test_data=test_data)
print(results)

test_data_with_predictions = predictor.add_predictions_to_dataset(test_data)


y_true = test_data_with_predictions['groundtruth']
y_pred = test_data_with_predictions['predicted_action']

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=test_data_with_predictions['groundtruth'].unique())

accuracy_from_cm = cm.diagonal().sum() / cm.sum()

print(f"Accuracy from Confusion Matrix: {accuracy_from_cm:.2f}")

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data_with_predictions['groundtruth'].unique())
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Ground Truth vs. Predicted Actions")
plt.show()

visualize_pca_with_topics1(test_data, state_attributes, 'groundtruth', pca=True )
visualize_pca_with_topics1(test_data, state_attributes, 'predicted_action', pca=True )

