# nl_main.py
# -*- coding: utf-8 -*-
"""
Main script for training a reinforcement learning agent, testing its performance, 
and visualizing results using PCA and a confusion matrix.
"""

from nl_utils.helpers_nl import visualize_pca_with_topics, generate_dataset
from nl_utils.agent_nl import Agent_NL
from nl_utils.testing import Testing
from nl_utils.predict_nl import Predict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Generate Training and Test Datasets
train_data, test_data = generate_dataset(n_samples=30_000, split=0.5)
print(f'Train Data Columns: {train_data.columns}')
print(f'Train Data Shape: {train_data.shape}')

# Define state attributes and actions
state_attributes = ['age', 'interest_anlegen', 'interest_finanzieren', 'interest_vorsorge']
state_size = len(state_attributes)
actions = ['interest_anlegen', 'interest_finanzieren', 'interest_vorsorge']
action_size = len(actions)

# Standardize the state attributes
scaler = StandardScaler()
train_data[state_attributes] = scaler.fit_transform(train_data[state_attributes])
test_data[state_attributes] = scaler.transform(test_data[state_attributes])

# Step 2: Initialize and Train the Agent
q_agent = Agent_NL(
    data_df=train_data,
    state_attributes=state_attributes,
    actions=actions,
    learning_rate=0.001,  # Learning rate for the optimizer
    target_update_freq=10,  # Target network update frequency
    gamma=0.95,  # Discount factor
    eps_decay=0.9  # Epsilon decay rate for exploration
)

print("Training the agent...")
q_agent.learn(n=train_data.shape[0])  # Train the agent with the entire training dataset
print("Training completed.")

# Step 3: Test the Agent and Compare Strategies
predictor = Predict(agent=q_agent, state_attributes=state_attributes)
tester = Testing(agent=q_agent, state_attributes=state_attributes, actions=actions)
results = tester.compare_strategies(test_data=test_data)

# Display test results
print("Test Results:", results)

# Step 4: Generate Predictions and Evaluate Performance
test_data_with_predictions = predictor.add_predictions_to_dataset(test_data)
y_true = test_data_with_predictions['groundtruth']
y_pred = test_data_with_predictions['predicted_action']

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=test_data_with_predictions['groundtruth'].unique())
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data_with_predictions['groundtruth'].unique())
cm_display.plot()
plt.title("Confusion Matrix: Ground Truth vs. Predicted Actions")
plt.show()

# Calculate and display accuracy
accuracy_from_cm = cm.diagonal().sum() / cm.sum()
print(f"Accuracy from Confusion Matrix: {accuracy_from_cm:.2f}")

# Step 5: Visualize Results with PCA
# Visualize PCA projections for ground truth and predictions
print("Visualizing PCA results...")
visualize_pca_with_topics(test_data, state_attributes, 'groundtruth', pca=True)
visualize_pca_with_topics(test_data, state_attributes, 'predicted_action', pca=True)
visualize_pca_with_topics(train_data, state_attributes, 'groundtruth', pca=True)
