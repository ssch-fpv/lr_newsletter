# helpers_nl.py
# -*- coding: utf-8 -*-
"""
This module contains helper functions for generating synthetic datasets 
and visualizing PCA with customer preferences.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

__all__ = ['generate_dataset', 'visualize_pca_with_topics']


def generate_dataset(n_samples=10_000, split=0.8):
    """
    Generates a synthetic dataset for customer preferences based on age, gender, 
    liabilities, and interests. The dataset is split into training and testing sets.

    Parameters:
        n_samples (int): Total number of samples to generate. Default is 10,000.
        split (float): Proportion of data to use for training. Default is 0.8.

    Returns:
        tuple: A tuple containing the training DataFrame and testing DataFrame.
    """
    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic features
    ages = np.random.randint(18, 70, size=n_samples)  # Age: Random integers between 18 and 70
    genders = np.random.choice(['M', 'F', 'Conjoint'], size=n_samples)  # Gender categories
    liabilities = np.random.normal(loc=50_000, scale=20_000, size=n_samples).clip(10_000, 200_000)
    
    epsilon = 0.01  # Small constant to avoid zero interests

    # Generate interest levels using beta distributions with age-specific adjustments
    interest_anlegen = np.random.beta(2, 5, size=n_samples) * (0.5 + 0.5 * (ages <= 30)) + epsilon
    interest_finanzieren = np.random.beta(3, 2, size=n_samples) * (0.5 + 0.5 * ((ages > 31) & (ages <= 50))) + epsilon
    interest_vorsorge = np.random.beta(2, 2, size=n_samples) * (0.5 + 0.5 * (ages > 50)) + epsilon

    # Combine interests and determine the preferred topic (highest interest)
    interests = np.vstack((interest_anlegen, interest_finanzieren, interest_vorsorge)).T
    topics = ['interest_anlegen', 'interest_finanzieren', 'interest_vorsorge']
    preferred_topic = [topics[np.argmax(row)] for row in interests]

    # One-hot encode gender categories
    gender_df = pd.get_dummies(genders, prefix='gender')

    # Combine all features into a DataFrame
    data = pd.DataFrame({
        'age': ages,
        'liabilities': liabilities,
        'interest_anlegen': interest_anlegen,
        'interest_finanzieren': interest_finanzieren,
        'interest_vorsorge': interest_vorsorge,
        'groundtruth': preferred_topic,
    }).join(gender_df)

    # Split the dataset into training and testing sets
    train_data = data.sample(frac=split, random_state=42)
    test_data = data.drop(train_data.index)

    return train_data, test_data


def visualize_pca_with_topics(data, features, col, pca=True):
    """
    Visualizes data using PCA or KernelPCA, with color-coding for different topics.

    Parameters:
        data (DataFrame): The input data containing features and a column with topic labels.
        features (list): List of feature column names to use for PCA.
        col (str): Column name indicating the topic labels.
        pca (bool): If True, uses PCA; if False, uses KernelPCA with RBF kernel.

    Returns:
        None: Displays a scatter plot.
    """
    # Standardize the feature columns for PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Apply dimensionality reduction using PCA or KernelPCA
    if pca:
        # Standard PCA with 2 components
        pca_model = PCA(n_components=2)
        data_pca = pca_model.fit_transform(data_scaled)
    else:
        # KernelPCA with RBF kernel
        kpca_model = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        data_pca = kpca_model.fit_transform(data_scaled)

    # Generate a scatter plot of the PCA results
    plt.figure(figsize=(8, 6))
    for topic in sorted(data[col].unique().tolist()):
        mask = data[col] == topic  # Boolean mask for the current topic
        plt.scatter(
            data_pca[mask, 0], 
            data_pca[mask, 1], 
            label=topic, 
            alpha=0.7, 
            s=0.7
        )
    plt.title('PCA: Customer Preferences')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
