# data.py
# -*- coding utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


__all__ = ['generate_dataset', 'visualize_pca_with_topics']

def generate_dataset(n_samples=10_000, split=0.8):
    np.random.seed(42)
    # Generate features (same as before)
    ages = np.random.randint(18, 70, size=n_samples)
    genders = np.random.choice(['M', 'F', 'Conjoint'], size=n_samples)
    liabilities = np.random.normal(loc=50_000, scale=20_000, size=n_samples).clip(10_000, 200_000)
    
    epsilon = 0.01
    
    interest_anlegen = np.random.beta(2, 5, size=n_samples) * (0.5 + 0.5 *(ages <= 30)) + epsilon
    interest_finanzieren = np.random.beta(3, 2, size=n_samples) * (0.5 + 0.5 * ((ages > 31) & (ages <= 50))) + epsilon
    interest_vorsorge = np.random.beta(2, 2, size=n_samples) * (0.5 + 0.5 * (ages > 50)) + epsilon


    # Determine the preferred topic as the one with the highest interest
    interests = np.vstack((interest_anlegen, interest_finanzieren, interest_vorsorge)).T
    topics = ['interest_anlegen', 'interest_finanzieren', 'interest_vorsorge']
    preferred_topic = [topics[np.argmax(row)] for row in interests]

     # Gender one-hot encoding
    gender_df = pd.get_dummies(genders, prefix='gender')

    # Combine everything into a DataFrame
    data = pd.DataFrame({
        'age': ages,
        'liabilities': liabilities,
        'interest_anlegen': interest_anlegen,
        'interest_finanzieren': interest_finanzieren,
        'interest_vorsorge': interest_vorsorge,
        'groundtruth': preferred_topic,
    }).join(gender_df)

    # Split into train and test
    train_data = data.sample(frac=split, random_state=42)
    test_data = data.drop(train_data.index)

    return train_data, test_data

#train_data, test_data = generate_dataset(n_samples=30_000, split=0.6)


def visualize_pca_with_topics(data, features, col, pca=True):
    # Standardize features for PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    if pca:
        # Apply PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
    else:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        data_pca = kpca.fit_transform(data_scaled)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    for topic in sorted(data[col].unique().tolist()):
        mask = data[col] == topic
        plt.scatter(data_pca[mask, 0], data_pca[mask, 1], label=topic, alpha=0.7)
    plt.title('PCA: Customer Preferences')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
