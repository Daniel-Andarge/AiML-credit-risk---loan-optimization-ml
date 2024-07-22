import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


    # Combine all normalized features
    features = customer_data[['R_norm', 'F_norm', 'M_norm', 'S_norm', 'Hour_norm', 'Day_norm', 'Month_norm', 'Year_norm']]

    # Scale features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Fit K-Means clustering
    kmeans = KMeans(n_clusters=2)
    customer_data['Cluster'] = kmeans.fit_predict(features_scaled)

    # Assign labels based on clusters
    cluster_centers = kmeans.cluster_centers_
    high_cluster = cluster_centers[:, -1].argmax()  # Using the last column for deciding high cluster
    customer_data['Label'] = customer_data['Cluster'].apply(lambda x: 'good' if x == high_cluster else 'bad')

    # Visualize the clusters using two principal components for simplicity
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    customer_data['PC1'] = principal_components[:, 0]
    customer_data['PC2'] = principal_components[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_data, x='PC1', y='PC2', hue='Label', palette='viridis')
    plt.title('RFMS Score Clusters with Temporal Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Label')
    plt.show()

    return customer_data

# Example usage
file_path = 'transactions.csv'
result_df = process_customer_data(file_path)
print(result_df)
