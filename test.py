import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spotify_data = pd.read_csv('music.csv')

# Data Summary
print("Dataset Summary:")
print(spotify_data.info())
print("First few rows:")
print(spotify_data.head())

# Drop missing values
spotify_data_cleaned = spotify_data.dropna()

# Convert categorical features to numerical using one-hot encoding
spotify_data_encoded = pd.get_dummies(
    spotify_data_cleaned, 
    columns=['playlist_genre', 'track_artist', 'playlist_name']
)

# Convert track_album_release_date to datetime
spotify_data_encoded['track_album_release_date'] = pd.to_datetime(
    spotify_data_encoded['track_album_release_date'], 
    format='%d-%m-%Y', errors='coerce', dayfirst=True
)

# Drop rows with NaN values in 'track_album_release_date'
spotify_data_encoded = spotify_data_encoded.dropna(subset=['track_album_release_date'])

# Ensure only numeric features are included
features_numeric = spotify_data_encoded.select_dtypes(include=[np.number])

# Data Visualization
plt.figure(figsize=(12, 6))
sns.histplot(spotify_data_cleaned['track_popularity'], bins=30, kde=True, color='blue')
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# Normalize the numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(features_scaled)

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(features_scaled)

# Visualize PCA results with K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=kmeans_labels, palette='viridis', s=50, edgecolor='k')
plt.title('PCA Projection with K-Means Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Visualize PCA results with DBSCAN clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=dbscan_labels, palette='coolwarm', s=50, edgecolor='k')
plt.title('PCA Projection with DBSCAN Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Print the explained variance ratio of PCA
print("Explained variance ratio of PCA:", pca.explained_variance_ratio_)

# Model Evaluation
print("Number of clusters found by K-Means:", len(set(kmeans_labels)))
print("Number of clusters found by DBSCAN:", len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))

# Next Steps and Improvements
print("Next Steps:")
print("- Tune hyperparameters for K-Means and DBSCAN (e.g., number of clusters, epsilon, min_samples)")
print("- Experiment with additional clustering methods (e.g., Hierarchical Clustering)")
print("- Use domain knowledge to interpret clusters and validate insights")
print("- Incorporate more relevant features or external data sources for better clustering")
