import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spotify_data = pd.read_csv('music.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(spotify_data.head())

# Check for missing values and clean data
missing_values = spotify_data.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

spotify_data_cleaned = spotify_data.dropna()
missing_values_cleaned = spotify_data_cleaned.isnull().sum()
print("Columns with missing values after cleaning:")
print(missing_values_cleaned[missing_values_cleaned > 0])

# Data types of the columns
data_types = spotify_data_cleaned.dtypes
print("Data types of the columns:")
print(data_types)

# Unique values for categorical features
categorical_features = spotify_data_cleaned.select_dtypes(include=['object']).nunique()
print("Unique values for categorical features:")
print(categorical_features)

# Column names
print("Column names:")
print(spotify_data_cleaned.columns)

# Feature Engineering
# Convert categorical features to numerical using one-hot encoding
spotify_data_encoded = pd.get_dummies(
    spotify_data_cleaned, 
    columns=['playlist_genre', 'track_artist', 'playlist_name']
)

# Convert track_album_release_date to datetime
spotify_data_encoded['track_album_release_date'] = pd.to_datetime(
    spotify_data_encoded['track_album_release_date'], 
    format='%d-%m-%Y',  # Adjust the format if needed
    errors='coerce',
    dayfirst=True
)

# Drop rows with NaN values in 'track_album_release_date'
spotify_data_encoded = spotify_data_encoded.dropna(subset=['track_album_release_date'])

# Verify the removal of NaN values
print("Columns with missing values after additional cleaning:")
print(spotify_data_encoded.isnull().sum()[spotify_data_encoded.isnull().sum() > 0])

# Generate dummy target variable
np.random.seed(42)
spotify_data_encoded['repeated_play'] = np.random.randint(2, size=len(spotify_data_encoded))

# Split the dataset into features and target variable
features = spotify_data_encoded.drop(columns=['repeated_play'])
target = spotify_data_encoded['repeated_play']

# Ensure only numeric features are included
features_numeric = features.select_dtypes(include=[np.number])

# Normalize the numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Model Building
# Create Random Forest model without hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Function to generate song recommendations
def recommend_songs(user_id, model, df, scaler, features, top_n=5):
    user_data = df.copy()
    user_data_scaled = scaler.transform(user_data[features])
    user_data['probability'] = model.predict_proba(user_data_scaled)[:, 1]
    recommendations = user_data.sort_values(by='probability', ascending=False).head(top_n)
    return recommendations[['track_id', 'track_name', 'probability']]

# Example of recommending songs for a user
user_id = 1
recommendations = recommend_songs(user_id, rf, spotify_data_encoded, scaler, features_numeric.columns)
print(recommendations)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot 1: Histogram of Track Popularity
plt.figure(figsize=(12, 8))
sns.histplot(spotify_data_cleaned['track_popularity'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Track Popularity', fontsize=16)
plt.xlabel('Track Popularity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()

# Plot 2: Bar Plot of Top 10 Artists by Popularity
top_artists = spotify_data_cleaned.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
plt.title('Top 10 Artists by Average Popularity', fontsize=16)
plt.xlabel('Average Popularity', fontsize=14)
plt.ylabel('Artist', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Plot 3: Heatmap of Feature Correlations
numeric_features = spotify_data_cleaned.select_dtypes(include=[np.number])
correlation_matrix = numeric_features.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, center=0)
plt.title('Heatmap of Feature Correlations', fontsize=16)
plt.show()

# Plot 4: Actual vs Predicted Values
plt.figure(figsize=(12, 8))
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat='density', linewidth=2)
sns.histplot(y_pred, color='red', label='Predicted', kde=True, stat='density', linewidth=2)
plt.title('Actual vs Predicted Values', fontsize=16)
plt.xlabel('Repeated Play', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
