# knn_model.py (Updated to use pickle directly)
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle  # Use pickle instead of joblib

# Load encoded movies
movies_encoded = pd.read_csv('movies_encoded.csv')

# Features: genre columns
features = movies_encoded.drop(['movieId', 'title'], axis=1)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(features)

# Save the feature data (the same data the model was trained on)
# This allows the app to load the data structure needed for querying
with open('movie_features.pkl', 'wb') as f:
    pickle.dump(features, f)

# Save the model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("KNN model and features successfully saved using pickle.")