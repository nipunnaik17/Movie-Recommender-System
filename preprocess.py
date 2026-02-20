# preprocess.py (Updated to use pickle for any future extensions, but CSV is still used here)
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle  # Added for potential future use

# Load movies dataset with error handling
try:
    movies = pd.read_csv('movies.csv')
except FileNotFoundError:
    print("Error: 'movies.csv' not found. Please ensure the file exists.")
    exit(1)

# Split genres into list
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Encode genres using One-Hot Encoding
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genres'])

genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=movies.index)
movies_encoded = pd.concat([movies[['movieId', 'title']], genre_df], axis=1)

movies_encoded.to_csv('movies_encoded.csv', index=False)
print("Preprocessing complete. 'movies_encoded.csv' saved.")