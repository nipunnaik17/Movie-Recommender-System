from flask import Flask, render_template, request, url_for, g
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Define paths for static files and template
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')

# Global variables for data and model
movies, knn, features = None, None, None

def load_data():
    """Loads all necessary data and model files."""
    global movies, knn, features
    try:
        # 1. Load movies data (includes titles and IDs)
        movies = pd.read_csv('movies_encoded.csv')
        
        # 2. Load KNN model
        with open('knn_model.pkl', 'rb') as f:
            knn = pickle.load(f)
        
        # 3. Load features data (used to query the model)
        # We assume it was saved as a joblib/pickle file from the training script
        try:
            with open('movie_features.pkl', 'rb') as f:
                features = pickle.load(f)
        except FileNotFoundError:
            # Fallback: if the features file is missing, calculate features from the movies CSV
            print("movie_features.pkl not found. Calculating features from movies_encoded.csv.")
            features = movies.drop(['movieId', 'title'], axis=1)

        print("Files loaded successfully.")

    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"CRITICAL ERROR loading files: {e}. Ensure all data/model files are correctly generated.")
        # Set critical variables to None to prevent application crash
        movies, knn, features = None, None, None

# Load data on startup
load_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Pass current time for cache-busting the plot image
    g.now = datetime.now().strftime('%Y%m%d%H%M%S')
    
    recommendations = None
    plot_path = None
    movie_name = None
    
    # Check for critical loading failures first
    if movies is None or knn is None or features is None:
        recommendations = ['A critical data file failed to load. Please ensure all files are correctly generated and uncorrupted.']
        return render_template('index.html', recommendations=recommendations, plot_path=plot_path)

    # --- POST Request Handling (Recommendation Logic) ---
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')

        if not movie_name:
            recommendations = ['Please enter a movie name to receive recommendations.']
        elif movie_name in movies['title'].values:
            try:
                # Find the index of the movie
                idx = movies[movies['title'] == movie_name].index[0]
                
                # Extract the feature vector for the chosen movie. 
                # Ensure it's in the 2D array format required by kneighbors.
                if isinstance(features, pd.DataFrame):
                    input_vector = features.iloc[idx].to_numpy().reshape(1, -1)
                elif isinstance(features, np.ndarray):
                    input_vector = features[idx].reshape(1, -1)
                else:
                    raise TypeError("Features must be a Pandas DataFrame or NumPy array.")

                # Find the nearest neighbors (k=6 to get 5 unique recommendations + the movie itself)
                distances, indices = knn.kneighbors(input_vector, n_neighbors=6)
                
                # Get the top 5 recommendations (excluding the input movie itself at index 0)
                recommended_indices = indices[0][1:]
                recommendations = movies['title'].iloc[recommended_indices].tolist()
                
                # --- Visualization Logic ---
                sim_scores = 1 - distances[0][1:]  # similarity = 1 - distance
                
                # Configure Matplotlib and Seaborn
                plt.figure(figsize=(10, 6))
                sns.set_theme(style="whitegrid")
                
                # Create the bar plot
                sns.barplot(x=recommendations, y=sim_scores, palette="viridis")
                
                # Enhance plot readability
                plt.ylabel('Similarity Score (Cosine)', fontsize=12)
                plt.title(f'Top 5 Movie Similarities to "{movie_name}"', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.ylim(0, 1) # Set y-limit from 0 to 1 for similarity
                
                # Ensure the directory exists and save the plot
                os.makedirs(PLOTS_DIR, exist_ok=True)
                plot_filename = 'similarity.png'
                plot_path = os.path.join(PLOTS_DIR, plot_filename)
                
                # Save the plot (bbox_inches='tight' prevents labels from being cut off)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close() # Close plot figure to free memory

            except Exception as e:
                print(f"Error during recommendation process: {e}")
                recommendations = ['An error occurred during the recommendation calculation. Check the console for details.']
                
        else:
            # Movie name not found
            recommendations = [f'Movie "{movie_name}" not found in our database. Please check the spelling.']
            
    # Render the template
    return render_template('index.html', 
                           movie_name=movie_name,
                           recommendations=recommendations, 
                           plot_path=plot_path,
                           all_movies=movies['title'].tolist() if movies is not None else [])

if __name__ == '__main__':
    # You must have a 'templates' folder with 'index.html' and a 'static/plots' folder
    # in the same directory as this script for it to run locally.
    app.run(debug=True)