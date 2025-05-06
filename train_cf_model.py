import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the necessary data
ratings_df = pd.read_csv('ratings.csv')
user_data_df = pd.read_csv('user_data.csv')  # This includes user preferences (e.g., genres)
books_df = pd.read_csv('data/books.csv')  # Book details like title, author, etc.

def train_cf_model(ratings_df, user_data_df):
    # Pivot the ratings_df into a user-item matrix
    pivot_table = ratings_df.pivot_table(index='user_id', columns='book_id', values='rating')
    
    # Use NearestNeighbors for collaborative filtering based on cosine similarity
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(pivot_table.fillna(0))  # Fill missing ratings with zero
    
    return model

def get_user_preferences(user_id, user_data_df):
    # Retrieve user preferences from the user_data_df (like favorite genres)
    user_preferences = user_data_df[user_data_df['user_id'] == user_id]
    return user_preferences

def predict_cf_ratings(user_id, books_df, cf_model, top_n=10):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    user_books = user_ratings['book_id'].tolist()
    
    # Get predictions for books not rated by the user
    predictions = []
    for book_id in books_df['book_id']:
        if book_id not in user_books:
            # Calculate similarity score for this book (using the CF model)
            distances, indices = cf_model.kneighbors([books_df[books_df['book_id'] == book_id].values[0]], n_neighbors=top_n)
            predictions.append((book_id, np.mean(distances)))
    
    # Sort predictions by similarity score
    predictions.sort(key=lambda x: x[1], reverse=False)
    
    return pd.DataFrame(predictions, columns=['book_id', 'predicted_rating'])

# Train the collaborative filtering model
cf_model = train_cf_model(ratings_df, user_data_df)
