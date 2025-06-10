import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# Load dataset (MovieLens 100k format)

# u.data format: user_id, item_id, rating, timestamp
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv("D:\\GitHub\\cloudcredits-\\Project 5\\ml-100k\\rating.csv", sep='\t', names=columns)

# Optional: Load movie titles (if you have u.item)
# movie_titles = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill missing values with 0 or mean (here using 0)
filled_matrix = user_item_matrix.fillna(0)

# Compute similarity between items
item_similarity = cosine_similarity(filled_matrix.T)  # Transpose so we compare items
item_similarity_df = pd.DataFrame(item_similarity, index=filled_matrix.columns, columns=filled_matrix.columns)

# Predict ratings for a user based on item similarity
def predict_ratings(user_id, top_n=10):
    user_ratings = user_item_matrix.loc[user_id]
    similar_scores = pd.Series(dtype=np.float64)

    for movie_id, rating in user_ratings.dropna().items():
        similarity_scores = item_similarity_df[movie_id]
        weighted_scores = similarity_scores * rating
        similar_scores = similar_scores.add(weighted_scores, fill_value=0)

    # Remove movies already rated
    already_rated = user_ratings[user_ratings.notna()].index
    similar_scores = similar_scores.drop(already_rated, errors='ignore')

    top_movies = similar_scores.sort_values(ascending=False).head(top_n)
    return top_movies

# Example: Get recommendations for user 10
recommendations = predict_ratings(user_id=10, top_n=10)
print("Top 10 movie recommendations for user 10:")
print(recommendations)
