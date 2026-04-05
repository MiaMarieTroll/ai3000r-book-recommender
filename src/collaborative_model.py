"""
collaborative_model.py

User-based collaborative filtering using KNN
"""

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np


# ============================================
# Build KNN model
# ============================================

def build_knn_model(user_item_matrix, normalize=False):
    """
    Build and fit a KNN model using cosine similarity.
    
    Parameters
    ----------
    user_item_matrix : DataFrame
        User-item rating matrix
    normalize : bool
        Unused - kept for compatibility.
    
    """
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(user_item_matrix.values)
    return model


# ============================================
# Find similar users
# ============================================

def find_similar_users(model, user_item_matrix, user_id, k=5):
    """
    Return k most similar users to the given user_id.

    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix.")

    # +1 because the nearest neighbor will usually be the user themself
    n_neighbors = min(k + 1, len(user_item_matrix))

    user_position = user_item_matrix.index.get_loc(user_id)
    user_vector = user_item_matrix.iloc[user_position].values.reshape(1, -1)

    distances, indices = model.kneighbors(user_vector, n_neighbors=n_neighbors)

    similar_users = []
    for distance, idx in zip(distances[0], indices[0]):
        neighbor_user_id = user_item_matrix.index[idx]

        if neighbor_user_id == user_id:
            continue

        similarity = 1 - distance
        similar_users.append((neighbor_user_id, similarity))

    return similar_users[:k]


# ============================================
# Generate recommendations
# ============================================

def recommend_books(user_id, user_item_matrix, books_df, model, n=5):
    """
    Generate top-n book recommendations for a user.

    Steps
    -----
    1. Find similar users
    2. Get books they liked
    3. Remove books the target user already rated
    4. Return top n recommendations

    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix.")

    similar_users = find_similar_users(model, user_item_matrix, user_id, k=5)

    if not similar_users:
        return pd.DataFrame(columns=["book_id", "title", "authors", "score"])

    target_user_ratings = user_item_matrix.loc[user_id]

    # Books already rated by target user
    already_rated = set(target_user_ratings[target_user_ratings > 0].index)

    # Weighted recommendation scores from similar users
    book_scores = {}
    book_counts = {}

    for similar_user_id, similarity in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user_id]

        liked_books = similar_user_ratings[similar_user_ratings > 0]

        for book_id, rating in liked_books.items():
            if book_id in already_rated:
                continue

            # Weighted score: rating * similarity (higher similarity = more weight)
            weighted_score = rating * similarity

            book_scores[book_id] = book_scores.get(book_id, 0) + weighted_score
            book_counts[book_id] = book_counts.get(book_id, 0) + 1

    if not book_scores:
        return pd.DataFrame(columns=["book_id", "title", "authors", "score"])

    recommendations = pd.DataFrame(
        {
            "book_id": list(book_scores.keys()),
            "score": list(book_scores.values()),
            "support": [book_counts[book_id] for book_id in book_scores.keys()],
        }
    )

    recommendations = recommendations.sort_values(
        by=["score", "support"], ascending=[False, False]
    ).head(n)

    # Keep only metadata columns needed for output to avoid book_id name collisions.
    book_metadata = books_df[["id", "title", "authors"]].copy()

    merged = recommendations.merge(
        book_metadata,
        left_on="book_id",
        right_on="id",
        how="left"
    )

    available_cols = ["book_id", "title", "authors", "score", "support"]
    selected_cols = [col for col in available_cols if col in merged.columns]

    result = merged[selected_cols].reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    # Return only the recommendation table so main.py can print/process it directly.
    return result
