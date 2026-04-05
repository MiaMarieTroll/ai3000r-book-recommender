"""
matrix_factorization_model.py

Matrix Factorization (SVD) for collaborative filtering.
More effective than KNN for sparse rating data.
"""

from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


def build_svd_model(user_item_matrix, n_factors=50):
    """
    Build SVD model for matrix factorization.
    
    Parameters
    ----------
    user_item_matrix : DataFrame
        User-item rating matrix
    n_factors : int
        Number of latent factors to learn. Default 50.
        Lower = faster, Higher = better quality (usual range 20-100)
    
    """
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(user_item_matrix.values)
    return svd


def get_recommendations_svd(user_id, user_item_matrix, books_df, svd_model, n=5):
    """
    Generate recommendations using SVD latent factors.
    
    Steps
    -----
    1. Get user latent factor vector
    2. Compute similarity to all item latent factors
    3. Remove already-rated items
    4. Return top n recommendations
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix.")

    # Get user's latent vector
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_matrix = user_item_matrix.to_numpy()
    user_vector = user_matrix[user_idx]
    
    # Project user to latent space
    user_latent = svd_model.transform(user_vector.reshape(1, -1))[0]
    
    # Project items (components are item latent factors)
    item_latents = svd_model.components_.T  # (n_items, n_factors)
    
    # Compute cosine similarity between user and all items
    similarities = item_latents @ user_latent / (
        np.linalg.norm(item_latents, axis=1) * np.linalg.norm(user_latent) + 1e-9
    )
    
    # Remove already-rated items
    already_rated = user_vector > 0
    similarities[already_rated] = -np.inf
    
    # Get top n
    top_idx = np.argsort(similarities)[-n:][::-1]
    top_idx = top_idx[np.isfinite(similarities[top_idx])]
    
    if len(top_idx) == 0:
        return pd.DataFrame(columns=["book_id", "title", "authors", "score"])
    
    book_ids = user_item_matrix.columns.to_numpy()[top_idx]
    scores = similarities[top_idx]
    
    recommendations = pd.DataFrame({
        "book_id": book_ids,
        "score": scores,
    })
    
    # Merge with metadata
    book_metadata = books_df[["id", "title", "authors"]].copy()
    merged = recommendations.merge(
        book_metadata,
        left_on="book_id",
        right_on="id",
        how="left"
    )
    
    selected_cols = [col for col in ["book_id", "title", "authors", "score"] if col in merged.columns]
    result = merged[selected_cols].reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    
    return result
