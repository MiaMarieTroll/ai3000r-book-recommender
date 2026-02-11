"""
collaborative_model.py

User-based collaborative filtering using KNN
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np


# ============================================
# TODO 1: Build KNN model
# ============================================

def build_knn_model(user_item_matrix):
    """
    Use cosine similarity
    Fit KNN model
    """
    pass


# ============================================
# TODO 2: Find similar users
# ============================================

def find_similar_users(model, user_item_matrix, user_id, k=5):
    """
    Return k most similar users
    """
    pass


# ============================================
# TODO 3: Generate recommendations
# ============================================

def recommend_books(user_id, user_item_matrix, books_df, model, n=5):
    """
    1. Find similar users
    2. Get books they liked
    3. Remove books user already rated
    4. Return top n recommendations
    """
    pass
