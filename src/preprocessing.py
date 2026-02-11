"""
preprocessing.py

Responsible for:
- Cleaning data
- Creating user-item matrix
"""

import pandas as pd


# ============================================
# TODO 1: Clean ratings data
# ============================================

def clean_ratings(ratings_df):
    """
    Remove:
    - Missing values
    - Duplicates
    """
    pass


# ============================================
# TODO 2: Create User-Item Matrix
# ============================================

def create_user_item_matrix(ratings_df):
    """
    Create pivot table:
    Rows: user_id
    Columns: book_id
    Values: rating
    """
    pass


# ============================================
# TODO 3: Fill missing values
# ============================================

def fill_missing(user_item_matrix):
    """
    Replace NaN with 0
    """
    pass
