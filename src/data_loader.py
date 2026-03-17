"""
data_loader.py

Responsible for:
- Loading datasets
- Basic validation
"""

import pandas as pd

def load_books(path):
    """
    Load books.csv
    Return: pandas DataFrame
    """
    books_df = pd.read_csv(path)
    print("Loaded books dataset with shape:", books_df.shape)
    return books_df

def load_ratings(path):
    """
    Load ratings.csv
    Return: pandas DataFrame
    """
    ratings_df = pd.read_csv(path)
    print("Loaded ratings dataset with shape:", ratings_df.shape)

    required_columns = {"user_id", "book_id", "rating"}
    missing_columns = required_columns - set(ratings_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in ratings data: {sorted(missing_columns)}")

    return ratings_df

def data_summary(df):
    """
    Print basic dataset info:
    - head()
    - info()
    - missing values
    """
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())
