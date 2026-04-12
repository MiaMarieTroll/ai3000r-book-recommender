"""
data_loader.py

Responsible for:
- Loading datasets
- Basic validation
"""

import pandas as pd


def _validate_required_columns(df, required_columns, dataset_name):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {sorted(missing_columns)}"
        )

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

    _validate_required_columns(
        ratings_df,
        required_columns={"user_id", "book_id", "rating"},
        dataset_name="ratings data",
    )

    return ratings_df


def load_book_tags(path):
    """
    Load book_tags.csv
    Return: pandas DataFrame
    """
    book_tags_df = pd.read_csv(path)
    print("Loaded book_tags dataset with shape:", book_tags_df.shape)

    _validate_required_columns(
        book_tags_df,
        required_columns={"goodreads_book_id", "tag_id", "count"},
        dataset_name="book_tags data",
    )

    return book_tags_df


def load_tags(path):
    """
    Load tags.csv
    Return: pandas DataFrame
    """
    tags_df = pd.read_csv(path)
    print("Loaded tags dataset with shape:", tags_df.shape)

    _validate_required_columns(
        tags_df,
        required_columns={"tag_id", "tag_name"},
        dataset_name="tags data",
    )

    return tags_df


def load_to_read(path):
    """
    Load to_read.csv
    Return: pandas DataFrame
    """
    to_read_df = pd.read_csv(path)
    print("Loaded to_read dataset with shape:", to_read_df.shape)

    _validate_required_columns(
        to_read_df,
        required_columns={"user_id", "book_id"},
        dataset_name="to_read data",
    )

    return to_read_df

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


