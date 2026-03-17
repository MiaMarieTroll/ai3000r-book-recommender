"""
preprocessing.py

Responsible for:
- Cleaning data
- Creating user-item matrix
"""
def clean_ratings(ratings_df):
    """
    Remove:
    - Missing values
    - Duplicates
    """
    ratings_df = ratings_df.dropna()
    ratings_df = ratings_df.drop_duplicates()
    print("Ratings after cleaning:", ratings_df.shape)
    return ratings_df

def create_user_item_matrix(ratings_df):
    """
    Create pivot table:
    Rows: user_id
    Columns: book_id
    Values: rating
    """
    user_item_matrix = ratings_df.pivot_table(
        index="user_id",
        columns="book_id",
        values="rating"
    )
    print("User-item matrix shape:", user_item_matrix.shape)
    return user_item_matrix

def fill_missing(user_item_matrix):
    """
    Replace NaN with 0
    """
    user_item_matrix = user_item_matrix.fillna(0)
    print("Missing values after fill:", user_item_matrix.isnull().sum().sum())
    return user_item_matrix
