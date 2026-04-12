"""
preprocessing.py

Responsible for:
- Cleaning data
- Creating user-item matrix
"""
import pandas as pd
import numpy as np
import re


BAD_TAG_BLACKLIST = {
    "to read",
    "currently reading",
    "read",
    "owned",
    "default",
    "favorites",
    "favourites",
    "my books",
    "books i own",
    "ebook",
    "kindle",
    "audiobook",
    "library",
    "wishlist",
    "wish list",
}


TAG_SYNONYMS = {
    "sci fi": "science fiction",
    "scifi": "science fiction",
    "science-fiction": "science fiction",
    "ya": "young adult",
    "young-adult": "young adult",
    "ya fiction": "young adult",
    "bio": "biography",
    "biographies": "biography",
    "memoirs": "memoir",
    "historical-fiction": "historical fiction",
    "rom-com": "romance",
    "thrillers": "thriller",
    "classics": "classic",
}


def _normalize_tag_name(tag_name):
    value = str(tag_name).lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    value = TAG_SYNONYMS.get(value, value)
    return value


def _is_informative_tag(tag_name):
    if not tag_name:
        return False
    if tag_name in BAD_TAG_BLACKLIST:
        return False
    if len(tag_name) < 3:
        return False

    # Remove tags dominated by non-alphanumeric symbols or numbers.
    if not re.search(r"[a-z]", tag_name):
        return False
    alnum_chars = sum(ch.isalnum() for ch in tag_name)
    if alnum_chars / max(len(tag_name), 1) < 0.6:
        return False

    digit_chars = sum(ch.isdigit() for ch in tag_name)
    if digit_chars / max(len(tag_name), 1) > 0.35:
        return False

    return True


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


def build_book_tag_features(
    book_tags_df,
    tags_df,
    min_count=5,
    max_tags_per_book=12,
):
    """
    Build cleaned, weighted tag features per book.

    Returns a DataFrame with columns:
    - book_id
    - tag_name
    - tag_weight (normalized per book)
    """
    merged = book_tags_df.merge(tags_df, on="tag_id", how="left")
    merged = merged.rename(columns={"goodreads_book_id": "book_id"})

    merged["tag_name"] = merged["tag_name"].fillna("").map(_normalize_tag_name)
    merged = merged[merged["tag_name"].map(_is_informative_tag)]

    # Keep only meaningful tag signals per book.
    filtered = merged[merged["count"] >= min_count].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["book_id", "tag_name", "tag_weight"])

    # Merge duplicate variants after normalization/synonym mapping.
    filtered = (
        filtered.groupby(["book_id", "tag_name"], as_index=False)["count"]
        .sum()
    )

    total_books = filtered["book_id"].nunique()
    tag_book_frequency = filtered.groupby("tag_name")["book_id"].nunique()

    # TF-IDF style weighting reduces dominance of globally common tags.
    idf_by_tag = np.log1p((1 + total_books) / (1 + tag_book_frequency)) + 1.0

    filtered["tf"] = np.log1p(filtered["count"].astype(float))
    filtered["idf"] = filtered["tag_name"].map(idf_by_tag)
    filtered["raw_weight"] = filtered["tf"] * filtered["idf"]

    filtered = filtered.sort_values(["book_id", "raw_weight"], ascending=[True, False])
    filtered = filtered.groupby("book_id").head(max_tags_per_book).copy()

    total_per_book = filtered.groupby("book_id")["raw_weight"].transform("sum")
    filtered["tag_weight"] = filtered["raw_weight"] / total_per_book

    return filtered[["book_id", "tag_name", "tag_weight"]].reset_index(drop=True)


def build_user_tag_profile(
    ratings_df,
    to_read_df,
    book_tag_features_df,
    min_positive_rating=4.0,
    to_read_weight=0.7,
):
    """
    Build per-user preference scores over tags from:
    - Positive ratings (>= min_positive_rating)
    - To-read intent (weighted by to_read_weight)
    """
    tag_cols = ["book_id", "tag_name", "tag_weight"]
    if book_tag_features_df.empty:
        return pd.DataFrame(columns=["user_id", "tag_name", "tag_score"])

    positive_ratings = ratings_df.loc[
        ratings_df["rating"] >= min_positive_rating,
        ["user_id", "book_id"],
    ].copy()

    from_ratings = positive_ratings.merge(
        book_tag_features_df[tag_cols],
        on="book_id",
        how="inner",
    )
    from_ratings["signal_weight"] = from_ratings["tag_weight"]

    from_to_read = to_read_df[["user_id", "book_id"]].merge(
        book_tag_features_df[tag_cols],
        on="book_id",
        how="inner",
    )
    from_to_read["signal_weight"] = from_to_read["tag_weight"] * to_read_weight

    combined = pd.concat([from_ratings, from_to_read], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["user_id", "tag_name", "tag_score"])

    profile = (
        combined.groupby(["user_id", "tag_name"], as_index=False)["signal_weight"]
        .sum()
        .rename(columns={"signal_weight": "tag_score"})
    )

    score_totals = profile.groupby("user_id")["tag_score"].transform("sum")
    profile["tag_score"] = profile["tag_score"] / score_totals

    return profile
