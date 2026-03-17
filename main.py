"""
main.py

Main pipeline for Book Recommender System
"""

from src.data_loader import load_books, load_ratings, data_summary
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing
from src.collaborative_model import build_knn_model, recommend_books
from src.baseline_model import compute_average_ratings, get_top_books
from src.evaluation import evaluate_model
import pandas as pd


def print_result(title, value, max_rows=None):
    print(f"\n{title}")
    if isinstance(value, pd.DataFrame):
        table = value.head(max_rows) if max_rows is not None else value
        print(table.to_string(index=False))
    else:
        print(value)


def main():
    # ============================================
    # Load Data
    # ============================================
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")

    print("\nBooks data summary:")
    data_summary(books)

    print("\nRatings data summary:")
    data_summary(ratings)

    # ============================================
    # Preprocess
    # ============================================
    ratings = clean_ratings(ratings)
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_matrix = fill_missing(user_item_matrix)
    print("User-item matrix ready:", user_item_matrix.shape)

    # ============================================
    # Baseline Model
    # ============================================
    print("\nRunning baseline model...")
    avg_ratings = compute_average_ratings(ratings)
    top_books = get_top_books(avg_ratings, books)
    print_result("Baseline model complete. Top 5 books:", top_books, max_rows=5)

    # ============================================
    # TODO 4: Collaborative Filtering
    # ============================================
    print("\nRunning KNN Model...")
    knn_model = build_knn_model(user_item_matrix)    

    recommended_user, recommendations = recommend_books(
        user_id=1,
        user_item_matrix=user_item_matrix,
        books_df=books,
        model=knn_model
    )

    # ============================================
    # TODO 5: Evaluation
    # ============================================
    evaluate_model()

    print_result("Top Books (Baseline):", top_books)
    print("Recommendations for user id:", recommended_user)
    print_result("Recommendations:", recommendations)


if __name__ == "__main__":
    main()
