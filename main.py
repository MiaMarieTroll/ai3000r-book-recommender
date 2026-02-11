"""
main.py

Main pipeline for Book Recommender System
"""

from src.data_loader import load_books, load_ratings
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing
from src.collaborative_model import build_knn_model, recommend_books
from src.baseline_model import compute_average_ratings, get_top_books
from src.evaluation import evaluate_model


def main():

    # ============================================
    # TODO 1: Load Data
    # ============================================
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")

    # ============================================
    # TODO 2: Preprocess
    # ============================================
    ratings = clean_ratings(ratings)
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_matrix = fill_missing(user_item_matrix)

    # ============================================
    # TODO 3: Baseline Model
    # ============================================
    avg_ratings = compute_average_ratings(ratings)
    top_books = get_top_books(avg_ratings, books)

    # ============================================
    # TODO 4: Collaborative Filtering
    # ============================================
    knn_model = build_knn_model(user_item_matrix)

    recommendations = recommend_books(
        user_id=1,
        user_item_matrix=user_item_matrix,
        books_df=books,
        model=knn_model
    )

    # ============================================
    # TODO 5: Evaluation
    # ============================================
    evaluate_model()

    print("Top Books (Baseline):")
    print(top_books)

    print("Recommendations:")
    print(recommendations)


if __name__ == "__main__":
    main()
