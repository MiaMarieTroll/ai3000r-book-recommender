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
    #print_result("Top Books (Baseline):", top_books)

    # ============================================
    # Collaborative Filtering
    # ============================================
    print("\nRunning KNN Model...")
    knn_model = build_knn_model(user_item_matrix)
    print("KNN model fitted successfully.")

    target_user_id = 5
    recommendations = recommend_books(
        user_id=target_user_id,
        user_item_matrix=user_item_matrix,
        books_df=books,
        model=knn_model
    )
    print_result("Recommendations for user id " + str(target_user_id) + ":", recommendations)

    # ============================================
    # Multiple Users Recommendations
    # ============================================
    print("Generating recommendations for multiple users...")
    print("=" * 60)
    
    test_user_ids = [1, 5, 10, 20]
    all_recommendations = []

    for uid in test_user_ids:
        if uid in user_item_matrix.index:
            user_recs = recommend_books(
                user_id=uid,
                user_item_matrix=user_item_matrix,
                books_df=books,
                model=knn_model,
                n=3
            )
            
            # Get the top recommended book for each user
            if len(user_recs) > 0:
                top_rec = user_recs.iloc[0]
                all_recommendations.append({
                    "user_id": uid,
                    "recommended_book": top_rec["title"],
                    "author": top_rec["authors"],
                    "score": round(top_rec["score"], 2)
                })

    if all_recommendations:
        comparison_df = pd.DataFrame(all_recommendations)
        print_result("Top Recommendation per User:", comparison_df)

    # ============================================
    # TODO 5: Evaluation
    print("\nRunning evaluation...")
    results = evaluate_model(
        ratings_df=ratings,
        books_df=books,
        k=5,
        test_size=0.2,
        random_state=42,
        max_users=100
    )

    print_result("Evaluation summary:", results)

if __name__ == "__main__":
    main()
