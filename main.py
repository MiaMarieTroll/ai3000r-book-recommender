"""
main.py

Main pipeline for Book Recommender System
"""

from src.data_loader import load_books, load_ratings, data_summary
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing
from src.collaborative_model import build_knn_model, recommend_books
from src.matrix_factorization_model import build_svd_model, get_recommendations_svd
from src.baseline_model import compute_average_ratings, get_top_books
from src.evaluation import evaluate_model, evaluate_model_svd
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
        max_users=100,
        min_test_rating=3.0
    )

    print_result("KNN Evaluation summary:", results)

    # ============================================
    # Matrix Factorization Model (SVD)
    # ============================================
    print("\n" + "=" * 60)
    print("MATRIX FACTORIZATION (SVD) MODEL")
    print("=" * 60)
    
    print("\nBuilding SVD model...")
    svd_model = build_svd_model(user_item_matrix, n_factors=50)
    print("SVD model fitted successfully.")

    target_user_id = 5
    svd_recommendations = get_recommendations_svd(
        user_id=target_user_id,
        user_item_matrix=user_item_matrix,
        books_df=books,
        svd_model=svd_model
    )
    print_result("SVD Recommendations for user id " + str(target_user_id) + ":", svd_recommendations)

    # ============================================
    # Multiple Users Recommendations (SVD)
    # ============================================
    print("\nGenerating SVD recommendations for multiple users...")
    print("=" * 60)
    
    svd_all_recommendations = []

    for uid in test_user_ids:
        if uid in user_item_matrix.index:
            user_recs = get_recommendations_svd(
                user_id=uid,
                user_item_matrix=user_item_matrix,
                books_df=books,
                svd_model=svd_model,
                n=3
            )
            
            # Get the top recommended book for each user
            if len(user_recs) > 0:
                top_rec = user_recs.iloc[0]
                svd_all_recommendations.append({
                    "user_id": uid,
                    "recommended_book": top_rec["title"],
                    "author": top_rec["authors"],
                    "score": round(top_rec["score"], 2)
                })

    if svd_all_recommendations:
        svd_comparison_df = pd.DataFrame(svd_all_recommendations)
        print_result("SVD Top Recommendation per User:", svd_comparison_df)

    # ============================================
    # Evaluation: SVD Model
    # ============================================
    print("\nRunning SVD evaluation...")
    svd_results = evaluate_model_svd(
        ratings_df=ratings,
        books_df=books,
        k=5,
        test_size=0.2,
        random_state=42,
        max_users=100,
        min_test_rating=3.0,
        n_factors=50
    )

    print_result("SVD Evaluation summary:", svd_results)

    # ============================================
    # SVD Factor Tuning
    # ============================================
    print("\nRunning SVD factor tuning...")
    factor_grid = [20, 50, 80, 120]
    tuning_rows = []

    for factor in factor_grid:
        tuned_result = evaluate_model_svd(
            ratings_df=ratings,
            books_df=books,
            k=5,
            test_size=0.2,
            random_state=42,
            max_users=100,
            min_test_rating=3.0,
            n_factors=factor,
            progress_every=0,
        )
        tuning_rows.append(
            {
                "n_factors": factor,
                "precision@5": tuned_result["precision@k"],
                "recall@5": tuned_result["recall@k"],
            }
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        by=["precision@5", "recall@5"],
        ascending=[False, False],
    )
    print_result("SVD Factor Tuning Results:", tuning_df)

    best_factors = int(tuning_df.iloc[0]["n_factors"])
    best_precision = float(tuning_df.iloc[0]["precision@5"])
    best_recall = float(tuning_df.iloc[0]["recall@5"])
    print(f"Best SVD factors by Precision@5/Recall@5: {best_factors}")

    # ============================================
    # Model Comparison
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_results = pd.DataFrame(
        {
            "Model": ["KNN", "SVD (50 factors)", f"SVD tuned ({best_factors} factors)"],
            "Precision@5": [results["precision@k"], svd_results["precision@k"], best_precision],
            "Recall@5": [results["recall@k"], svd_results["recall@k"], best_recall],
            "Evaluated Users": [results["evaluated_users"], svd_results["evaluated_users"], svd_results["evaluated_users"]],
        }
    )
    
    print_result("Performance Comparison:", comparison_results)

if __name__ == "__main__":
    main()
