"""
main.py

Main pipeline for Book Recommender System
"""

from src.data_loader import (
    load_books,
    load_ratings,
    load_book_tags,
    load_tags,
    load_to_read,
    data_summary,
)
from src.preprocessing import (
    clean_ratings,
    create_user_item_matrix,
    fill_missing,
    build_book_tag_features,
    build_user_tag_profile,
)
from src.collaborative_model import build_knn_model, recommend_books
from src.matrix_factorization_model import build_svd_model, get_recommendations_svd
from src.baseline_model import compute_average_ratings, get_top_books
from src.evaluation import (
    evaluate_model,
    evaluate_model_svd,
    evaluate_model_hybrid_knn,
    evaluate_model_hybrid_svd,
)
from src.hybrid_model import rerank_recommendations_hybrid
import pandas as pd


def print_result(title, value, max_rows=None):
    print(f"\n{title}")
    if isinstance(value, pd.DataFrame):
        table = value.head(max_rows) if max_rows is not None else value
        print(table.to_string(index=False))
    else:
        print(value)


def main():
    svd_factors = 300
    recommendation_n = 10
    candidate_n = 400
    target_user_id = 5

    # Hybrid weights (set to_read_weight=0.0 to disable to-read signal).
    hybrid_weights = {
        "collaborative_weight": 0.60,
        "content_weight": 0.30,
        "to_read_weight": 0.10,
    }

    # ============================================
    # Load Data
    # ============================================
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    book_tags = load_book_tags("data/book_tags.csv")
    tags = load_tags("data/tags.csv")
    to_read = load_to_read("data/to_read.csv")

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

    # Build content/intent features from tags and to-read signals.
    book_tag_features = build_book_tag_features(
        book_tags,
        tags,
        min_count=5,
        max_tags_per_book=10,
    )
    user_tag_profile = build_user_tag_profile(
        ratings,
        to_read,
        book_tag_features,
        to_read_weight=hybrid_weights["to_read_weight"],
    )
    print("Book-tag features shape:", book_tag_features.shape)
    print("User-tag profile shape:", user_tag_profile.shape)

    # ============================================
    # Baseline Model
    # ============================================
    print("\nRunning baseline model...")
    avg_ratings = compute_average_ratings(ratings)
    top_books = get_top_books(avg_ratings, books)
    print_result("Baseline model complete. Top 5 books:", top_books, max_rows=5)

    # ============================================
    # Collaborative Filtering
    # ============================================
    print("\nRunning KNN Model...")
    knn_model = build_knn_model(user_item_matrix)
    print("KNN model fitted successfully.")

    recommendations = recommend_books(
        user_id=target_user_id,
        user_item_matrix=user_item_matrix,
        books_df=books,
        model=knn_model,
        n=candidate_n,
        similar_k=40,
    )

    recommendations_hybrid = rerank_recommendations_hybrid(
        recommendations_df=recommendations,
        user_id=target_user_id,
        book_tag_features_df=book_tag_features,
        user_tag_profile_df=user_tag_profile,
        to_read_df=to_read,
        **hybrid_weights,
    ).head(recommendation_n)
    print_result(
        "Hybrid recommendations for user id " + str(target_user_id) + ":",
        recommendations_hybrid,
    )

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
                n=candidate_n,
                similar_k=40,
            )

            user_recs = rerank_recommendations_hybrid(
                recommendations_df=user_recs,
                user_id=uid,
                book_tag_features_df=book_tag_features,
                user_tag_profile_df=user_tag_profile,
                to_read_df=to_read,
                **hybrid_weights,
            ).head(recommendation_n)
            
            # Get the top recommended book for each user
            if len(user_recs) > 0:
                top_rec = user_recs.iloc[0]
                all_recommendations.append({
                    "user_id": uid,
                    "recommended_book": top_rec["title"],
                    "author": top_rec["authors"],
                    "score": round(top_rec["hybrid_score"], 3)
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

    print("\nRunning hybrid KNN evaluation...")
    hybrid_knn_results = evaluate_model_hybrid_knn(
        ratings_df=ratings,
        books_df=books,
        book_tag_features_df=book_tag_features,
        to_read_df=to_read,
        k=5,
        test_size=0.2,
        random_state=42,
        max_users=100,
        min_test_rating=3.0,
        candidate_n=400,
        **hybrid_weights,
    )
    print_result("Hybrid KNN Evaluation summary:", hybrid_knn_results)

    # ============================================
    # Matrix Factorization Model (SVD)
    # ============================================
    print("\n" + "=" * 60)
    print("MATRIX FACTORIZATION (SVD) MODEL")
    print("=" * 60)
    
    print("\nBuilding SVD model...")
    svd_model = build_svd_model(user_item_matrix, n_factors=svd_factors)
    print("SVD model fitted successfully.")

    svd_recommendations = get_recommendations_svd(
        user_id=target_user_id,
        user_item_matrix=user_item_matrix,
        books_df=books,
        svd_model=svd_model,
        n=recommendation_n,
    )

    svd_recommendations_hybrid = rerank_recommendations_hybrid(
        recommendations_df=svd_recommendations,
        user_id=target_user_id,
        book_tag_features_df=book_tag_features,
        user_tag_profile_df=user_tag_profile,
        to_read_df=to_read,
        **hybrid_weights,
    )
    print_result(
        "SVD hybrid recommendations for user id " + str(target_user_id) + ":",
        svd_recommendations_hybrid,
    )

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
                n=recommendation_n,
            )

            user_recs = rerank_recommendations_hybrid(
                recommendations_df=user_recs,
                user_id=uid,
                book_tag_features_df=book_tag_features,
                user_tag_profile_df=user_tag_profile,
                to_read_df=to_read,
                **hybrid_weights,
            )
            
            # Get the top recommended book for each user
            if len(user_recs) > 0:
                top_rec = user_recs.iloc[0]
                svd_all_recommendations.append({
                    "user_id": uid,
                    "recommended_book": top_rec["title"],
                    "author": top_rec["authors"],
                    "score": round(top_rec["hybrid_score"], 3)
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
        n_factors=svd_factors
    )

    print_result("SVD Evaluation summary:", svd_results)

    print("\nRunning hybrid SVD evaluation (candidate_n=400)...")
    hybrid_svd_results = evaluate_model_hybrid_svd(
        ratings_df=ratings,
        books_df=books,
        book_tag_features_df=book_tag_features,
        to_read_df=to_read,
        k=5,
        test_size=0.2,
        random_state=42,
        max_users=100,
        min_test_rating=3.0,
        n_factors=svd_factors,
        candidate_n=400,
        **hybrid_weights,
    )

    print_result("Hybrid SVD Evaluation summary:", hybrid_svd_results)

    # ============================================
    # Model Comparison
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_results = pd.DataFrame(
        {
            "Model": [
                "KNN",
                "KNN + Hybrid rerank",
                f"SVD ({svd_factors} factors)",
                f"SVD + Hybrid rerank ({svd_factors} factors)",
            ],
            "Precision@5": [
                results["precision@k"],
                hybrid_knn_results["precision@k"],
                svd_results["precision@k"],
                hybrid_svd_results["precision@k"],
            ],
            "Recall@5": [
                results["recall@k"],
                hybrid_knn_results["recall@k"],
                svd_results["recall@k"],
                hybrid_svd_results["recall@k"],
            ],
            "Evaluated Users": [
                results["evaluated_users"],
                hybrid_knn_results["evaluated_users"],
                svd_results["evaluated_users"],
                hybrid_svd_results["evaluated_users"],
            ],
        }
    )
    
    print_result("Performance Comparison:", comparison_results)

if __name__ == "__main__":
    main()
