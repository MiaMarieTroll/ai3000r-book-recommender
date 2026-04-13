"""
Run a practical 3-phase tuning sweep for hybrid reranking.

Outputs CSV files under results/:
- run1_candidate_sweep.csv
- run2_weight_sweep.csv
- run3_adaptive_policy_sweep.csv
"""

import os
import pandas as pd

from src.data_loader import load_books, load_ratings, load_book_tags, load_tags, load_to_read
from src.preprocessing import clean_ratings, build_book_tag_features
from src.evaluation import evaluate_model_hybrid_knn, evaluate_model_hybrid_knn_adaptive


def _to_row(config_name, params, metrics):
    row = {"config": config_name}
    row.update(params)
    row.update(metrics)
    return row


def run_three_phase_tuning(
    data_dir="data",
    k=5,
    test_size=0.2,
    random_state=42,
    max_users=200,
    min_test_rating=3.0,
):
    print("Loading data...")
    books = load_books(os.path.join(data_dir, "books.csv"))
    ratings = load_ratings(os.path.join(data_dir, "ratings.csv"))
    book_tags = load_book_tags(os.path.join(data_dir, "book_tags.csv"))
    tags = load_tags(os.path.join(data_dir, "tags.csv"))
    to_read = load_to_read(os.path.join(data_dir, "to_read.csv"))

    ratings = clean_ratings(ratings)
    book_tag_features = build_book_tag_features(
        book_tags,
        tags,
        min_count=5,
        max_tags_per_book=10,
    )

    os.makedirs("results", exist_ok=True)

    # Run 1: candidate pool sweep (recall-focused).
    run1_rows = []
    candidate_values = [100, 200, 400]
    for candidate_n in candidate_values:
        result = evaluate_model_hybrid_knn(
            ratings_df=ratings,
            books_df=books,
            book_tag_features_df=book_tag_features,
            to_read_df=to_read,
            k=k,
            candidate_n=candidate_n,
            test_size=test_size,
            random_state=random_state,
            max_users=max_users,
            min_test_rating=min_test_rating,
            progress_every=0,
            collaborative_weight=0.55,
            content_weight=0.30,
            to_read_weight=0.15,
        )
        run1_rows.append(
            _to_row(
                "candidate_sweep",
                {"candidate_n": candidate_n},
                result,
            )
        )

    run1_df = pd.DataFrame(run1_rows).sort_values(
        by=["recall@k", "precision@k"],
        ascending=[False, False],
    )
    run1_path = os.path.join("results", "run1_candidate_sweep.csv")
    run1_df.to_csv(run1_path, index=False)
    best_candidate_n = int(run1_df.iloc[0]["candidate_n"])

    # Run 2: fusion weight sweep (precision-focused with recall tie-break).
    run2_rows = []
    collaborative_grid = [0.4, 0.5, 0.6]
    content_grid = [0.2, 0.3, 0.4]
    to_read_grid = [0.1, 0.2]

    for cw in collaborative_grid:
        for tw in content_grid:
            for rw in to_read_grid:
                total = round(cw + tw + rw, 10)
                if abs(total - 1.0) > 1e-9:
                    continue

                result = evaluate_model_hybrid_knn(
                    ratings_df=ratings,
                    books_df=books,
                    book_tag_features_df=book_tag_features,
                    to_read_df=to_read,
                    k=k,
                    candidate_n=best_candidate_n,
                    test_size=test_size,
                    random_state=random_state,
                    max_users=max_users,
                    min_test_rating=min_test_rating,
                    progress_every=0,
                    collaborative_weight=cw,
                    content_weight=tw,
                    to_read_weight=rw,
                )
                run2_rows.append(
                    _to_row(
                        "weight_sweep",
                        {
                            "candidate_n": best_candidate_n,
                            "collaborative_weight": cw,
                            "content_weight": tw,
                            "to_read_weight": rw,
                        },
                        result,
                    )
                )

    run2_df = pd.DataFrame(run2_rows).sort_values(
        by=["precision@k", "recall@k"],
        ascending=[False, False],
    )
    run2_path = os.path.join("results", "run2_weight_sweep.csv")
    run2_df.to_csv(run2_path, index=False)

    best_w = run2_df.iloc[0]
    best_weights = {
        "collaborative_weight": float(best_w["collaborative_weight"]),
        "content_weight": float(best_w["content_weight"]),
        "to_read_weight": float(best_w["to_read_weight"]),
    }

    # Run 3: adaptive cold/warm rerank policy sweep.
    run3_rows = []
    cold_threshold_values = [5, 10, 20]

    warm_weights = best_weights
    cold_candidates = [
        {"collaborative_weight": 0.45, "content_weight": 0.40, "to_read_weight": 0.15},
        {"collaborative_weight": 0.40, "content_weight": 0.45, "to_read_weight": 0.15},
    ]

    for cold_threshold in cold_threshold_values:
        for cold_w in cold_candidates:
            result = evaluate_model_hybrid_knn_adaptive(
                ratings_df=ratings,
                books_df=books,
                book_tag_features_df=book_tag_features,
                to_read_df=to_read,
                k=k,
                candidate_n=best_candidate_n,
                test_size=test_size,
                random_state=random_state,
                max_users=max_users,
                min_test_rating=min_test_rating,
                progress_every=0,
                cold_user_max_interactions=cold_threshold,
                cold_weights=cold_w,
                warm_weights=warm_weights,
            )
            run3_rows.append(
                _to_row(
                    "adaptive_policy_sweep",
                    {
                        "candidate_n": best_candidate_n,
                        "cold_user_max_interactions": cold_threshold,
                        "cold_collaborative_weight": cold_w["collaborative_weight"],
                        "cold_content_weight": cold_w["content_weight"],
                        "cold_to_read_weight": cold_w["to_read_weight"],
                        "warm_collaborative_weight": warm_weights["collaborative_weight"],
                        "warm_content_weight": warm_weights["content_weight"],
                        "warm_to_read_weight": warm_weights["to_read_weight"],
                    },
                    result,
                )
            )

    run3_df = pd.DataFrame(run3_rows).sort_values(
        by=["precision@k", "recall@k"],
        ascending=[False, False],
    )
    run3_path = os.path.join("results", "run3_adaptive_policy_sweep.csv")
    run3_df.to_csv(run3_path, index=False)

    print("\nSaved tuning results:")
    print(f"- {run1_path}")
    print(f"- {run2_path}")
    print(f"- {run3_path}")

    print("\nBest configs:")
    print("Run 1 best:")
    print(run1_df.head(1).to_string(index=False))
    print("\nRun 2 best:")
    print(run2_df.head(1).to_string(index=False))
    print("\nRun 3 best:")
    print(run3_df.head(1).to_string(index=False))


if __name__ == "__main__":
    run_three_phase_tuning()
