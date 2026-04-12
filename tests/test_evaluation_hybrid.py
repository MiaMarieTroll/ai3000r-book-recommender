"""
Tests for hybrid evaluation functions.
"""

import pandas as pd

from src.evaluation import evaluate_model_hybrid_knn, evaluate_model_hybrid_svd


def _build_small_ratings():
    rows = [
        {"user_id": 1, "book_id": 1, "rating": 5},
        {"user_id": 1, "book_id": 2, "rating": 4},
        {"user_id": 1, "book_id": 3, "rating": 2},
        {"user_id": 2, "book_id": 1, "rating": 4},
        {"user_id": 2, "book_id": 4, "rating": 5},
        {"user_id": 2, "book_id": 5, "rating": 3},
        {"user_id": 3, "book_id": 2, "rating": 5},
        {"user_id": 3, "book_id": 3, "rating": 4},
        {"user_id": 3, "book_id": 6, "rating": 2},
        {"user_id": 4, "book_id": 4, "rating": 4},
        {"user_id": 4, "book_id": 5, "rating": 5},
        {"user_id": 4, "book_id": 6, "rating": 3},
    ]
    return pd.DataFrame(rows)


def _build_tag_features():
    rows = [
        {"book_id": 1, "tag_name": "fantasy", "tag_weight": 1.0},
        {"book_id": 2, "tag_name": "fantasy", "tag_weight": 0.6},
        {"book_id": 2, "tag_name": "young-adult", "tag_weight": 0.4},
        {"book_id": 3, "tag_name": "classics", "tag_weight": 1.0},
        {"book_id": 4, "tag_name": "mystery", "tag_weight": 1.0},
        {"book_id": 5, "tag_name": "romance", "tag_weight": 1.0},
        {"book_id": 6, "tag_name": "science-fiction", "tag_weight": 1.0},
    ]
    return pd.DataFrame(rows)


def _build_to_read():
    rows = [
        {"user_id": 1, "book_id": 4},
        {"user_id": 2, "book_id": 2},
        {"user_id": 3, "book_id": 5},
        {"user_id": 4, "book_id": 1},
    ]
    return pd.DataFrame(rows)


def _assert_metric_result(result):
    assert set(result.keys()) == {"precision@k", "recall@k", "evaluated_users"}
    assert 0.0 <= result["precision@k"] <= 1.0
    assert 0.0 <= result["recall@k"] <= 1.0
    assert result["evaluated_users"] >= 0


def test_evaluate_model_hybrid_knn_returns_metrics():
    ratings = _build_small_ratings()
    tag_features = _build_tag_features()
    to_read = _build_to_read()

    result = evaluate_model_hybrid_knn(
        ratings_df=ratings,
        books_df=pd.DataFrame(),
        book_tag_features_df=tag_features,
        to_read_df=to_read,
        k=3,
        candidate_n=5,
        test_size=0.25,
        random_state=42,
        max_users=4,
        min_test_rating=3.0,
        progress_every=0,
    )

    _assert_metric_result(result)


def test_evaluate_model_hybrid_svd_returns_metrics():
    ratings = _build_small_ratings()
    tag_features = _build_tag_features()
    to_read = _build_to_read()

    result = evaluate_model_hybrid_svd(
        ratings_df=ratings,
        books_df=pd.DataFrame(),
        book_tag_features_df=tag_features,
        to_read_df=to_read,
        k=3,
        candidate_n=5,
        test_size=0.25,
        random_state=42,
        max_users=4,
        min_test_rating=3.0,
        progress_every=0,
        n_factors=2,
    )

    _assert_metric_result(result)
