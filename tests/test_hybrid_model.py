"""
Tests for hybrid reranking utilities.
"""

import pandas as pd

from src.hybrid_model import rerank_recommendations_hybrid


def test_rerank_recommendations_hybrid_returns_expected_columns():
    recommendations = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "book_id": [10, 20, 30],
            "title": ["A", "B", "C"],
            "authors": ["X", "Y", "Z"],
            "score": [0.2, 0.6, 0.4],
        }
    )

    book_tag_features = pd.DataFrame(
        {
            "book_id": [10, 20, 30],
            "tag_name": ["fantasy", "romance", "fantasy"],
            "tag_weight": [1.0, 1.0, 1.0],
        }
    )

    user_tag_profile = pd.DataFrame(
        {
            "user_id": [1, 1],
            "tag_name": ["fantasy", "romance"],
            "tag_score": [0.8, 0.2],
        }
    )

    to_read = pd.DataFrame(
        {
            "user_id": [1],
            "book_id": [30],
        }
    )

    reranked = rerank_recommendations_hybrid(
        recommendations_df=recommendations,
        user_id=1,
        book_tag_features_df=book_tag_features,
        user_tag_profile_df=user_tag_profile,
        to_read_df=to_read,
    )

    assert len(reranked) == 3
    assert "hybrid_score" in reranked.columns
    assert "content_score" in reranked.columns
    assert "to_read_score" in reranked.columns
    assert reranked["rank"].tolist() == [1, 2, 3]


def test_rerank_recommendations_hybrid_empty_input():
    empty_recs = pd.DataFrame(columns=["rank", "book_id", "title", "authors", "score"])
    result = rerank_recommendations_hybrid(
        recommendations_df=empty_recs,
        user_id=1,
        book_tag_features_df=pd.DataFrame(columns=["book_id", "tag_name", "tag_weight"]),
        user_tag_profile_df=pd.DataFrame(columns=["user_id", "tag_name", "tag_score"]),
        to_read_df=pd.DataFrame(columns=["user_id", "book_id"]),
    )

    assert result.empty

