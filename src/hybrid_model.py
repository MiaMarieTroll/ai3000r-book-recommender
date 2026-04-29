"""
hybrid_model.py

Hybrid reranking utilities for recommendation candidates.

This module blends collaborative scores with content affinity (tags)
and to-read intent to produce a final hybrid ranking.
"""

import pandas as pd


def _min_max_normalize(series):
    if series.empty:
        return series

    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def _safe_normalize_weights(collaborative_weight, content_weight, to_read_weight):
    weights = {
        "collaborative": max(float(collaborative_weight), 0.0),
        "content": max(float(content_weight), 0.0),
        "to_read": max(float(to_read_weight), 0.0),
    }

    total = sum(weights.values())

    if total <= 0:
        return {"collaborative": 1.0, "content": 0.0, "to_read": 0.0}

    return {name: value / total for name, value in weights.items()}


def compute_content_scores(
    user_id,
    candidate_book_ids,
    book_tag_features_df,
    user_tag_profile_df,
):
    """
    Compute content affinity score for each candidate book based on user tag profile.
    """
    user_profile = user_tag_profile_df[user_tag_profile_df["user_id"] == user_id]

    if user_profile.empty:
        return {book_id: 0.0 for book_id in candidate_book_ids}

    candidate_tags = book_tag_features_df[
        book_tag_features_df["book_id"].isin(candidate_book_ids)
    ]

    if candidate_tags.empty:
        return {book_id: 0.0 for book_id in candidate_book_ids}

    merged = candidate_tags.merge(
        user_profile[["tag_name", "tag_score"]],
        on="tag_name",
        how="left",
    )

    merged["tag_score"] = merged["tag_score"].fillna(0.0)
    merged["content_contribution"] = merged["tag_weight"] * merged["tag_score"]

    scores = (
        merged.groupby("book_id", as_index=True)["content_contribution"]
        .sum()
        .to_dict()
    )

    return {
        book_id: float(scores.get(book_id, 0.0))
        for book_id in candidate_book_ids
    }


def compute_to_read_boosts(user_id, candidate_book_ids, to_read_df):
    """
    Return binary boosts for candidates that appear in a user's to-read list.
    """
    user_to_read = set(
        to_read_df.loc[to_read_df["user_id"] == user_id, "book_id"].tolist()
    )

    return {
        book_id: 1.0 if book_id in user_to_read else 0.0
        for book_id in candidate_book_ids
    }


def rerank_recommendations_hybrid(
    recommendations_df,
    user_id,
    book_tag_features_df,
    user_tag_profile_df,
    to_read_df,
    collaborative_weight=0.7,
    content_weight=0.2,
    to_read_weight=0.1,
    min_content_matches=3,
):
    """
    Re-rank collaborative recommendations with content and to-read signals.
    """
    if recommendations_df.empty:
        return recommendations_df

    reranked = recommendations_df.copy()
    candidate_book_ids = reranked["book_id"].tolist()

    reranked["collaborative_score_norm"] = _min_max_normalize(
        reranked["score"].astype(float)
    )

    content_scores = compute_content_scores(
        user_id=user_id,
        candidate_book_ids=candidate_book_ids,
        book_tag_features_df=book_tag_features_df,
        user_tag_profile_df=user_tag_profile_df,
    )

    reranked["content_score"] = reranked["book_id"].map(content_scores).fillna(0.0)

    reranked["content_score_norm"] = _min_max_normalize(
        reranked["content_score"].astype(float)
    )

    to_read_scores = compute_to_read_boosts(
        user_id=user_id,
        candidate_book_ids=candidate_book_ids,
        to_read_df=to_read_df,
    )

    reranked["to_read_score"] = reranked["book_id"].map(to_read_scores).fillna(0.0)

    content_matches = int((reranked["content_score"] > 0).sum())
    content_signal_ok = content_matches >= min_content_matches

    base_weights = _safe_normalize_weights(
        collaborative_weight=collaborative_weight,
        content_weight=content_weight,
        to_read_weight=to_read_weight,
    )

    if not content_signal_ok:
        adaptive_weights = {
            "collaborative": 1.0,
            "content": 0.0,
            "to_read": 0.0,
        }
    else:
        coverage = content_matches / max(len(reranked), 1)
        content_boost = 0.5 + coverage

        adaptive_weights = _safe_normalize_weights(
            collaborative_weight=base_weights["collaborative"],
            content_weight=base_weights["content"] * content_boost,
            to_read_weight=base_weights["to_read"],
        )

    reranked["hybrid_score"] = (
        adaptive_weights["collaborative"] * reranked["collaborative_score_norm"]
        + adaptive_weights["content"] * reranked["content_score_norm"]
        + adaptive_weights["to_read"] * reranked["to_read_score"]
    )

    reranked["content_matches"] = content_matches

    reranked = reranked.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    reranked["rank"] = reranked.index + 1

    return reranked