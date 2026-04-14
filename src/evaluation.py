import numpy as np
import pandas as pd
import os
import sys

# Allow running this file directly: `python src/evaluation.py`.
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.preprocessing import create_user_item_matrix, fill_missing, build_user_tag_profile
from src.collaborative_model import build_knn_model
from src.rag.content_model import compute_content_scores, rerank_recommendations_hybrid


EMPTY_RESULTS = {
    "precision@k": 0.0,
    "recall@k": 0.0,
    "evaluated_users": 0,
}


def _empty_results():
    return dict(EMPTY_RESULTS)


def _split_train_test(ratings_df, test_size, random_state):
    shuffled = ratings_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    return train_df, test_df


def _relevant_items_by_user(test_df, min_test_rating):
    relevant_test_df = test_df.loc[test_df["rating"] >= min_test_rating, ["user_id", "book_id"]]
    if relevant_test_df.empty:
        return relevant_test_df, {}
    relevant_by_user = relevant_test_df.groupby("user_id")["book_id"].apply(set).to_dict()
    return relevant_test_df, relevant_by_user


def _common_users(relevant_by_user, user_item_matrix, max_users):
    common_users = list(set(relevant_by_user).intersection(user_item_matrix.index))
    if max_users is not None:
        common_users = common_users[:max_users]
    return common_users


def _finalize_metrics(precisions, recalls):
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    return {
        "precision@k": round(avg_precision, 4),
        "recall@k": round(avg_recall, 4),
        "evaluated_users": len(precisions),
    }


def _prepare_rerank_candidates(candidate_df):
    candidate_df = candidate_df.copy()
    candidate_df["title"] = ""
    candidate_df["authors"] = ""
    return candidate_df


def precision_at_k(recommended_items, relevant_items, k):
    if k <= 0:
        return 0.0
    recommended_top_k = recommended_items[:k]
    if len(recommended_top_k) == 0:
        return 0.0
    hits = sum(1 for item in recommended_top_k if item in relevant_items)
    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    if not relevant_items:
        return 0.0
    recommended_top_k = recommended_items[:k]
    hits = sum(1 for item in recommended_top_k if item in relevant_items)
    return hits / len(relevant_items)


def recommend_book_ids_fast_from_neighbors(
    user_pos,
    neighbor_positions,
    neighbor_distances,
    matrix_values,
    item_ids,
    n=5,
):
    """
    Fast recommendation using precomputed nearest-neighbor results.
    Returns only book_ids, no DataFrame, no metadata merge.
    """
    # Remove self if present in neighbor list.
    mask = neighbor_positions != user_pos
    neighbor_positions = neighbor_positions[mask]
    neighbor_distances = neighbor_distances[mask]

    if len(neighbor_positions) == 0:
        return []

    similarities = 1 / (1 + neighbor_distances)
    neighbor_matrix = matrix_values[neighbor_positions]

    weighted_scores = np.average(neighbor_matrix, axis=0, weights=similarities)

    already_rated = matrix_values[user_pos] > 0
    weighted_scores[already_rated] = -np.inf

    candidate_count = int(np.isfinite(weighted_scores).sum())
    if candidate_count == 0:
        return []

    n_select = min(n, candidate_count)
    top_idx = np.argpartition(weighted_scores, -n_select)[-n_select:]
    top_idx = top_idx[np.argsort(weighted_scores[top_idx])[::-1]]

    return item_ids[top_idx].tolist()


def recommend_candidates_fast_from_neighbors(
    user_pos,
    neighbor_positions,
    neighbor_distances,
    matrix_values,
    item_ids,
    n=30,
):
    """
    Fast candidate generation with scores from nearest-neighbor results.
    Returns DataFrame: [book_id, score].
    """
    # Remove self if present in neighbor list.
    mask = neighbor_positions != user_pos
    neighbor_positions = neighbor_positions[mask]
    neighbor_distances = neighbor_distances[mask]

    if len(neighbor_positions) == 0:
        return pd.DataFrame(columns=["book_id", "score"])

    similarities = 1 / (1 + neighbor_distances)
    neighbor_matrix = matrix_values[neighbor_positions]
    weighted_scores = np.average(neighbor_matrix, axis=0, weights=similarities)

    already_rated = matrix_values[user_pos] > 0
    weighted_scores[already_rated] = -np.inf

    finite_mask = np.isfinite(weighted_scores)
    candidate_count = int(finite_mask.sum())
    if candidate_count == 0:
        return pd.DataFrame(columns=["book_id", "score"])

    n_select = min(n, candidate_count)
    top_idx = np.argpartition(weighted_scores, -n_select)[-n_select:]
    top_idx = top_idx[np.argsort(weighted_scores[top_idx])[::-1]]

    return pd.DataFrame(
        {
            "book_id": item_ids[top_idx],
            "score": weighted_scores[top_idx],
        }
    ).reset_index(drop=True)


def recommend_candidates_with_content_boost(
    user_id,
    similarities,
    item_ids,
    already_rated_mask,
    book_tag_features_df,
    user_tag_profile_df,
    candidate_n=100,
    content_candidate_n=None,
):
    """
    Build an expanded candidate pool for SVD reranking.

    The pool combines top SVD-scored items with top content-matched items so
    the hybrid reranker has candidates with non-zero tag affinity.
    """
    if content_candidate_n is None:
        content_candidate_n = candidate_n

    finite_mask = np.isfinite(similarities)
    candidate_count = int(finite_mask.sum())
    if candidate_count == 0:
        return pd.DataFrame(columns=["book_id", "score"])

    n_select = min(max(candidate_n, 1), candidate_count)
    top_idx = np.argpartition(similarities, -n_select)[-n_select:]
    top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]

    svd_candidates = pd.DataFrame(
        {
            "book_id": item_ids[top_idx],
            "score": similarities[top_idx],
        }
    )

    unrated_item_ids = item_ids[~already_rated_mask]
    content_scores = compute_content_scores(
        user_id=user_id,
        candidate_book_ids=unrated_item_ids.tolist(),
        book_tag_features_df=book_tag_features_df,
        user_tag_profile_df=user_tag_profile_df,
    )

    content_df = pd.DataFrame(
        {
            "book_id": list(content_scores.keys()),
            "score": list(content_scores.values()),
        }
    )
    content_df = content_df[content_df["score"] > 0].sort_values("score", ascending=False)
    content_df = content_df.head(min(content_candidate_n, len(content_df)))

    combined = pd.concat([svd_candidates, content_df], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["book_id", "score"])

    combined = combined.groupby("book_id", as_index=False)["score"].max()
    return combined.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_model(
    ratings_df=None,
    books_df=None,
    k=5,
    test_size=0.2,
    random_state=42,
    max_users=None,
    min_test_rating=3.0,
    progress_every=50,
):
    """
    Evaluate collaborative filtering model.
    
    Parameters
    ----------
    k : int
        k for evaluation (precision@k, recall@k). Default 5.
    min_test_rating : float
        Minimum rating to consider as "relevant" in test set. Default 3.0 (lowered from 4).
    """
    if ratings_df is None:
        print("Evaluation skipped: ratings_df must be provided.")
        return _empty_results()

    required_cols = {"user_id", "book_id", "rating"}
    missing_cols = required_cols - set(ratings_df.columns)
    if missing_cols:
        raise ValueError(f"ratings_df is missing required columns: {sorted(missing_cols)}")

    if ratings_df.empty:
        print("Evaluation skipped: ratings_df is empty.")
        return _empty_results()

    train_df, test_df = _split_train_test(ratings_df, test_size, random_state)

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.1f}%")
    print(f"Missing values after fill: {int(user_item_matrix.isna().sum().sum())}")

    model = build_knn_model(user_item_matrix)

    relevant_test_df, relevant_by_user = _relevant_items_by_user(test_df, min_test_rating)

    if relevant_test_df.empty:
        print(f"Evaluation skipped: no relevant test items found (min_rating={min_test_rating}).")
        return _empty_results()

    print(f"Test items (rating >= {min_test_rating}): {len(relevant_test_df)}")
    print(f"Users with test items: {len(relevant_by_user)}")

    common_users = _common_users(relevant_by_user, user_item_matrix, max_users)

    print(f"Evaluating {len(common_users)} users (k={k})...")

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    matrix_values = user_item_matrix.to_numpy(dtype=np.float32, copy=False)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}

    eval_positions = np.array([user_to_pos[user_id] for user_id in common_users], dtype=np.int64)

    # Batch nearest-neighbor lookup for all evaluated users.
    neighbor_count = min(6, len(user_item_matrix))
    distances_batch, indices_batch = model.kneighbors(
        matrix_values[eval_positions],
        n_neighbors=neighbor_count,
    )

    precisions = []
    recalls = []

    for i, user_id in enumerate(common_users, start=1):
        user_pos = eval_positions[i - 1]
        relevant_items = relevant_by_user[user_id]

        try:
            recommended_items = recommend_book_ids_fast_from_neighbors(
                user_pos=user_pos,
                neighbor_positions=indices_batch[i - 1],
                neighbor_distances=distances_batch[i - 1],
                matrix_values=matrix_values,
                item_ids=item_ids,
                n=k,
            )
        except Exception:
            continue

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(common_users)} users...")

    results = _finalize_metrics(precisions, recalls)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Precision@{k}: {results['precision@k']}")
    print(f"Recall@{k}: {results['recall@k']}")
    print(f"Evaluated users: {results['evaluated_users']}")

    return results


def evaluate_model_svd(
    ratings_df=None,
    books_df=None,
    k=5,
    test_size=0.2,
    random_state=42,
    max_users=None,
    min_test_rating=3.0,
    progress_every=50,
    n_factors=50,
):
    """
    Evaluate matrix factorization (SVD) model.
    
    Parameters
    ----------
    k : int
        k for evaluation (precision@k, recall@k). Default 5.
    min_test_rating : float
        Minimum rating to consider as "relevant" in test set. Default 3.0.
    n_factors : int
        Number of latent factors for SVD. Default 50.
    """
    if ratings_df is None:
        print("Evaluation skipped: ratings_df must be provided.")
        return _empty_results()

    required_cols = {"user_id", "book_id", "rating"}
    missing_cols = required_cols - set(ratings_df.columns)
    if missing_cols:
        raise ValueError(f"ratings_df is missing required columns: {sorted(missing_cols)}")

    if ratings_df.empty:
        print("Evaluation skipped: ratings_df is empty.")
        return _empty_results()

    train_df, test_df = _split_train_test(ratings_df, test_size, random_state)

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Missing values after fill: {int(user_item_matrix.isna().sum().sum())}")

    # Build SVD model
    from src.matrix_factorization_model import build_svd_model
    svd_model = build_svd_model(user_item_matrix, n_factors=n_factors)

    relevant_test_df, relevant_by_user = _relevant_items_by_user(test_df, min_test_rating)

    if relevant_test_df.empty:
        print(f"Evaluation skipped: no relevant test items found (min_rating={min_test_rating}).")
        return _empty_results()

    print(f"Test items (rating >= {min_test_rating}): {len(relevant_test_df)}")
    print(f"Users with test items: {len(relevant_by_user)}")

    common_users = _common_users(relevant_by_user, user_item_matrix, max_users)

    print(f"Evaluating {len(common_users)} users (k={k}, n_factors={n_factors})...")

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    matrix_values_dense = user_item_matrix.to_numpy(dtype=np.float64)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}

    precisions = []
    recalls = []

    for i, user_id in enumerate(common_users, start=1):
        user_pos = user_to_pos[user_id]
        relevant_items = relevant_by_user[user_id]

        try:
            # Get user latent vector
            user_vector = matrix_values_dense[user_pos]
            user_latent = svd_model.transform(user_vector.reshape(1, -1))[0]

            # Get item latent vectors
            item_latents = svd_model.components_.T

            # Compute cosine similarities
            similarities = item_latents @ user_latent / (
                np.linalg.norm(item_latents, axis=1) * np.linalg.norm(user_latent) + 1e-9
            )

            # Remove already-rated items
            already_rated = user_vector > 0
            similarities[already_rated] = -np.inf

            # Get top k
            top_idx = np.argsort(similarities)[-k:][::-1]
            top_idx = top_idx[np.isfinite(similarities[top_idx])]

            recommended_items = item_ids[top_idx].tolist() if len(top_idx) > 0 else []

        except Exception:
            recommended_items = []

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(common_users)} users...")

    results = _finalize_metrics(precisions, recalls)

    print("\nEvaluation Results (SVD)")
    print("------------------------")
    print(f"Precision@{k}: {results['precision@k']}")
    print(f"Recall@{k}: {results['recall@k']}")
    print(f"Evaluated users: {results['evaluated_users']}")

    return results


def evaluate_model_hybrid_knn(
    ratings_df=None,
    books_df=None,
    book_tag_features_df=None,
    to_read_df=None,
    k=5,
    candidate_n=30,
    test_size=0.2,
    random_state=42,
    max_users=None,
    min_test_rating=3.0,
    progress_every=50,
    collaborative_weight=0.7,
    content_weight=0.2,
    to_read_weight=0.1,
):
    """
    Evaluate KNN + hybrid reranking (tags + to-read).
    """
    if book_tag_features_df is None or to_read_df is None:
        raise ValueError("Hybrid KNN evaluation requires book_tag_features_df and to_read_df.")

    train_df, test_df = _split_train_test(ratings_df, test_size, random_state)

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)
    model = build_knn_model(user_item_matrix)

    user_tag_profile_df = build_user_tag_profile(
        ratings_df=train_df,
        to_read_df=to_read_df,
        book_tag_features_df=book_tag_features_df,
        to_read_weight=to_read_weight,
    )

    relevant_test_df, relevant_by_user = _relevant_items_by_user(test_df, min_test_rating)
    if relevant_test_df.empty:
        return _empty_results()

    common_users = _common_users(relevant_by_user, user_item_matrix, max_users)

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    matrix_values = user_item_matrix.to_numpy(dtype=np.float32, copy=False)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}
    eval_positions = np.array([user_to_pos[user_id] for user_id in common_users], dtype=np.int64)

    # Use a wider neighbor set so reranking receives a richer candidate pool.
    neighbor_count = min(max(20, candidate_n // 2) + 1, len(user_item_matrix))
    distances_batch, indices_batch = model.kneighbors(
        matrix_values[eval_positions],
        n_neighbors=neighbor_count,
    )

    precisions = []
    recalls = []

    for i, user_id in enumerate(common_users, start=1):
        user_pos = eval_positions[i - 1]
        relevant_items = relevant_by_user[user_id]

        candidate_df = recommend_candidates_fast_from_neighbors(
            user_pos=user_pos,
            neighbor_positions=indices_batch[i - 1],
            neighbor_distances=distances_batch[i - 1],
            matrix_values=matrix_values,
            item_ids=item_ids,
            n=max(k, candidate_n),
        )

        if candidate_df.empty:
            recommended_items = []
        else:
            # Minimal schema expected by reranker.
            candidate_df = _prepare_rerank_candidates(candidate_df)
            reranked_df = rerank_recommendations_hybrid(
                recommendations_df=candidate_df,
                user_id=user_id,
                book_tag_features_df=book_tag_features_df,
                user_tag_profile_df=user_tag_profile_df,
                to_read_df=to_read_df,
                collaborative_weight=collaborative_weight,
                content_weight=content_weight,
                to_read_weight=to_read_weight,
            )
            recommended_items = reranked_df["book_id"].head(k).tolist()

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(common_users)} users (hybrid KNN)...")

    return _finalize_metrics(precisions, recalls)


def evaluate_model_hybrid_svd(
    ratings_df=None,
    books_df=None,
    book_tag_features_df=None,
    to_read_df=None,
    k=5,
    candidate_n=30,
    content_candidate_n=None,
    test_size=0.2,
    random_state=42,
    max_users=None,
    min_test_rating=3.0,
    progress_every=50,
    n_factors=50,
    collaborative_weight=0.7,
    content_weight=0.2,
    to_read_weight=0.1,
):
    """
    Evaluate SVD + hybrid reranking (tags + to-read).
    """
    if book_tag_features_df is None or to_read_df is None:
        raise ValueError("Hybrid SVD evaluation requires book_tag_features_df and to_read_df.")

    train_df, test_df = _split_train_test(ratings_df, test_size, random_state)

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)

    from src.matrix_factorization_model import build_svd_model
    svd_model = build_svd_model(user_item_matrix, n_factors=n_factors)
    knn_model = build_knn_model(user_item_matrix)

    user_tag_profile_df = build_user_tag_profile(
        ratings_df=train_df,
        to_read_df=to_read_df,
        book_tag_features_df=book_tag_features_df,
        to_read_weight=to_read_weight,
    )

    relevant_test_df, relevant_by_user = _relevant_items_by_user(test_df, min_test_rating)
    if relevant_test_df.empty:
        return _empty_results()

    common_users = _common_users(relevant_by_user, user_item_matrix, max_users)

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    matrix_values_dense = user_item_matrix.to_numpy(dtype=np.float64)
    matrix_values = user_item_matrix.to_numpy(dtype=np.float32, copy=False)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}
    eval_positions = np.array([user_to_pos[user_id] for user_id in common_users], dtype=np.int64)

    neighbor_count = min(max(20, candidate_n // 2) + 1, len(user_item_matrix))
    distances_batch, indices_batch = knn_model.kneighbors(
        matrix_values[eval_positions],
        n_neighbors=neighbor_count,
    )

    precisions = []
    recalls = []

    for i, user_id in enumerate(common_users, start=1):
        user_pos = eval_positions[i - 1]
        relevant_items = relevant_by_user[user_id]

        user_vector = matrix_values_dense[user_pos]
        user_latent = svd_model.transform(user_vector.reshape(1, -1))[0]
        item_latents = svd_model.components_.T

        similarities = item_latents @ user_latent / (
            np.linalg.norm(item_latents, axis=1) * np.linalg.norm(user_latent) + 1e-9
        )

        already_rated = user_vector > 0
        similarities[already_rated] = -np.inf

        finite_mask = np.isfinite(similarities)
        candidate_count = int(finite_mask.sum())

        if candidate_count == 0:
            recommended_items = []
        else:
            svd_candidate_df = recommend_candidates_with_content_boost(
                user_id=user_id,
                similarities=similarities,
                item_ids=item_ids,
                already_rated_mask=already_rated,
                book_tag_features_df=book_tag_features_df,
                user_tag_profile_df=user_tag_profile_df,
                candidate_n=max(k, candidate_n),
                content_candidate_n=content_candidate_n,
            )

            knn_candidate_df = recommend_candidates_fast_from_neighbors(
                user_pos=user_pos,
                neighbor_positions=indices_batch[i - 1],
                neighbor_distances=distances_batch[i - 1],
                matrix_values=matrix_values,
                item_ids=item_ids,
                n=max(k, candidate_n),
            )

            candidate_df = pd.concat([svd_candidate_df, knn_candidate_df], ignore_index=True)

            if not candidate_df.empty:
                candidate_df = candidate_df.groupby("book_id", as_index=False)["score"].max()

            if candidate_df.empty:
                recommended_items = []
            else:
                # Minimal schema expected by reranker.
                candidate_df = _prepare_rerank_candidates(candidate_df)
                reranked_df = rerank_recommendations_hybrid(
                    recommendations_df=candidate_df,
                    user_id=user_id,
                    book_tag_features_df=book_tag_features_df,
                    user_tag_profile_df=user_tag_profile_df,
                    to_read_df=to_read_df,
                    collaborative_weight=collaborative_weight,
                    content_weight=content_weight,
                    to_read_weight=to_read_weight,
                )
                recommended_items = reranked_df["book_id"].head(k).tolist()
        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(common_users)} users (hybrid SVD)...")

    return _finalize_metrics(precisions, recalls)


def evaluate_model_hybrid_knn_adaptive(
    ratings_df=None,
    books_df=None,
    book_tag_features_df=None,
    to_read_df=None,
    k=5,
    candidate_n=30,
    test_size=0.2,
    random_state=42,
    max_users=None,
    min_test_rating=3.0,
    progress_every=50,
    cold_user_max_interactions=10,
    cold_weights=None,
    warm_weights=None,
):
    """
    Evaluate KNN + hybrid reranking with adaptive weights based on user history.

    Cold users (few train interactions) can use a different blend than warm users.
    """
    if book_tag_features_df is None or to_read_df is None:
        raise ValueError("Adaptive hybrid KNN evaluation requires book_tag_features_df and to_read_df.")

    if cold_weights is None:
        cold_weights = {
            "collaborative_weight": 0.45,
            "content_weight": 0.40,
            "to_read_weight": 0.15,
        }
    if warm_weights is None:
        warm_weights = {
            "collaborative_weight": 0.65,
            "content_weight": 0.25,
            "to_read_weight": 0.10,
        }

    train_df, test_df = _split_train_test(ratings_df, test_size, random_state)

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)
    model = build_knn_model(user_item_matrix)

    train_interactions = train_df.groupby("user_id")["book_id"].count().to_dict()

    max_to_read_weight = max(
        float(cold_weights.get("to_read_weight", 0.0)),
        float(warm_weights.get("to_read_weight", 0.0)),
    )

    user_tag_profile_df = build_user_tag_profile(
        ratings_df=train_df,
        to_read_df=to_read_df,
        book_tag_features_df=book_tag_features_df,
        to_read_weight=max_to_read_weight,
    )

    relevant_test_df, relevant_by_user = _relevant_items_by_user(test_df, min_test_rating)
    if relevant_test_df.empty:
        return _empty_results()

    common_users = _common_users(relevant_by_user, user_item_matrix, max_users)

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    matrix_values = user_item_matrix.to_numpy(dtype=np.float32, copy=False)
    user_to_pos = {uid: i for i, uid in enumerate(user_ids)}
    eval_positions = np.array([user_to_pos[user_id] for user_id in common_users], dtype=np.int64)

    neighbor_count = min(max(20, candidate_n // 2) + 1, len(user_item_matrix))
    distances_batch, indices_batch = model.kneighbors(
        matrix_values[eval_positions],
        n_neighbors=neighbor_count,
    )

    precisions = []
    recalls = []

    for i, user_id in enumerate(common_users, start=1):
        user_pos = eval_positions[i - 1]
        relevant_items = relevant_by_user[user_id]

        candidate_df = recommend_candidates_fast_from_neighbors(
            user_pos=user_pos,
            neighbor_positions=indices_batch[i - 1],
            neighbor_distances=distances_batch[i - 1],
            matrix_values=matrix_values,
            item_ids=item_ids,
            n=max(k, candidate_n),
        )

        if candidate_df.empty:
            recommended_items = []
        else:
            interaction_count = int(train_interactions.get(user_id, 0))
            selected_weights = (
                cold_weights
                if interaction_count <= cold_user_max_interactions
                else warm_weights
            )

            candidate_df = _prepare_rerank_candidates(candidate_df)
            reranked_df = rerank_recommendations_hybrid(
                recommendations_df=candidate_df,
                user_id=user_id,
                book_tag_features_df=book_tag_features_df,
                user_tag_profile_df=user_tag_profile_df,
                to_read_df=to_read_df,
                collaborative_weight=float(selected_weights.get("collaborative_weight", 0.7)),
                content_weight=float(selected_weights.get("content_weight", 0.2)),
                to_read_weight=float(selected_weights.get("to_read_weight", 0.1)),
            )
            recommended_items = reranked_df["book_id"].head(k).tolist()

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))

        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(common_users)} users (adaptive hybrid KNN)...")

    return _finalize_metrics(precisions, recalls)