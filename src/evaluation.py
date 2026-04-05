import numpy as np
import os
import sys

# Allow running this file directly: `python src/evaluation.py`.
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.preprocessing import create_user_item_matrix, fill_missing
from src.collaborative_model import build_knn_model


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
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    required_cols = {"user_id", "book_id", "rating"}
    missing_cols = required_cols - set(ratings_df.columns)
    if missing_cols:
        raise ValueError(f"ratings_df is missing required columns: {sorted(missing_cols)}")

    if ratings_df.empty:
        print("Evaluation skipped: ratings_df is empty.")
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    shuffled = ratings_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))

    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.1f}%")
    print(f"Missing values after fill: {int(user_item_matrix.isna().sum().sum())}")

    model = build_knn_model(user_item_matrix)

    relevant_test_df = test_df.loc[test_df["rating"] >= min_test_rating, ["user_id", "book_id"]]

    if relevant_test_df.empty:
        print(f"Evaluation skipped: no relevant test items found (min_rating={min_test_rating}).")
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    relevant_by_user = relevant_test_df.groupby("user_id")["book_id"].apply(set).to_dict()

    print(f"Test items (rating >= {min_test_rating}): {len(relevant_test_df)}")
    print(f"Users with test items: {len(relevant_by_user)}")

    common_users = list(set(relevant_by_user).intersection(user_item_matrix.index))
    if max_users is not None:
        common_users = common_users[:max_users]

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

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    results = {
        "precision@k": round(avg_precision, 4),
        "recall@k": round(avg_recall, 4),
        "evaluated_users": len(precisions),
    }

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
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    required_cols = {"user_id", "book_id", "rating"}
    missing_cols = required_cols - set(ratings_df.columns)
    if missing_cols:
        raise ValueError(f"ratings_df is missing required columns: {sorted(missing_cols)}")

    if ratings_df.empty:
        print("Evaluation skipped: ratings_df is empty.")
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    shuffled = ratings_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))

    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()

    user_item_matrix = create_user_item_matrix(train_df)
    user_item_matrix = fill_missing(user_item_matrix)

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Missing values after fill: {int(user_item_matrix.isna().sum().sum())}")

    # Build SVD model
    from src.matrix_factorization_model import build_svd_model
    svd_model = build_svd_model(user_item_matrix, n_factors=n_factors)

    relevant_test_df = test_df.loc[test_df["rating"] >= min_test_rating, ["user_id", "book_id"]]

    if relevant_test_df.empty:
        print(f"Evaluation skipped: no relevant test items found (min_rating={min_test_rating}).")
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    relevant_by_user = relevant_test_df.groupby("user_id")["book_id"].apply(set).to_dict()

    print(f"Test items (rating >= {min_test_rating}): {len(relevant_test_df)}")
    print(f"Users with test items: {len(relevant_by_user)}")

    common_users = list(set(relevant_by_user).intersection(user_item_matrix.index))
    if max_users is not None:
        common_users = common_users[:max_users]

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

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    results = {
        "precision@k": round(avg_precision, 4),
        "recall@k": round(avg_recall, 4),
        "evaluated_users": len(precisions),
    }

    print("\nEvaluation Results (SVD)")
    print("------------------------")
    print(f"Precision@{k}: {results['precision@k']}")
    print(f"Recall@{k}: {results['recall@k']}")
    print(f"Evaluated users: {results['evaluated_users']}")

    return results