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
    min_test_rating=4,
    progress_every=50,
):
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

    model = build_knn_model(user_item_matrix)

    relevant_test_df = test_df.loc[test_df["rating"] >= min_test_rating, ["user_id", "book_id"]]

    if relevant_test_df.empty:
        print("Evaluation skipped: no relevant test items found.")
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "evaluated_users": 0,
        }

    relevant_by_user = relevant_test_df.groupby("user_id")["book_id"].apply(set).to_dict()

    common_users = list(set(relevant_by_user).intersection(user_item_matrix.index))
    if max_users is not None:
        common_users = common_users[:max_users]

    print(f"Evaluating {len(common_users)} users (k={k})...")

    user_ids = user_item_matrix.index.to_numpy()
    item_ids = user_item_matrix.columns.to_numpy()
    # float32 reduces memory bandwidth pressure during repeated scoring.
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