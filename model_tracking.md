### Experiment Overview

This file tracks and compares different machine learning approaches for the book recommendation system and evaluates their effectiveness. At this stage, we compare KNN and SVD models along with enhancements such as hybrid reranking.

The goal of this document is to maintain a record of experimental results while selecting only the best-performing model for the final system.

The recommender system development began with a simple popularity-based baseline model, which recommends the highest-rated books overall. This serves as a reference point before evaluating personalized recommendation methods.

After establishing the baseline, we evaluated personalized machine learning models to improve recommendation quality.

| Model | Parameters  | Precision@5 | Recall@5 | Notes                    |
| ----- | ----------- | ----------- | -------- | ------------------------ |
| KNN   | k = 5       | 0.044       | 0.0665   | First personalized model |
| SVD   | 50 factors  | 0.036       | 0.0559   | Underperformed vs KNN    |
| SVD   | 120 factors | 0.052       | 0.0967   | Best collaborative model |

---

### Hybrid Reranking

A hybrid reranking strategy was introduced to enhance recommendation quality by combining collaborative filtering outputs with content-based signals. Specifically, book **tags** and **book_tags** were used to refine recommendation rankings.

#### Initial Hybrid Results (Candidate Set = 100)

| Model                   | Precision@5 | Recall@5 | Evaluated Users | Notes                   |
| ----------------------- | ----------- | -------- | --------------- | ----------------------- |
| KNN                     | 0.044       | 0.0665   | 100             | Baseline KNN            |
| KNN + Hybrid rerank     | 0.084       | 0.1573   | 100             | Significant improvement |
| SVD (120 factors)       | 0.052       | 0.0967   | 100             | Base SVD                |
| SVD + Hybrid rerank     | 0.064       | 0.1250   | 100             | Moderate improvement    |
| SVD tuned (120 factors) | 0.052       | 0.0967   | 100             | No improvement          |

The hybrid reranking approach significantly improved performance, particularly for KNN, where Precision@5 nearly doubled compared to the base model.

---

### Performance Comparison (Candidate Set: 100 → 150)

| Model                             | Precision@5 | Recall@5 | Evaluated Users |
| --------------------------------- | ----------- | -------- | --------------- |
| KNN                               | 0.044       | 0.0665   | 100             |
| KNN + Hybrid rerank               | 0.086       | 0.1662   | 100             |
| SVD (120 factors)                 | 0.052       | 0.0967   | 100             |
| SVD + Hybrid rerank (120 factors) | 0.064       | 0.1312   | 100             |
| SVD tuned (120 factors)           | 0.052       | 0.0967   | 100             |

Increasing the candidate pool from **100 to 150** led to slight improvements, especially for the KNN-based hybrid model.

---

### Final Selection After 3-Phase Sweep (max_users = 100)

Testing was conducted using **KNN + Hybrid Reranking**.

| Run                     | Selected Configuration                                          | Precision@5 | Recall@5 | Decision            |
| ----------------------- | --------------------------------------------------------------- | ----------- | -------- | ------------------- |
| Run 1 (candidate sweep) | `candidate_n = 400`                                             | 0.092       | 0.1870   | Best candidate pool |
| Run 2 (weight sweep)    | `collaborative = 0.6`, `content = 0.3`, `to_read = 0.1`         | 0.092       | 0.1870   | Best static blend   |
| Run 3 (adaptive policy) | Cold: `0.45 / 0.4 / 0.15`, Warm: `0.6 / 0.3 / 0.1`, threshold=5 | 0.092       | 0.1870   | No improvement      |

---

### Final Configuration

```python
candidate_n = 400
collaborative_weight = 0.6
content_weight = 0.3
to_read_weight = 0.1
```

## Final Model

We selected **KNN + Hybrid Reranking** as our final model due to superior performance.

## Other Models Explored

- SVD (baseline)
- SVD + Hybrid
- Tuned SVD

These models were evaluated but not selected.
