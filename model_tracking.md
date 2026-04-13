### Experiment Overview

This file tracks and compares different machine learning approaches for the book recommendation system and evaluates their effectiveness. We compare KNN and SVD models along with enhancements such as hybrid reranking.

The goal is to document experimental results while selecting the best-performing model for the final system.

The recommender system development began with a simple popularity-based baseline model, which recommends the highest-rated books overall. This serves as a reference before evaluating personalized recommendation methods.

---

### Baseline Models

| Model | Parameters  | Precision@5 | Recall@5 | Notes                    |
| ----- | ----------- | ----------- | -------- | ------------------------ |
| KNN   | k = 5       | 0.044       | 0.0665   | First personalized model |
| SVD   | 50 factors  | 0.036       | 0.0559   | Underperformed vs KNN    |
| SVD   | 120 factors | 0.052       | 0.0967   | Best initial SVD model   |

---

## KNN Experiments

### Hybrid Reranking (KNN)

A hybrid reranking strategy was introduced to enhance recommendation quality by combining collaborative filtering outputs with content-based signals (book **tags** and **book_tags**).

#### Results

| Model               | Precision@5 | Recall@5 | Evaluated Users |
| ------------------- | ----------- | -------- | --------------- |
| KNN                 | 0.044       | 0.0665   | 100             |
| KNN + Hybrid rerank | 0.084       | 0.1573   | 100             |

Hybrid reranking significantly improves KNN performance, nearly doubling Precision@5.

---

### Candidate Size Impact (KNN + Hybrid rerank)

| Candidate Set | Precision@5 | Recall@5 |
| ------------- | ----------- | -------- |
| 100           | 0.084       | 0.1573   |
| 150           | 0.086       | 0.1662   |

Increasing the candidate pool improves recall and slightly improves precision.

---

### Final KNN Optimization (3-Phase Sweep)

| Run             | Configuration            | Precision@5 | Recall@5 |
| --------------- | ------------------------ | ----------- | -------- |
| Candidate sweep | `candidate_n = 400`      | 0.092       | 0.1870   |
| Weight tuning   | `0.6 / 0.3 / 0.1`        | 0.092       | 0.1870   |
| Adaptive policy | Cold/Warm weights tested | 0.092       | 0.1870   |

Final decision: static configuration (simpler, same performance).

---

## SVD Experiments

### Baseline and Hybrid (SVD)

| Model                             | Precision@5 | Recall@5 | Evaluated Users |
| --------------------------------- | ----------- | -------- | --------------- |
| SVD (300 factors)                 | 0.078       | 0.1487   | 100             |
| SVD + Hybrid rerank (300 factors) | 0.080       | 0.1471   | 100             |

Hybrid reranking provides only marginal improvement for SVD.

---

## Final Configuration

```python
candidate_n = 400
collaborative_weight = 0.6
content_weight = 0.3
to_read_weight = 0.1
```

| Model                             | Precision@5 | Recall@5 | Evaluated Users |
| --------------------------------- | ----------- | -------- | --------------- |
| KNN                               | 0.044       | 0.0665   | 100             |
| KNN + Hybrid rerank               | 0.092       | 0.1870   | 100             |
| SVD (300 factors)                 | 0.078       | 0.1487   | 100             |
| SVD + Hybrid rerank (300 factors) | 0.080       | 0.1471   | 100             |

## Final Model

We selected **KNN + Hybrid Reranking** as our final model due to superior performance.

## Other Models Explored

- SVD (baseline)
- SVD + Hybrid

These models were evaluated but not selected.
