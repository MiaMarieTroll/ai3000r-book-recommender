### Experiment Overview

This file is used to track and compare different machine learning approaches for the book recommendation system and evaluate their effectiveness. At this stage, we are comparing KNN and SVD models along with various enhancements such as hybrid reranking. The purpose of this document is to keep a record of experimental results, while only the best-performing model will be selected and retained in the final repository.

The recommender system development started with a simple popularity-based baseline model. This baseline recommends the highest-rated books overall and serves as a reference before evaluating personalized recommendation methods.

After establishing the baseline, we evaluated personalized machine learning models to improve recommendation quality.

| Model | Parameters  | Precision@5 | Recall@5 | Notes                    |
| ----- | ----------- | ----------- | -------- | ------------------------ |
| KNN   | k = 5       | 0.044       | 0.0665   | First personalized model |
| SVD   | 50 factors  | 0.036       | 0.0559   | Underperformed vs KNN    |
| SVD   | 120 factors | 0.052       | 0.0967   | Best collaborative model |

---

### Hybrid Reranking

A hybrid reranking strategy was introduced to enhance recommendation quality by combining collaborative filtering outputs with content-based signals. Specifically, book **tags** and **book_tags** were used to refine the ranking of recommended items.

The performance of the models with and without hybrid reranking is shown below:

| Model                   | Precision@5 | Recall@5 | Evaluated Users | Notes                                      |
| ----------------------- | ----------- | -------- | --------------- | ------------------------------------------ |
| KNN                     | 0.044       | 0.0665   | 100             | Baseline KNN                               |
| KNN + Hybrid rerank     | 0.084       | 0.1573   | 100             | Significant improvement with hybrid method |
| SVD (120 factors)       | 0.052       | 0.0967   | 100             | Base SVD                                   |
| SVD + Hybrid rerank     | 0.064       | 0.1250   | 100             | Moderate improvement                       |
| SVD tuned (120 factors) | 0.052       | 0.0967   | 100             | No improvement from tuning                 |

The hybrid reranking approach produced the best overall performance, particularly when combined with KNN, where Precision@5 nearly doubled compared to the base model.
