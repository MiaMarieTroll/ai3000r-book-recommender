### Experiment Overview

The recommender system development started with a simple popularity-based baseline model. This baseline recommends the highest-rated books overall and is used as a reference before testing personalized recommendation methods.

After establishing the baseline, we evaluated personalized machine learning models to improve recommendation quality.

| Model | Parameters  | Precision@5 | Recall@5 | Notes                    |
| ----- | ----------- | ----------- | -------- | ------------------------ |
| KNN   | k = 5       | 0.044       | 0.0665   | First personalized model |
| SVD   | 50 factors  | 0.036       | 0.0559   | Underperformed vs KNN    |
| SVD   | 120 factors | 0.052       | 0.0967   | Best so far              |
