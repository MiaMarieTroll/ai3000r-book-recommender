# Book Recommender System

**Course:** Artificial Intelligence for Business Applications — AI3000R-1 26V

---

## Description

This project implements a **hybrid book recommender system** using multiple approaches:

- Baseline popularity model (average rating + rating count)
- Collaborative filtering with KNN (user-based nearest neighbors)
- Matrix factorization with SVD (latent-factor collaborative filtering)
- Hybrid reranking that combines collaborative score, content/tag affinity, and to-read signals

The project is inspired by recommender-system methods from course material.

---

## Current Scope and RAG Status

### Implemented

| Component | Status | Notes |
| --- | --- | --- |
| Data loading and validation | Implemented | Loads `books`, `ratings`, `book_tags`, `tags`, `to_read` |
| Preprocessing | Implemented | Cleaning, user-item matrix, and missing-value handling |
| Baseline recommender | Implemented | Popularity-based ranking |
| KNN collaborative filtering | Implemented | User-based cosine similarity |
| SVD matrix factorization | Implemented | Latent-factor recommendations |
| Hybrid reranking | Implemented | `src/hybrid_model.py` |
| Evaluation | Implemented | Precision@K and Recall@K for KNN/SVD and hybrid variants |
| Hybrid tuning sweeps | Implemented | `src/run_hybrid_tuning.py` |

### Future work (not implemented yet)

| Module | Status | Notes |
| --- | --- | --- |
| `src/rag/retriever.py` | Not implemented | Placeholder/TODO |
| `src/rag/generator.py` | Not implemented | Placeholder/TODO |
| `src/rag/rag_pipeline.py` | Partial scaffold | Depends on retriever and generator |


---

## Project Structure

```text
ai3000r-book-recommender/
|
+-- data/
|   +-- books.csv
|   +-- ratings.csv
|   +-- book_tags.csv
|   +-- tags.csv
|   +-- to_read.csv
|
+-- src/
|   +-- data_loader.py
|   +-- preprocessing.py
|   +-- baseline_model.py
|   +-- collaborative_model.py
|   +-- matrix_factorization_model.py
|   +-- hybrid_model.py
|   +-- evaluation.py
|   +-- run_hybrid_tuning.py
|   +-- rag/
|       +-- content_model.py      # Backward-compatibility shim
|       +-- retriever.py          # Future work
|       +-- generator.py          # Future work
|       +-- rag_pipeline.py       # Future work
|
+-- tests/
|   +-- __init__.py
|   +-- test_data_loader.py
|   +-- test_preprocessing.py
|   +-- test_baseline_model.py
|   +-- test_collaborative.py
|   +-- test_hybrid_model.py
|
+-- main.py
+-- requirements.txt
+-- README.md
```

---

## Dataset

This project uses the [GoodBooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k).

| File | Description |
| --- | --- |
| `books.csv` | Book metadata (`id`, title, author, etc.) |
| `ratings.csv` | User ratings (`user_id`, `book_id`, `rating`) |
| `book_tags.csv` | Book-to-tag links with tag counts |
| `tags.csv` | Tag ID to tag name mapping |
| `to_read.csv` | User to-read lists |

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to Run

### Main pipeline

```powershell
python main.py
```

### Run all tests

```powershell
python -m pytest tests/ -v
```

### Run one test module (optional)

```powershell
python -m pytest tests/test_collaborative.py -v
```

### Run hybrid tuning sweeps

```powershell
python -m src.run_hybrid_tuning
```

This runs three experiment phases and saves CSV files in `results/`:

- `run1_candidate_sweep.csv`
- `run2_weight_sweep.csv`
- `run3_adaptive_policy_sweep.csv`

---

## Pipeline Overview

1. **Load data** - read and validate all required datasets
2. **Preprocess** - clean ratings, build user-item matrix, fill missing values
3. **Baseline model** - popularity-based top-N recommendations
4. **KNN model** - collaborative filtering from similar users
5. **Hybrid reranking (KNN output)** - blend collaborative, content tags, and to-read signals
6. **SVD model** - latent-factor recommendations
7. **Hybrid reranking (SVD output)** - rerank SVD candidates with the same hybrid logic
8. **Evaluation** - compare KNN, KNN+Hybrid, SVD, and SVD+Hybrid with Precision@5 and Recall@5

---

## Authors

- Mia Marie Iversen Trollstol - project setup/structure, data loading, preprocessing, baseline model and overall structure of the report and presentation.
- Chui Ling Ng - collaborative filtering (KNN), SVD, hybrid reranking, evaluation
