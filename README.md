# Book Recommender System

**Course:** Artificial Intelligence for Business Applications — AI3000R-1 26V

---

## Description

This project implements a **collaborative filtering book recommender system** using the K-Nearest Neighbors (KNN) algorithm. The system recommends books to users based on the preferences of similar users.

The project is inspired by the chapter _"Building a Recommendation System" in Artificial intelligence with Python : your complete guide to building intelligent apps using Python 3.x and TensorFlow 2_ and covers:

- Extracting nearest neighbors
- Building a K-Nearest Neighbors classifier
- Computing similarity scores
- Finding similar users using collaborative filtering
- Building a book recommendation system

---

## Project Structure

```
ai3000r-book-recommender/
│
├── data/
│   ├── books.csv           # Book metadata
│   └── ratings.csv         # User ratings
│
├── src/
│   ├── data_loader.py       # Load and validate datasets
│   ├── preprocessing.py     # Clean data, build user-item matrix
│   ├── baseline_model.py    # Popularity-based baseline recommender
│   ├── collaborative_model.py  # KNN collaborative filtering
│   ├── evaluation.py        # Precision@K, Recall@K metrics
│   └── rag/
│       ├── content_model.py     # Content-based filtering (Future)
│       ├── retriever.py         # RAG retriever (Future)
│       ├── generator.py         # RAG generator (Future)
│       └── rag_pipeline.py      # RAG pipeline (Future)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_baseline_model.py
│   └── test_collaborative.py  # Tests for collaborative filtering
│
├── main.py                  # Main pipeline
├── requirements.txt         # Project dependencies
└── README.md
```

---

## Dataset

[GoodBooks-10k](https://github.com/zygmuntz/goodbooks-10k) — contains 10,000 books and ~1 million ratings from real users.

| File          | Description                                   |
| ------------- | --------------------------------------------- |
| `books.csv`   | Book metadata (title, author, year, etc.)     |
| `ratings.csv` | User ratings (`user_id`, `book_id`, `rating`) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Main Pipeline

```bash
python main.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Hybrid Tuning Sweeps

```bash
python -m src.run_hybrid_tuning
```

This runs 3 experiment phases and saves CSV outputs in `results/`:

- `run1_candidate_sweep.csv`
- `run2_weight_sweep.csv`
- `run3_adaptive_policy_sweep.csv`

Run a single test module (optional):

```bash
python -m pytest tests/test_collaborative.py -v
```

---

## Pipeline Overview

1. **Load Data** ✅ — Read `books.csv` and `ratings.csv`, print summary
2. **Preprocess** ✅ — Clean ratings, build user-item matrix, fill missing values with 0
3. **Baseline Model** ✅ — Recommend top-N most popular books by average rating
4. **Collaborative Filtering** ✅ — KNN model finds similar users and recommends books they liked
5. **Evaluation** 📝 — Measure recommendation quality with Precision@K and Recall@K (in progress)
6. **Testing** ✅ — Automated tests for `data_loader`, `preprocessing`, `baseline_model`, and `collaborative_model`

---

## Authors

- Mia Marie Iversen Trollstøl — data loading, preprocessing, baseline model
- Chui Ling Ng — collaborative filtering (KNN), evaluation
