# Book Recommender System
**Course:** Artificial Intelligence for Business Applications — AI3000R-1 26V

---

## Description
This project implements a **collaborative filtering book recommender system** using the K-Nearest Neighbors (KNN) algorithm. The system recommends books to users based on the preferences of similar users.

The project is inspired by the chapter *"Building a Recommendation System" in Artificial intelligence with Python : your complete guide to building intelligent apps using Python 3.x and TensorFlow 2* and covers:
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
│   └── evaluation.py        # Precision@K, Recall@K metrics
│
├── main.py                  # Main pipeline
└── README.md
```

---

## Dataset
[GoodBooks-10k](https://github.com/zygmuntz/goodbooks-10k) — contains 10,000 books and ~1 million ratings from real users.

| File | Description |
|---|---|
| `books.csv` | Book metadata (title, author, year, etc.) |
| `ratings.csv` | User ratings (`user_id`, `book_id`, `rating`) |

---

## Setup
```bash
pip install pandas numpy scikit-learn
```

---

## How to Run
```bash
python main.py
```

---

## Pipeline Overview
1. **Load Data** — Read `books.csv` and `ratings.csv`, print summary
2. **Preprocess** — Clean ratings, build user-item matrix, fill missing values with 0
3. **Baseline Model** — Recommend top-N most popular books by average rating
4. **Collaborative Filtering** — KNN model finds similar users and recommends books they liked
5. **Evaluation** — Measure recommendation quality with Precision@K and Recall@K

---

## Authors
- Mia — data loading, preprocessing, baseline model
- TODO: Add name — collaborative filtering (KNN), evaluation

