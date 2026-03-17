"""
baseline_model.py

Simple popularity-based recommendation
Baseline-modellen vår er en enkel popularity-based recommender. 
Først grupperer vi alle ratings per bok (`book_id`) og beregner:
1) gjennomsnittlig rating (`avg_rating`) og 
2) antall ratings (`rating_count`).

Deretter filtrerer vi bort bøker med få vurderinger (minimum `min_ratings`) for å unngå at bøker med veldig få stemmer får kunstig høy score. 
Vi kobler så resultatet med bokmetadata (tittel og forfatter), sorterer etter høyest `avg_rating` (og deretter `rating_count`), og returnerer Top-N bøker.

Det vi viser i output er altså de høyest rangerte bøkene totalt i datasettet, sammen med tittel, forfatter, gjennomsnittlig rating og hvor mange ratings hver bok har fått.
Dette fungerer som en baseline/referanse før mer avansert, personlig anbefaling med collaborative filtering (KNN).
"""


# ============================================
# Compute Average Ratings
# ============================================

def compute_average_ratings(ratings_df):
    """
    Group by book_id
    Calculate mean rating
    """
    avg_ratings = (
        ratings_df.groupby("book_id", as_index=False)["rating"]
        .agg(avg_rating="mean", rating_count="count")
    )
    return avg_ratings


# ============================================
# Get Top N Books
# ============================================

def get_top_books(avg_ratings, books_df, n=10, min_ratings=50):
    """
    Return top n highest rated books
    """
    filtered = avg_ratings[avg_ratings["rating_count"] >= min_ratings]
    merged = filtered.merge(books_df, left_on="book_id", right_on="id", how="left")
    top_books = merged.sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(n)

    # Keep a small, clear output table for display in main.py.
    available_cols = ["book_id", "title", "authors", "avg_rating", "rating_count"]
    selected_cols = [col for col in available_cols if col in top_books.columns]
    result = top_books[selected_cols].reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result
