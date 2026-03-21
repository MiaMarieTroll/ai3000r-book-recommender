"""
test_baseline_model.py

Simple tests for baseline_model functions
"""

from src.data_loader import load_books, load_ratings
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing
from src.baseline_model import compute_average_ratings, get_top_books


def test_compute_average_ratings():
    """Test average ratings computation"""
    print("\n=== Test 1: Compute Average Ratings ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    
    # Compute average ratings
    avg_ratings = compute_average_ratings(ratings)
    
    print(f"✓ Computed average ratings for {len(avg_ratings)} books")
    print(f"  Columns: {list(avg_ratings.columns)}")
    print(f"  First 3 rows:")
    print(avg_ratings.head(3).to_string(index=False))


def test_get_top_books():
    """Test getting top N books"""
    print("\n=== Test 2: Get Top N Books ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    
    # Compute average ratings
    avg_ratings = compute_average_ratings(ratings)
    
    # Get top books
    top_books = get_top_books(avg_ratings, books, n=5, min_ratings=50)
    
    print(f"✓ Retrieved top {len(top_books)} books")
    print(f"  Columns: {list(top_books.columns)}")
    print(f"\nTop 5 books by rating:")
    print(top_books.to_string(index=False))


def test_get_top_books_no_duplicates():
    """Test that top books has correct rank ordering"""
    print("\n=== Test 3: Rank Ordering ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    
    # Compute average ratings and get top books
    avg_ratings = compute_average_ratings(ratings)
    top_books = get_top_books(avg_ratings, books, n=10, min_ratings=50)
    
    # Check rank ordering (should be 1, 2, 3, ...)
    expected_ranks = list(range(1, len(top_books) + 1))
    actual_ranks = list(top_books["rank"])
    
    if expected_ranks == actual_ranks:
        print(f"✓ Rank ordering correct (1 to {len(top_books)})")
    else:
        print(f"✗ Rank ordering incorrect. Expected {expected_ranks}, got {actual_ranks}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("BASELINE MODEL TESTS")
    print("=" * 60)
    
    try:
        test_compute_average_ratings()
        test_get_top_books()
        test_get_top_books_no_duplicates()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

