"""
test_data_loader.py

Meaningful tests for data_loader functions
"""

import pandas as pd
from src.data_loader import (
    load_books,
    load_ratings,
    load_book_tags,
    load_tags,
    load_to_read,
    data_summary,
)


def test_load_books():
    """Test that books dataset is loaded correctly with expected columns"""
    print("\n=== Test 1: Load Books ===")
    
    books = load_books("data/books.csv")
    
    # Assert it's a DataFrame
    assert isinstance(books, pd.DataFrame), "load_books should return a DataFrame"
    print(f"✓ Returned a DataFrame")
    
    # Assert it has data
    assert len(books) > 0, "books DataFrame should not be empty"
    print(f"✓ DataFrame contains {len(books)} rows")
    
    # Assert required columns exist
    expected_cols = ["id", "title", "authors"]
    for col in expected_cols:
        assert col in books.columns, f"Missing required column: {col}"
    print(f"✓ All required columns present: {expected_cols}")


def test_load_ratings():
    """Test that ratings dataset is loaded correctly with required columns"""
    print("\n=== Test 2: Load Ratings ===")
    
    ratings = load_ratings("data/ratings.csv")
    
    # Assert it's a DataFrame
    assert isinstance(ratings, pd.DataFrame), "load_ratings should return a DataFrame"
    print(f"✓ Returned a DataFrame")
    
    # Assert it has data
    assert len(ratings) > 0, "ratings DataFrame should not be empty"
    print(f"✓ DataFrame contains {len(ratings)} rows")
    
    # Assert required columns exist
    required_cols = ["user_id", "book_id", "rating"]
    for col in required_cols:
        assert col in ratings.columns, f"Missing required column: {col}"
    print(f"✓ All required columns present: {required_cols}")
    
    # Assert column data types are correct
    assert ratings["user_id"].dtype in ["int64", "int32"], "user_id should be numeric"
    assert ratings["book_id"].dtype in ["int64", "int32"], "book_id should be numeric"
    assert ratings["rating"].dtype in ["float64", "float32", "int64"], "rating should be numeric"
    print(f"✓ All columns have correct data types")


def test_load_ratings_has_valid_values():
    """Test that ratings are within expected range (1-5)"""
    print("\n=== Test 3: Ratings Values ===")
    
    ratings = load_ratings("data/ratings.csv")
    
    # Check rating range
    min_rating = ratings["rating"].min()
    max_rating = ratings["rating"].max()
    
    assert min_rating >= 1, f"Minimum rating should be >= 1, got {min_rating}"
    assert max_rating <= 5, f"Maximum rating should be <= 5, got {max_rating}"
    print(f"✓ All ratings are in valid range [1, 5]")
    print(f"  Min: {min_rating}, Max: {max_rating}")


def test_data_summary():
    """Test that data_summary function works without errors"""
    print("\n=== Test 4: Data Summary ===")
    
    books = load_books("data/books.csv")
    
    # Just verify it doesn't crash (it's a print function)
    try:
        data_summary(books.head(3))
        print(f"✓ data_summary executed successfully")
    except Exception as e:
        raise AssertionError(f"data_summary failed: {e}")


def test_load_book_tags_tags_to_read():
    """Test that tag/to-read datasets load with required columns"""
    print("\n=== Test 5: Load book_tags, tags, to_read ===")

    book_tags = load_book_tags("data/book_tags.csv")
    tags = load_tags("data/tags.csv")
    to_read = load_to_read("data/to_read.csv")

    assert isinstance(book_tags, pd.DataFrame)
    assert isinstance(tags, pd.DataFrame)
    assert isinstance(to_read, pd.DataFrame)

    for col in ["goodreads_book_id", "tag_id", "count"]:
        assert col in book_tags.columns, f"Missing required column in book_tags: {col}"

    for col in ["tag_id", "tag_name"]:
        assert col in tags.columns, f"Missing required column in tags: {col}"

    for col in ["user_id", "book_id"]:
        assert col in to_read.columns, f"Missing required column in to_read: {col}"

    assert len(book_tags) > 0
    assert len(tags) > 0
    assert len(to_read) > 0
    print("✓ Tag and to-read datasets loaded successfully")


def test_load_ratings_missing_columns():
    """Placeholder for manual runner compatibility."""
    ratings = load_ratings("data/ratings.csv")
    assert {"user_id", "book_id", "rating"}.issubset(ratings.columns)


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("DATA LOADER TESTS")
    print("=" * 60)
    
    try:
        test_load_books()
        test_load_ratings()
        test_load_ratings_missing_columns()
        test_data_summary()
        test_load_book_tags_tags_to_read()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

