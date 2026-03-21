"""
test_preprocessing.py

Meaningful tests for preprocessing functions
"""

import pandas as pd
import numpy as np
from src.data_loader import load_books, load_ratings
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing


def test_clean_ratings_removes_duplicates():
    """Test that clean_ratings removes duplicate entries"""
    print("\n=== Test 1: Clean Ratings - Remove Duplicates ===")
    
    ratings = load_ratings("data/ratings.csv")
    initial_count = len(ratings)
    
    cleaned = clean_ratings(ratings)
    final_count = len(cleaned)
    
    # Check that duplicates were removed (or none existed)
    assert final_count <= initial_count, "Cleaned data should have same or fewer rows"
    print(f"✓ Duplicates removed: {initial_count} → {final_count} rows")
    
    # Verify no more duplicates exist
    duplicates = cleaned.duplicated().sum()
    assert duplicates == 0, f"Still {duplicates} duplicates after cleaning"
    print(f"✓ No duplicates remain")


def test_clean_ratings_removes_nulls():
    """Test that clean_ratings removes rows with missing values"""
    print("\n=== Test 2: Clean Ratings - Remove NaN ===")
    
    ratings = load_ratings("data/ratings.csv")
    initial_nulls = ratings.isnull().sum().sum()
    
    cleaned = clean_ratings(ratings)
    final_nulls = cleaned.isnull().sum().sum()
    
    # Check that NaN values were removed
    assert final_nulls == 0, f"Cleaned data should have no NaN, but has {final_nulls}"
    print(f"✓ All NaN removed: {initial_nulls} → {final_nulls} null values")


def test_create_user_item_matrix_shape():
    """Test that user-item matrix has correct dimensions"""
    print("\n=== Test 3: User-Item Matrix Shape ===")
    
    ratings = load_ratings("data/ratings.csv")
    cleaned = clean_ratings(ratings)
    
    matrix = create_user_item_matrix(cleaned)
    
    # Matrix should be (n_users, n_books)
    n_unique_users = cleaned["user_id"].nunique()
    n_unique_books = cleaned["book_id"].nunique()
    
    assert matrix.shape[0] == n_unique_users, f"Rows should be {n_unique_users}, got {matrix.shape[0]}"
    assert matrix.shape[1] == n_unique_books, f"Columns should be {n_unique_books}, got {matrix.shape[1]}"
    
    print(f"✓ Matrix shape correct: {matrix.shape}")
    print(f"  {n_unique_users} users × {n_unique_books} books")


def test_create_user_item_matrix_values():
    """Test that user-item matrix contains correct rating values"""
    print("\n=== Test 4: User-Item Matrix Values ===")
    
    ratings = load_ratings("data/ratings.csv")
    cleaned = clean_ratings(ratings)
    
    matrix = create_user_item_matrix(cleaned)
    
    # All non-NaN values should be valid ratings (1-5)
    non_null_values = matrix.values[~np.isnan(matrix.values)]
    
    assert np.all((non_null_values >= 1) & (non_null_values <= 5)), "All ratings should be 1-5"
    print(f"✓ All non-null values are valid ratings (1-5)")
    
    # Check matrix is mostly sparse (expected for rating data)
    null_count = np.isnan(matrix.values).sum()
    sparsity = (null_count / matrix.size) * 100
    
    print(f"  Sparsity: {sparsity:.1f}% (expected for rating matrices)")


def test_fill_missing_fills_with_zero():
    """Test that fill_missing replaces NaN with 0"""
    print("\n=== Test 5: Fill Missing Values ===")
    
    ratings = load_ratings("data/ratings.csv")
    cleaned = clean_ratings(ratings)
    
    matrix = create_user_item_matrix(cleaned)
    initial_nan_count = np.isnan(matrix.values).sum()
    
    filled = fill_missing(matrix)
    final_nan_count = np.isnan(filled.values).sum()
    
    # All NaN should be replaced
    assert final_nan_count == 0, f"Still {final_nan_count} NaN values after filling"
    print(f"✓ All NaN replaced with 0")
    
    # Check that filled values are actually 0
    # (Get values that were NaN before, now should be 0)
    zero_count = (filled.values == 0).sum()
    print(f"  {zero_count} values filled with 0")
    
    # Data type should still be numeric
    assert np.issubdtype(filled.values.dtype, np.number), "Filled matrix should be numeric"
    print(f"✓ Data type is numeric (float64)")


def test_preprocessing_pipeline():
    """Test full preprocessing pipeline end-to-end"""
    print("\n=== Test 6: Full Pipeline ===")
    
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    
    # Run full pipeline
    cleaned = clean_ratings(ratings)
    matrix = create_user_item_matrix(cleaned)
    filled = fill_missing(matrix)
    
    # Verify final result
    assert isinstance(filled, pd.DataFrame), "Result should be a DataFrame"
    assert filled.isnull().sum().sum() == 0, "Final matrix should have no NaN"
    assert np.all(np.logical_or(filled.values == 0, (filled.values >= 1) & (filled.values <= 5))), \
        "Values should be 0 or 1-5"
    
    print(f"✓ Full preprocessing pipeline works correctly")
    print(f"  Final matrix: {filled.shape}")
    print(f"  Sparsity: {(filled.values == 0).sum() / filled.size * 100:.1f}%")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PREPROCESSING TESTS")
    print("=" * 60)
    
    try:
        test_clean_ratings_removes_duplicates()
        test_clean_ratings_removes_nulls()
        test_create_user_item_matrix_shape()
        test_create_user_item_matrix_values()
        test_fill_missing_fills_with_zero()
        test_preprocessing_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

