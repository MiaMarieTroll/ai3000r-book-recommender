"""
test_collaborative.py

Simple tests for collaborative_model functions
"""

from src.data_loader import load_books, load_ratings
from src.preprocessing import clean_ratings, create_user_item_matrix, fill_missing
from src.collaborative_model import build_knn_model, find_similar_users, recommend_books


def test_build_knn_model():
    """Test KNN model building"""
    print("\n=== Test 1: Build KNN Model ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_matrix = fill_missing(user_item_matrix)
    
    # Build model
    model = build_knn_model(user_item_matrix)
    
    print(f"✓ KNN model created successfully")
    print(f"  Matrix shape: {user_item_matrix.shape}")
    print(f"  Model: {model}")


def test_find_similar_users():
    """Test finding similar users"""
    print("\n=== Test 2: Find Similar Users ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_matrix = fill_missing(user_item_matrix)
    
    # Build model
    model = build_knn_model(user_item_matrix)
    
    # Find similar users
    target_user = 2
    similar_users = find_similar_users(model, user_item_matrix, target_user, k=5)
    
    print(f"✓ Found {len(similar_users)} similar users to user {target_user}")
    for user_id, similarity in similar_users:
        print(f"  User {user_id}: similarity = {similarity:.4f}")


def test_recommend_books():
    """Test book recommendations"""
    print("\n=== Test 3: Recommend Books ===")
    
    # Load and preprocess data
    books = load_books("data/books.csv")
    ratings = load_ratings("data/ratings.csv")
    ratings = clean_ratings(ratings)
    user_item_matrix = create_user_item_matrix(ratings)
    user_item_matrix = fill_missing(user_item_matrix)
    
    # Build model
    model = build_knn_model(user_item_matrix)
    
    # Get recommendations
    target_user = 1
    recommendations = recommend_books(
        user_id=target_user,
        user_item_matrix=user_item_matrix,
        books_df=books,
        model=model,
        n=5
    )

    assert "book_id" in recommendations.columns, "Recommendations must include book_id"
    assert recommendations["book_id"].notna().all(), "Recommended book_id values cannot be null"
    
    print(f"✓ Generated {len(recommendations)} recommendations for user {target_user}")
    print("\nTop 3 recommendations:")
    print(recommendations.head(3).to_string(index=False))


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("COLLABORATIVE MODEL TESTS")
    print("=" * 60)
    
    try:
        test_build_knn_model()
        test_find_similar_users()
        test_recommend_books()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

