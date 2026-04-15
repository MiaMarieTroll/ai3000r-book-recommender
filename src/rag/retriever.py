"""RAG retriever placeholder.

Status: future work. This module is intentionally not implemented yet.
"""


class Retriever:
    def __init__(self, book_data):
        """Initialize retriever placeholder with source book data."""
        self.book_data = book_data
        raise NotImplementedError(
            "Retriever is not implemented yet. "
            "Planned for a future full RAG pipeline."
        )

    def retrieve(self, top_books, k=3):
        """Retrieve relevant context for top books (future implementation)."""
        raise NotImplementedError(
            "Retrieval functionality is not implemented yet."
        )
