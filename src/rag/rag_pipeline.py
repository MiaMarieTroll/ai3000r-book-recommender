"""RAG pipeline placeholder.

Status: future work. A full RAG flow is deferred until retriever and
generator modules are implemented.
"""

class RAGPipeline:
    def __init__(self, book_data):
        self.book_data = book_data
        raise NotImplementedError(
            "RAGPipeline is not implemented yet. "
            "Retriever and Generator modules are placeholders."
        )

    def recommend(self, top_books):
        """Return RAG-enhanced recommendations (future implementation)."""
        raise NotImplementedError(
            "RAG recommendation flow is not implemented yet."
        )
