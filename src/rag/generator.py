"""RAG generator placeholder.

Status: future work. This module is intentionally not implemented yet.
"""


class Generator:
    def __init__(self, model_name=None):
        """Initialize generator placeholder with optional model name."""
        self.model_name = model_name
        raise NotImplementedError(
            "Generator is not implemented yet. "
            "Planned for a future full RAG pipeline."
        )

    def generate(self, user_profile, recommended_items, retrieved_context):
        """Generate explanation text from retrieved context (future implementation)."""
        raise NotImplementedError(
            "Generation functionality is not implemented yet."
        )
