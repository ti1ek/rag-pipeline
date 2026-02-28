from langchain_community.embeddings import HuggingFaceEmbeddings


class E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFace E5 embeddings with automatic query/passage prefixes."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])


def get_embedding_model(model_name: str = "intfloat/multilingual-e5-large") -> HuggingFaceEmbeddings:
    """Load a HuggingFace embedding model with appropriate prefixes."""
    kwargs = {
        "model_name": model_name,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    }

    if "e5" in model_name.lower():
        return E5Embeddings(**kwargs)

    return HuggingFaceEmbeddings(**kwargs)
