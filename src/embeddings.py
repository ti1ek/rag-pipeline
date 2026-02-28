from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "intfloat/multilingual-e5-large") -> HuggingFaceEmbeddings:
    """Load a HuggingFace embedding model with appropriate query/passage prefixes."""
    # E5 models need "query: " and "passage: " prefixes
    if "e5" in model_name.lower():
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="query: ",
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
