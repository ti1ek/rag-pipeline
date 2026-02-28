from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def create_collection(
    documents: list[Document],
    embedding_model,
    collection_name: str = "rag_documents",
) -> tuple[QdrantVectorStore, QdrantClient]:
    """Create a new Qdrant collection from documents (in-memory)."""
    client = QdrantClient(":memory:")

    # Get embedding dimension
    test_embedding = embedding_model.embed_query("test")
    dim = len(test_embedding)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    vector_store.add_documents(documents)
    return vector_store, client
