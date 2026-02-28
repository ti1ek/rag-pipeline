from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import QDRANT_PATH


def create_collection(
    documents: list[Document],
    embedding_model,
    collection_name: str = "rag_documents",
) -> QdrantVectorStore:
    """Create a new Qdrant collection from documents."""
    client = QdrantClient(path=QDRANT_PATH)

    # Delete if exists to allow recreation
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Get embedding dimension from a test embed
    test_embedding = embedding_model.embed_query("test")
    dim = len(test_embedding)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.close()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        path=QDRANT_PATH,
        collection_name=collection_name,
    )
    return vector_store


def load_collection(
    embedding_model,
    collection_name: str = "rag_documents",
) -> QdrantVectorStore:
    """Load an existing Qdrant collection."""
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=QDRANT_PATH,
        collection_name=collection_name,
    )
