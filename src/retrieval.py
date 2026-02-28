from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def dense_retrieve(vector_store, query: str, top_k: int = 5) -> list[Document]:
    """Dense retrieval using vector similarity."""
    return vector_store.similarity_search(query, k=top_k)


class BM25Retriever:
    """BM25 retriever over a list of documents."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]


def hybrid_retrieve(
    vector_store,
    bm25_retriever: BM25Retriever,
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
) -> list[Document]:
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF).
    alpha=1 -> vector only, alpha=0 -> BM25 only.
    """
    k_rrf = 60  # RRF constant

    vector_docs = vector_store.similarity_search(query, k=top_k * 2)
    bm25_docs = bm25_retriever.retrieve(query, top_k=top_k * 2)

    # Build RRF scores keyed by document content
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + alpha * (1.0 / (k_rrf + rank + 1))
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + (1 - alpha) * (1.0 / (k_rrf + rank + 1))
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [doc_map[k] for k in sorted_keys]


def rerank(query: str, documents: list[Document], top_k: int = 5, model_name: str = "BAAI/bge-reranker-v2-m3") -> list[Document]:
    """Rerank documents using a cross-encoder model."""
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(model_name, use_fp16=True)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.compute_score(pairs)

    if isinstance(scores, float):
        scores = [scores]

    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
