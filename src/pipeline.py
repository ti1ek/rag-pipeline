from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.config import DEFAULT_CONFIG, OPENAI_API_KEY
from src.chunking import chunk_text
from src.embeddings import get_embedding_model
from src.vectorstore import create_collection
from src.retrieval import dense_retrieve, BM25Retriever, hybrid_retrieve, rerank
from src.generation import generate_answer


class RAGPipeline:
    """Configurable RAG pipeline wiring parsing, chunking, retrieval, and generation."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.embedding_model = get_embedding_model(self.config["embedding_model"])
        self.vector_store = None
        self.bm25_retriever = None
        self.documents = None
        self.qdrant_client = None

    def ingest(self, parsed_texts: dict[str, str]):
        """Chunk and index parsed texts into vector store."""
        all_docs = []
        for source, text in parsed_texts.items():
            docs = chunk_text(
                text, source,
                strategy=self.config["chunking_strategy"],
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
            )
            all_docs.extend(docs)

        self.documents = all_docs
        self.vector_store, self.qdrant_client = create_collection(
            all_docs, self.embedding_model, self.config["collection_name"]
        )
        self.bm25_retriever = BM25Retriever(all_docs)
        return len(all_docs)

    def _rewrite_query(self, question: str) -> str:
        """Rewrite query using LLM for better retrieval."""
        llm = ChatOpenAI(model=self.config["llm_model"], api_key=OPENAI_API_KEY, temperature=0)
        prompt = (
            "Перефразируй следующий вопрос для лучшего поиска по документам. "
            "Сохрани смысл, но сделай запрос более точным и информативным. "
            "Верни только перефразированный вопрос.\n\n"
            f"Вопрос: {question}"
        )
        return llm.invoke(prompt).content

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve relevant documents for a question."""
        q = self._rewrite_query(question) if self.config["use_query_rewriting"] else question

        alpha = self.config["alpha"]
        top_k = self.config["top_k"]

        # Pure dense
        if alpha == 1.0 or self.bm25_retriever is None:
            docs = dense_retrieve(self.vector_store, q, top_k=top_k)
        # Pure BM25
        elif alpha == 0.0:
            docs = self.bm25_retriever.retrieve(q, top_k=top_k)
        # Hybrid
        else:
            docs = hybrid_retrieve(self.vector_store, self.bm25_retriever, q, top_k=top_k, alpha=alpha)

        if self.config["use_reranking"]:
            docs = rerank(q, docs, top_k=top_k, model_name=self.config["reranker_model"])

        return docs

    def query(self, question: str) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        docs = self.retrieve(question)
        answer = generate_answer(question, docs, model=self.config["llm_model"])
        return {
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "source_documents": docs,
        }
