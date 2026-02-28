import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", ".qdrant_storage")

# Default hyperparameters
DEFAULT_CONFIG = {
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "chunking_strategy": "fixed",  # fixed, recursive, layout_aware
    "embedding_model": "intfloat/multilingual-e5-large",
    "top_k": 5,
    "alpha": 0.5,  # hybrid search: 0=BM25 only, 1=vector only
    "use_reranking": False,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "use_query_rewriting": False,
    "llm_model": "gpt-4o-mini",
    "collection_name": "rag_documents",
}
