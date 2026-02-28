# RAG Pipeline for Kazakh Companies

A production-ready Retrieval-Augmented Generation pipeline for answering questions over Kazakh company annual reports written in Russian/Kazakh, featuring complex PDF layouts (financial tables, ESG diagrams, multi-column blocks).

**Documents:**
- `ktj.pdf` — KTZh integrated annual report (~368 pages, financial tables, ESG data, diagrams)
- `matnp_2024_rus.pdf` — Maten Petroleum annual report (~20–30 pages, oil production data, sales tables)

---

## Pipeline Overview

```
PDF
 └─► LlamaParse (markdown with table preservation)
      └─► Chunking (Fixed / Recursive / Layout-Aware)
           └─► multilingual-e5-large embeddings
                └─► Qdrant (cosine similarity, in-memory)
                     └─► Retrieval (Dense / BM25 / Hybrid RRF)
                          └─► Cross-Encoder Reranking (optional)
                               └─► Query Rewriting (optional)
                                    └─► GPT-4o-mini → Answer
```

---

## Two Modes

### Naive RAG
- Fixed-size chunking (1024 tokens, 200 overlap)
- Dense retrieval via cosine similarity
- Direct LLM answer generation

### Advanced RAG
- Layout-aware chunking (tables preserved intact, markdown headers as boundaries)
- Hybrid Search: Vector + BM25 fused via **Reciprocal Rank Fusion (RRF)**
- Cross-encoder reranking with **BAAI/bge-reranker-v2-m3**
- Query Rewriting pre-retrieval technique (LLM rephrases the query in Russian for better recall)

---

## Framework & Library Choices

### PDF Parsing — LlamaParse
**Chosen over:** PyPDF2, pdfplumber, PDFMiner, DocLing, Unstructured

LlamaParse was selected because the source documents contain complex layouts — multi-column blocks, nested financial tables, and diagrams — which plain text extractors fail to parse correctly. LlamaParse outputs clean Markdown with tables converted to pipe format, which is critical for downstream chunking and retrieval accuracy.

### Orchestration — LangChain
**Chosen over:** LlamaIndex, custom wiring

LangChain provides `RecursiveCharacterTextSplitter`, `Document`, `ChatOpenAI`, and `HuggingFaceEmbeddings` with a unified interface. It allowed rapid composition of chunking, embedding, retrieval, and generation steps without boilerplate. The `langchain-qdrant` integration works seamlessly with Qdrant's vector store.

### Vector Store — Qdrant (in-memory)
**Chosen over:** ChromaDB, FAISS, Pinecone

Qdrant was selected for its native support of cosine distance, efficient in-memory mode for experimentation, and a clean Python client (`qdrant-client`). The `langchain-qdrant` wrapper integrates directly with LangChain's `VectorStore` interface. In-memory mode eliminates persistence overhead during hyperparameter sweeps.

### Embeddings — intfloat/multilingual-e5-large
**Chosen over:** OpenAI text-embedding-ada-002, BAAI/bge-m3, sentence-transformers/paraphrase-multilingual

`multilingual-e5-large` is specifically designed for multilingual retrieval with strong Russian-language performance. It uses asymmetric query/passage prefixes (`query: ...` / `passage: ...`) which are applied automatically in the custom `E5Embeddings` wrapper. Embeddings are normalized for cosine similarity.

For large tables (>2048 tokens), **BAAI/bge-m3** (8192-token context) is used as an alternative.

### BM25 — rank_bm25 (BM25Okapi)
**Chosen over:** Elasticsearch, Whoosh, custom TF-IDF

`rank_bm25` is a lightweight, dependency-free implementation of BM25Okapi that runs fully in-memory. Combined with dense retrieval via **Reciprocal Rank Fusion (RRF)**, it dramatically improves recall for exact names, numbers, and financial figures that embeddings tend to miss.

### Reranker — BAAI/bge-reranker-v2-m3
**Chosen over:** cross-encoder/ms-marco, Cohere Rerank API

`bge-reranker-v2-m3` is a multilingual cross-encoder that scores `(query, passage)` pairs jointly, providing more accurate relevance signals than bi-encoder cosine similarity alone. It runs locally via `FlagEmbedding` with FP16 for speed.

### LLM — GPT-4o-mini (OpenAI API)
**Chosen over:** Qwen 2.5 (local), GPT-4o, Claude

GPT-4o-mini provides an optimal cost/quality tradeoff for answer generation in Russian. The system prompt instructs the model to answer strictly from the provided context, cite specific figures and facts, and respond in Russian.

### Evaluation — RAGAS
**Chosen over:** manual scoring, TruLens, DeepEval

RAGAS provides four reference-based metrics aligned with RAG quality:
- **Faithfulness** — is the answer grounded in retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Recall** — does retrieved context cover the ground truth?
- **Context Precision** — is the retrieved context ranked well?

Evaluation is run against a **golden dataset of 30 QA pairs** covering both documents.

---

## Chunking Strategies Compared

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Fixed** | Token-based splits (1024 tokens, 200 overlap) | Uniform text blocks |
| **Recursive** | Markdown-aware splits (`##`, `###`, `\n\n`, `\n`) | Structured narrative text |
| **Layout-Aware** | Header-boundary splits, tables kept intact | Financial tables, mixed layouts |

---

## Experiments (Task 2)

6+ experiments following greedy search (one hyperparameter changed at a time, best carried forward):

| # | Variable | Values Tested |
|---|----------|--------------|
| 1 | Chunk size | 512 / **1024** / 2048 |
| 2 | Chunk overlap | 0 / 100 / **200** |
| 3 | Chunking strategy | fixed / recursive / **layout_aware** |
| 4 | Top-K | 3 / **5** / 10 |
| 5 | Alpha (hybrid weight) | 0.3 / **0.5** / 0.7 / 1.0 |
| 6 | Reranking | off / **on** |
| 7 | Embedding model | multilingual-e5-large / bge-m3 |

---

## Stack Summary

| Component | Choice | Why |
|-----------|--------|-----|
| PDF Parsing | **LlamaParse** | Handles complex layouts, outputs clean Markdown with tables |
| Orchestration | **LangChain** | Unified interface for splitting, embedding, retrieval, generation |
| Vector Store | **Qdrant** (in-memory) | Fast cosine search, clean Python client, LangChain integration |
| Embeddings | **intfloat/multilingual-e5-large** | Strong Russian/multilingual retrieval, asymmetric E5 prefixes |
| Sparse Retrieval | **rank_bm25** (BM25Okapi) | Exact match for names, numbers, financial figures |
| Fusion | **Reciprocal Rank Fusion (RRF)** | Combines dense + sparse rankings without score normalization |
| Reranker | **BAAI/bge-reranker-v2-m3** | Multilingual cross-encoder, runs locally via FlagEmbedding |
| LLM | **GPT-4o-mini** | Cost-efficient, strong Russian output, grounded generation |
| Evaluation | **RAGAS** | Reference-based metrics: Faithfulness, Relevancy, Recall, Precision |
| Tokenization | **tiktoken** (cl100k_base) | Accurate token counting for chunk size control |

---

## Project Structure

```
rag/
├── data/
│   ├── raw/                    # Source PDFs
│   ├── parsed/                 # LlamaParse cache (gitignored)
│   └── golden_dataset.json     # 30 QA pairs for RAGAS evaluation
├── src/
│   ├── config.py               # API keys, paths, default hyperparameters
│   ├── parsing.py              # LlamaParse wrapper with caching
│   ├── chunking.py             # Fixed, recursive, layout-aware strategies
│   ├── embeddings.py           # E5Embeddings with query/passage prefixes
│   ├── vectorstore.py          # Qdrant collection creation and indexing
│   ├── retrieval.py            # Dense, BM25Okapi, hybrid RRF, cross-encoder reranking
│   ├── generation.py           # GPT-4o-mini answer generation
│   ├── pipeline.py             # RAGPipeline class wiring all components
│   └── evaluation.py           # RAGAS evaluation wrapper
└── notebooks/
    ├── task1a_naive_rag.ipynb      # Naive RAG: parse → chunk → index → query
    ├── task1b_advanced_rag.ipynb   # Advanced RAG: hybrid search, reranking, query rewriting
    ├── task2a_experiments.ipynb    # 6+ hyperparameter experiments with RAGAS scores
    └── task2b_ragas_analysis.ipynb # Analysis: metric tables, per-experiment conclusions
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys:
# OPENAI_API_KEY=...
# LLAMA_CLOUD_API_KEY=...
# HF_TOKEN=...
```

Run notebooks in order: `task1a` → `task1b` → `task2a` → `task2b`
