# RAG Pipeline for Kazakh Company Annual Reports

RAG (Retrieval-Augmented Generation) pipeline for answering questions over two Kazakh company annual reports:
- **ktj.pdf** — KTZh integrated annual report (~368 pages)
- **matnp_2024_rus.pdf** — Maten Petroleum annual report (~30 pages)

## Architecture

```
PDF → LlamaParse (markdown) → Chunking → Embeddings → Qdrant → Retrieval → GPT-4o-mini → Answer
```

**Naive RAG**: Fixed chunking → Dense retrieval → LLM generation

**Advanced RAG**: Layout-aware chunking → Hybrid search (Vector + BM25 via RRF) → Cross-encoder reranking → Query rewriting → LLM generation

## Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | LlamaParse |
| Vector DB | Qdrant (local) |
| Embeddings | intfloat/multilingual-e5-large |
| LLM | GPT-4o-mini |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Evaluation | RAGAS |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY and LLAMA_CLOUD_API_KEY in .env
```

## Project Structure

```
rag/
├── data/
│   ├── raw/                  # PDF files
│   ├── parsed/               # LlamaParse cache (gitignored)
│   └── golden_dataset.json   # 30 QA pairs for evaluation
├── src/
│   ├── config.py             # Settings and default hyperparameters
│   ├── parsing.py            # LlamaParse wrapper
│   ├── chunking.py           # Fixed, recursive, layout-aware strategies
│   ├── embeddings.py         # HuggingFace embedding loader
│   ├── vectorstore.py        # Qdrant collection management
│   ├── retrieval.py          # Dense, BM25, hybrid RRF, reranking
│   ├── generation.py         # GPT-4o-mini answer generation
│   ├── pipeline.py           # RAGPipeline class
│   └── evaluation.py         # RAGAS evaluation wrapper
└── notebooks/
    ├── task1a_naive_rag.ipynb
    ├── task1b_advanced_rag.ipynb
    ├── task2a_experiments.ipynb
    └── task2b_ragas_analysis.ipynb
```

## Running Notebooks

Run notebooks in order:
1. **Task 1A** — Naive RAG: parse PDFs, chunk, index, query
2. **Task 1B** — Advanced RAG: compare chunking, retrieval, reranking
3. **Task 2A** — Experiments: 6+ hyperparameter sweeps with RAGAS evaluation
4. **Task 2B** — Analysis: summary tables, per-metric analysis, conclusions
