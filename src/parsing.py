import os
from llama_parse import LlamaParse
from src.config import LLAMA_CLOUD_API_KEY, RAW_DIR, PARSED_DIR


def parse_pdf(filename: str, use_cache: bool = True) -> str:
    """Parse a PDF using LlamaParse, with markdown file caching."""
    cache_path = os.path.join(PARSED_DIR, f"{os.path.splitext(filename)[0]}.md")
    os.makedirs(PARSED_DIR, exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        language="ru",
    )

    pdf_path = os.path.join(RAW_DIR, filename)
    documents = parser.load_data(pdf_path)
    text = "\n\n".join(doc.text for doc in documents)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def parse_all_pdfs(use_cache: bool = True) -> dict[str, str]:
    """Parse all PDFs in data/raw/. Returns {filename: markdown_text}."""
    results = {}
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            results[filename] = parse_pdf(filename, use_cache=use_cache)
    return results
