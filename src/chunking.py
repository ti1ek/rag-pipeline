import re
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _token_length(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_fixed(text: str, source: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> list[Document]:
    """Fixed-size chunking using token count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source, "strategy": "fixed"}) for c in chunks]


def chunk_recursive(text: str, source: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> list[Document]:
    """Recursive chunking with markdown-aware separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source, "strategy": "recursive"}) for c in chunks]


def chunk_layout_aware(text: str, source: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> list[Document]:
    """Layout-aware chunking: split on markdown headers/table boundaries, keep tables intact."""
    # Split into sections by headers and table blocks
    sections = re.split(r"(?=\n#{1,4} )", text)

    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", " ", ""],
    )

    for section in sections:
        if not section.strip():
            continue

        # Check if section contains a table (markdown pipe tables)
        table_blocks = re.split(r"(\n\|.+\|(?:\n\|.+\|)*)", section)

        for block in table_blocks:
            if not block.strip():
                continue

            # If block is a table and fits in chunk_size, keep it intact
            if block.strip().startswith("|") and _token_length(block) <= chunk_size:
                if len(block.strip()) < 50:
                    continue
                documents.append(Document(
                    page_content=block.strip(),
                    metadata={"source": source, "strategy": "layout_aware", "is_table": True},
                ))
            else:
                # Split normally
                chunks = splitter.split_text(block)
                for c in chunks:
                    if len(c.strip()) < 50:
                        continue
                    documents.append(Document(
                        page_content=c,
                        metadata={"source": source, "strategy": "layout_aware", "is_table": False},
                    ))

    return documents


CHUNKING_STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "layout_aware": chunk_layout_aware,
}


def chunk_text(text: str, source: str, strategy: str = "fixed", **kwargs) -> list[Document]:
    """Chunk text using the specified strategy."""
    return CHUNKING_STRATEGIES[strategy](text, source, **kwargs)
