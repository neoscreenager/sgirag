import hashlib
from pathlib import Path
from typing import Iterable

import pdfplumber


def sha256_for_path(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def list_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted([path for path in pdf_dir.glob("*.pdf") if path.is_file()])


def load_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = []
        for page in pdf.pages:
            # Extract text
            text = page.extract_text() or ""
            # Extract tables and format them as readable text
            tables = page.extract_tables()
            table_texts = []
            for table in tables:
                if table:
                    # Convert table rows to a simple text format
                    table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                    table_texts.append(table_text)
            # Combine text and tables
            page_content = text
            if table_texts:
                page_content += "\n\nTables:\n" + "\n\n".join(table_texts)
            pages.append(page_content.strip())
        return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if not text:
        return []

    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def iter_pdf_chunks(path: Path, chunk_size: int = 1000, overlap: int = 150) -> Iterable[tuple[str, int, str]]:
    text = load_pdf_text(path)
    full_hash = sha256_for_path(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for index, chunk in enumerate(chunks, start=1):
        yield full_hash, index, chunk
