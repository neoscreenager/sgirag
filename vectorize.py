import os
from pathlib import Path
from typing import Iterable

import chromadb

from local_models import LocalLMStudioClient
from pdf_utils import iter_pdf_chunks, list_pdf_files, sha256_for_path


CHROMA_COLLECTION_NAME = "pdf_documents"


def get_chroma_client(persist_directory: Path) -> chromadb.Client:
    return chromadb.PersistentClient(path=str(persist_directory))


def normalize_source(path: Path, root: Path) -> str:
    return str(path.relative_to(root).as_posix())


def get_existing_sources(collection: chromadb.api.models.Collection.Collection) -> dict[str, dict[str, list[str]]]:
    data = collection.get(include=["metadatas", "documents"])
    result: dict[str, dict[str, list[str]]] = {}
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []
    for metadata, record_id in zip(metadatas, ids):
        source = metadata.get("source")
        digest = metadata.get("hash")
        if not source:
            continue
        entry = result.setdefault(source, {"hash": None, "ids": []})
        if digest:
            entry["hash"] = digest
        entry["ids"].append(record_id)
    return result


def add_pdf_to_collection(
    collection: chromadb.api.models.Collection.Collection,
    source: str,
    filename: str,
    file_hash: str,
    chunks: Iterable[tuple[str, int, str]],
    embed_client: LocalLMStudioClient,
) -> int:
    ids = []
    documents = []
    metadatas = []
    texts = []

    for chunk_hash, index, text in chunks:
        chunk_id = f"{source}::{index}"
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append({"source": source, "filename": filename, "hash": file_hash, "chunk_index": index})
        texts.append(text)

    if not texts:
        return 0

    embeddings = embed_client.embed_texts(texts)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def vectorize_pdfs(pdf_dir: Path, chroma_dir: Path, base_url: str | None = None, force_reindex: bool = False) -> dict[str, int]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    embed_client = LocalLMStudioClient(base_url=base_url)
    client = get_chroma_client(chroma_dir)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    indexed_sources = get_existing_sources(collection)
    processed = {"added": 0, "updated": 0, "skipped": 0}

    for pdf_file in list_pdf_files(pdf_dir):
        source = normalize_source(pdf_file, pdf_dir)
        file_hash = sha256_for_path(pdf_file)
        source_entry = indexed_sources.get(source)

        if not force_reindex and source_entry and source_entry.get("hash") == file_hash:
            processed["skipped"] += 1
            continue

        if source_entry and source_entry.get("ids"):
            collection.delete(ids=source_entry["ids"])
            action = "updated"
        else:
            action = "added"

        chunks = list(iter_pdf_chunks(pdf_file))
        if not chunks:
            continue

        add_pdf_to_collection(collection, source, pdf_file.name, file_hash, chunks, embed_client)
        processed[action] += 1

    return processed


if __name__ == "__main__":
    root = Path(__file__).parent
    print(f"root: {root}")
    pdf_dir = f"{root}\\pdfdata"
    chroma_dir = f"{root}\\chromadbrag"
    print("Starting PDF vectorization...")
    print(f"PDF source folder: {pdf_dir}")
    print(f"Chromadb persistence: {chroma_dir}")

    result = vectorize_pdfs(pdf_dir=Path(pdf_dir), chroma_dir=Path(chroma_dir), force_reindex=True)
    print("Vectorization result:", result)
