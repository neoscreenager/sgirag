import os
from pathlib import Path
from typing import Any

import streamlit as st
import chromadb

from local_models import LocalLMStudioClient
from vectorize import vectorize_pdfs, CHROMA_COLLECTION_NAME, get_chroma_client
from pdf_utils import list_pdf_files


def get_base_url() -> str:
    base_url = os.getenv("LOCAL_LM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def initialize_chroma(persist_directory: Path) -> chromadb.Client:
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def render_sidebar(client: LocalLMStudioClient, index_status: dict[str, Any]) -> None:
    st.sidebar.title("Local PDF RAG")
    st.sidebar.markdown("Query PDF documents with a locally hosted LM Studio model.")
    st.sidebar.markdown(f"**Base URL:** `{get_base_url()}`")
    st.sidebar.markdown(f"**LLM model:** `{client.llm_model_id}`")
    st.sidebar.markdown(f"**Embedding model:** `{client.embedding_model_id}`")
    st.sidebar.markdown(f"**Langfuse tracing:** `{ 'enabled' if client.tracing_enabled else 'disabled' }`")
    st.sidebar.divider()
    st.sidebar.subheader("Index status")
    st.sidebar.write(f"Documents indexed: **{index_status.get('documents', 0)}**")
    st.sidebar.write(f"Added this run: **{index_status.get('added', 0)}**")
    st.sidebar.write(f"Skipped: **{index_status.get('skipped', 0)}**")
    if st.sidebar.button("Rebuild index"):
        st.session_state.rebuild = True

    st.sidebar.divider()
    st.sidebar.subheader("Langfuse status")
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    st.sidebar.write(f"Tracing enabled: **{'yes' if client.tracing_enabled else 'no'}**")
    st.sidebar.write(f"Langfuse base URL: **{langfuse_base_url or 'not set'}**")
    st.sidebar.write(f"Auth configured: **{'yes' if langfuse_public_key and langfuse_secret_key else 'no'}**")


def query_collection(collection: chromadb.api.models.Collection.Collection, client: LocalLMStudioClient, question: str) -> dict[str, Any]:
    embedding = client.embed_text(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    found = []
    context_parts = []

    if metadatas:
        for item_metadata, item_documents in zip(metadatas[0], documents[0]):
            source = item_metadata.get("source")
            filename = item_metadata.get("filename")
            context_parts.append(item_documents)
            if source and filename:
                found.append({"source": source, "filename": filename})

    try:
        answer = client.generate_answer(question, "\n\n".join(context_parts))
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
    return {"answer": answer, "sources": found}


def download_button_for_pdf(pdf_path: Path, label: str) -> None:
    if not pdf_path.exists():
        st.error(f"Missing file: {pdf_path.name}")
        return
    with pdf_path.open("rb") as handle:
        data = handle.read()
    st.download_button(label=label, data=data, file_name=pdf_path.name, mime="application/pdf")


def main() -> None:
    st.set_page_config(page_title="Local PDF RAG", layout="wide")
    st.title("Local PDF RAG — Streamlit Chat Interface")

    root = Path(__file__).parent
    pdf_dir = root / "pdfdata"
    chroma_dir = root / "chromadbrag"
    base_url = get_base_url()

    timeout = int(os.getenv("LOCAL_LM_TIMEOUT", "60"))
    client = LocalLMStudioClient(base_url=base_url, timeout=timeout)
    chroma_client = initialize_chroma(chroma_dir)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "rebuild" not in st.session_state:
        st.session_state.rebuild = False

    if st.session_state.rebuild or collection.count() == 0:
        with st.spinner("Indexing PDF documents. This may take a moment..."):
            index_result = vectorize_pdfs(pdf_dir=pdf_dir, chroma_dir=chroma_dir, base_url=base_url)
        st.session_state.rebuild = False
    else:
        index_result = {"documents": collection.count(), "added": 0, "skipped": 0}

    render_sidebar(client, index_result)

    question = st.text_area("Ask a question about your PDF collection", height=140)
    if st.button("Send") and question.strip():
        with st.spinner("Retrieving relevant documents and generating an answer..."):
            result = query_collection(collection, client, question)
        st.session_state.chat_history.append({"role": "user", "text": question})
        st.session_state.chat_history.append({"role": "assistant", "text": result["answer"]})
        st.session_state.last_sources = result["sources"]

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['text']}")
        else:
            st.markdown(f"**Assistant:** {message['text']}")

    if st.session_state.last_sources:
        st.markdown("---")
        st.subheader("Relevant PDF documents")
        seen = set()
        for source_info in st.session_state.last_sources:
            source = source_info["source"]
            filename = source_info["filename"]
            if source in seen:
                continue
            seen.add(source)
            st.write(f"- **{filename}**")
            source_path = pdf_dir / source
            download_button_for_pdf(source_path, label=f"Download {filename}")

    st.markdown("---")
    st.markdown("*Tip: add PDFs to the `pdfdata/` directory and click Rebuild index.*")


if __name__ == "__main__":
    main()
