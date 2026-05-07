# Local PDF RAG with Streamlit and LM Studio

A production-quality Retrieval-Augmented Generation (RAG) project that uses a locally hosted LM Studio LLM and embeddings API.

## Features

- Streamlit chat interface for querying PDF documents.
- Dynamic model discovery from a locally running LM Studio base URL (`http://127.0.0.1:1234/v1`).
- Locally hosted embedding model automatically selected.
- Persistent Chromadb index in `./chromadbrag`.
- Downloads a PDF document when the document name is clicked.
- Separate PDF vectorization module with incremental updates.

## Project Structure

- `app.py` - Streamlit app for querying the indexed PDF documents.
- `vectorize.py` - PDF ingestion and vectorization logic.
- `local_models.py` - Local LM Studio model discovery and API wrappers.
- `pdf_utils.py` - PDF text extraction and chunking utilities.
- `chromadbrag/` - Persistent Chromadb store.
- `pdfdata/` - Folder for source PDF documents.

## Requirements

- Python 3.10+
- LM Studio running locally and exposing an OpenAI-compatible API at `http://127.0.0.1:1234/v1`.
- PDF files placed in the `pdfdata/` folder.

## Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### 1. Vectorize PDF files

Run the ingestion step to index all PDFs under `pdfdata/`:

```powershell
python vectorize.py
```

This command will only index new or changed PDFs, leaving previously indexed files unchanged.

### 2. Start the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal and ask questions about your PDF data.

## Configuration

The app will use `http://127.0.0.1:1234/v1` by default. To override, set:

```powershell
$env:LOCAL_LM_BASE_URL = "http://127.0.0.1:1234/v1"
```

To enable Langfuse tracing, set your Langfuse endpoint and optional credentials:

```powershell
$env:LANGFUSE_BASE_URL = "http://127.0.0.1:8000"
$env:LANGFUSE_PUBLIC_KEY = "<your-public-key>"
$env:LANGFUSE_SECRET_KEY = "<your-secret-key>"
```

If credentials are omitted, the app still attempts OTLP export to the Langfuse base URL.

## How it Works

1. `vectorize.py` reads `pdfdata/` and converts PDF pages into text chunks.
2. Chunks are embedded using the locally hosted LM Studio embeddings model.
3. Embeddings are stored in a persistent Chromadb collection.
4. `app.py` queries the index, retrieves source documents, and generates answers using the locally hosted LLM.
5. The interface returns the document name and offers a download button.

## Notes

- `vectorize.py` is intentionally separate so PDF ingestion can be triggered independently of the Streamlit UI.
- The index is stored in `chromadbrag/` and is reused across app restarts.

