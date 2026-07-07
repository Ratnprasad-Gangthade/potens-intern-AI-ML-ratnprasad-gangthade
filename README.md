# Potens RAG Pipeline
A modular LangChain + FAISS RAG pipeline for analyzing PDF research papers with Gemini.

## Overview
Potens RAG Pipeline loads PDFs, chunks document text, embeds the chunks, stores them in FAISS, and answers questions using only retrieved context.
The project supports grounded question answering, citations, hallucination prevention, contradiction analysis, and multilingual queries.

## Features
| Feature | Description |
| --- | --- |
| PDF ingestion | Loads `.pdf` files from a documents folder. |
| Local vector search | Stores embedded chunks in a FAISS index. |
| RAG question answering | Answers using retrieved chunks only. |
| Citations | Returns source file, page, chunk ID, and snippet. |
| Hallucination prevention | Checks context sufficiency before answering. |
| Contradiction analysis | Compares two documents on a specific topic. |
| Multilingual support | Translates non-English queries for retrieval and translates answers back. |

## Project Structure
```text
.
|-- Documents/
|-- rag_flow/
|   |-- __init__.py
|   |-- contradiction.py
|   |-- document_loader.py
|   |-- language_handling.py
|   |-- qa_engine.py
|   `-- vector_store.py
|-- rag_pipeline.py
|-- requirements.txt
|-- .env
`-- README.md
```
| Path | Purpose |
| --- | --- |
| `rag_flow/document_loader.py` | PDF loading. |
| `rag_flow/vector_store.py` | Chunking, embeddings, and FAISS creation. |
| `rag_flow/qa_engine.py` | QA, sufficiency checks, multilingual handling, and citations. |
| `rag_flow/contradiction.py` | Document comparison and contradiction analysis. |
| `rag_flow/language_handling.py` | Language detection and translation. |
| `rag_pipeline.py` | Backward-compatible exports. |

## Tech Stack
| Tool | Use |
| --- | --- |
| Python | Core implementation. |
| LangChain | PDF loading, splitting, model wrappers, and FAISS integration. |
| FAISS | Local vector similarity search. |
| Google Gemini | LLM and embedding provider. |
| `gemini-2.5-flash` | QA, translation, sufficiency checks, and comparison. |
| `gemini-embedding-2-preview` | Text embeddings. |
| `python-dotenv` | Loads `.env` variables. |
| `pypdf` | PDF parsing backend. |

## RAG Pipeline
The pipeline follows a retrieval-first workflow:
1. Load PDFs from `Documents/`.
2. Split documents into overlapping chunks.
3. Assign a `chunk_id` to each chunk.
4. Generate Gemini embeddings.
5. Store chunks in a FAISS index.
6. Translate non-English queries to English when needed.
7. Retrieve top-k relevant chunks.
8. Check whether the context is sufficient.
9. Generate an answer only from retrieved context.
10. Translate the final answer back when needed.
11. Return the answer with citations.

### Chunking Strategy
- Splitter: `RecursiveCharacterTextSplitter`
- Chunk size: `1000`
- Chunk overlap: `200`
- Each chunk receives a `chunk_id`.
- Chunk metadata is reused for citations and document filtering.

### Embedding Strategy
- Embedding model: `gemini-embedding-2-preview`
- Vector store: FAISS
- Saved index path: `faiss_index`

### Retrieval Strategy
- QA uses FAISS similarity search.
- Default QA retrieval depth: `k=2`
- Contradiction analysis retrieves chunks separately from each selected document.
- Document comparison filters by `metadata["source"]`.
- The system does not compare entire PDFs directly.

## Hallucination Prevention
Before answer generation, the pipeline checks whether retrieved chunks contain enough information.
If context is insufficient, unrelated, or incomplete, it returns: `The uploaded research papers do not contain enough information to answer this question.`
For non-English queries, this message is translated into the user's language.
The answer prompt also instructs Gemini to:
- Use only the provided context.
- Avoid outside knowledge.
- Avoid guessing.
- Refuse unsupported answers.

## Citation Strategy
Each answer includes citations from retrieved chunks.
Citation fields:
- `source_file`
- `page`
- `chunk_id`
- `snippet`
- `source_file` is returned as a file basename.
- Snippets come from retrieved chunks.
- Citations are not translated.
- File names, pages, chunk IDs, and snippets stay unchanged for multilingual queries.

## Contradiction Analysis
The project supports RAG-based comparison between two documents.
The comparison flow:
- Retrieve top-k chunks from Document A.
- Retrieve top-k chunks from Document B.
- Build a comparison prompt from retrieved chunks only.
- Ask Gemini to classify the relationship.
Supported verdicts: `Agree`, `Partially Agree`, `Contradict`, `Have Insufficient Information`.
If relevant context is missing from either document, the verdict is insufficient.
If Gemini returns invalid comparison JSON, the pipeline returns an insufficient-information verdict with an error explanation.

## Multilingual Support
The QA flow supports non-English user questions.
Workflow:
- Detect the query language with Gemini.
- Translate only the query to English when needed.
- Retrieve and generate using the English query.
- Translate only the final answer back to the original language.
- Leave citations unchanged.

## Installation
```bash
python -m venv .venv
pip install -r requirements.txt
```
Windows PowerShell activation:
```powershell
.\.venv\Scripts\Activate.ps1
```
Create `.env`:
```env
GOOGLE_API_KEY=your_api_key_here
```
Place PDF files inside `Documents/`.

## Usage
Build the vector database:
```python
from rag_pipeline import build_vector_db
vector_db = build_vector_db("Documents")
```
Ask a question:
```python
from rag_pipeline import answer_query
response = answer_query(vector_db, "What method does the paper propose?")
print(response["answer"])
print(response["citations"])
```
Compare two documents:
```python
from rag_pipeline import compare_documents
comparison = compare_documents(
    vector_db,
    "Documents/paper_a.pdf",
    "Documents/paper_b.pdf",
    "model performance"
)
print(comparison["verdict"])
print(comparison["explanation"])
```

## Design Decisions
- The pipeline is split into small modules for document loading, vector storage, QA, language handling, and contradiction analysis.
- FAISS is used as a local vector store to keep retrieval simple and offline after indexing.
- Gemini is reused for generation, translation, sufficiency checks, and comparison to avoid extra model dependencies.
- `rag_pipeline.py` remains as a compatibility layer for existing imports.

## Current Limitations / Unfinished Work
- There is no helper yet for loading an already saved FAISS index from `faiss_index`.
- Automated tests are not included.
- Citation-building logic is repeated in QA and contradiction modules.
- Error handling for missing environment variables, empty folders, and failed model calls is minimal.

## Future Improvements
- Add a FAISS index loading helper.
- Add focused tests for QA, citations, multilingual handling, and contradiction parsing.
- Extract shared citation formatting into a small helper.
- Improve validation and error messages for setup and runtime failures.
