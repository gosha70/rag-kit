# ragkit

A document ingestion and embedding pipeline for RAG (Retrieval-Augmented Generation) applications. Converts documents into chunked vector embeddings stored in a Chroma vectorstore, with support for multiple embedding providers and 16 file formats.

## Features

- **Multi-format document ingestion** -- CSV, DDL, Excel, Java, JavaScript, JSON, HTML, Markdown, PDF, Python, RTF, SQL, Text, XML, XSL, YAML
- **Pluggable embedding models** -- HuggingFace Instructor, LLama CPP, OpenAI, Ollama, GPT4All
- **Chroma vectorstore** with async batch processing
- **File watcher** for continuous ingestion from a watched directory
- **Configurable chunking** with per-format text splitters

## Project Structure

```
api/                  # CLI entry point for document splitting
embeddings/           # Embedding model abstraction + provider configs
rag/
  embeddings/chroma/  # Chroma vectorstore service + CLI entry point
  file_watcher.py     # Watchdog-based file monitor
unstructured/
  converters/         # Per-format document converters
  document_splitter.py
  file_type.py        # Supported file type enum
vectorstores/         # Document store service (load, save, zip)
tests/
```

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
pip install pytest  # for running tests
```

For OpenAI embeddings, export `OPENAI_API_KEY` as an environment variable before running. The production code reads it via `os.getenv`; `.env` files are not automatically loaded.

## Usage

### Split documents to disk

Loads documents from a directory, splits them into chunks, and saves the splits as JSON files.

```bash
# Split all supported file types in a directory
python -m api.ragkit_app --dir_path ./docs

# Split specific file types with glob patterns
python -m api.ragkit_app \
  --dir_path ./src \
  --file_types java py \
  --file_patterns "java:**/*Service*" "py:**/*test*" \
  --persist_directory ./output/splits
```

| Argument | Description | Default |
|---|---|---|
| `--dir_path` | Root directory to search | `.` |
| `--file_types` | File extensions to process (space-separated, no dot) | all supported |
| `--file_patterns` | Glob patterns per type (`type:pattern`) | `**/*` |
| `--persist_directory` | Output directory for split JSON files | auto-generated |

### Create embedding vectorstore

Builds a Chroma vectorstore from documents, pre-split JSON files, or a zip archive.

```bash
# From raw documents
python -m rag.embeddings.chroma.embedding_chroma_service \
  --dir_path ./docs \
  --file_types md html \
  --model_name "hkunlp/instructor-large" \
  --persist_directory ./output/vectorstore

# From pre-split JSON files
python -m rag.embeddings.chroma.embedding_chroma_service \
  --splits_directory ./output/splits \
  --model_name "hkunlp/instructor-large" \
  --persist_directory ./output/vectorstore

# From a zip archive of splits
python -m rag.embeddings.chroma.embedding_chroma_service \
  --zip_file ./splits.zip \
  --persist_directory ./output/vectorstore

# Test the vectorstore after creation
python -m rag.embeddings.chroma.embedding_chroma_service \
  --dir_path ./docs \
  --persist_directory ./output/vectorstore \
  --test_question "How does authentication work?"
```

| Argument | Description | Default |
|---|---|---|
| `--dir_path` | Root directory to search | `.` |
| `--zip_file` | Zip file with pre-split documents | -- |
| `--splits_directory` | Directory with pre-split JSON | -- |
| `--file_types` | File extensions (space-separated) | all supported |
| `--file_patterns` | Glob patterns per type | `**/*` |
| `--model_name` | Embedding model name | `hkunlp/instructor-large` |
| `--collection_name` | Chroma collection name | `EGOGE_DOCUMENTS_DB` |
| `--persist_directory` | Vectorstore output path | -- |
| `--test_question` | Query to validate the vectorstore | -- |

## Supported Embedding Models

| Provider | Models |
|---|---|
| HuggingFace Instructor | `hkunlp/instructor-large` (default), `hkunlp/instructor-xl`, `hkunlp/instructor-base` |
| LLama CPP | `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat` |
| OpenAI | `openai/text-embedding-ada-002`, `openai/text-embedding-babbage-001`, `openai/text-embedding-curie-001` |
| Ollama | `ollama-7b`, `ollama-13b`, `ollama-30b` |
| GPT4All | `gpt4all-lora-quantized`, `gpt4all-mpt-7b`, `gpt4all-j-6b` |

Model names must match the keys in `embeddings/models_constants.py` exactly. Unrecognized names silently fall back to the default (`hkunlp/instructor-large`).

## Tests

```bash
python -m pytest tests/
```

## License

CC-BY-SA-4.0
