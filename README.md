# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that combines document retrieval with LLM capabilities to provide accurate, context-aware answers. The system uses Milvus as a vector database and Ollama for local LLM inference.

## Overview

This project implements a RAG pipeline that:
- Ingests documents (PDF and text files) and creates vector embeddings
- Stores embeddings in Milvus vector database for efficient similarity search
- Retrieves relevant document chunks based on user queries
- Generates answers using Ollama LLM with the retrieved context

## Features

- **Document Loading**: Support for PDF and TXT file formats
- **Flexible Text Chunking**: 
  - Recursive character-based splitting
  - Semantic chunking using embeddings for more meaningful splits
- **Vector Storage**: Milvus vector database with Docker Compose setup
- **Local LLM**: Ollama integration for privacy-focused, offline inference
- **CLI**: Ask questions in real-time with execution time tracking
- **Testing**: Automated test runner with multiple validation strategies
- **Custom Logging**: Configurable logging

## Workflow

```
                        Documents (PDF/TXT)
                                 │
                                 ▼
                        Text Splitter (Recursive/Semantic)
                                 │
                                 ▼
                    ┌─► Embeddings (Ollama)
                    │            │
                    │            ▼
                    │   Milvus Vector Database
                    │            │
                    │            ▼
                    └─  Ollama LLM (llama3.1) ◄── Question
                                 │
                                 ▼
                               Answer
```

## Prerequisites

- Python 3.12+
- Docker / Podman
- Ollama with required models installed (embedding model & LLM model)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG
```

### 2. Install Python dependencies

```bash
pip install langchain langchain-ollama langchain-milvus langchain-experimental
pip install pypdf colorama
```

### 3. Create volumes for Database Persistence
```bash
    mkdir volumes/etcd, volumes/milvus, volumes/minio
```

### 4. Start Milvus vector database

With Docker Compose:
```bash
docker compose up -d
```
or Podman Compose:
```bash
podman compose up -d
```

This starts:
- **etcd**: Distributed key-value store for Milvus metadata
- **MinIO**: Object storage for Milvus data
- **Milvus Standalone**: Vector database server (port 19530)

### 4. Install Ollama models

```bash
ollama pull llama3.1
ollama pull embeddinggemma
```

## Configuration

Default configuration in `CustomRag.py`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `DEFAULT_MODEL` | `llama3.1` | LLM model for answer generation |
| `DEFAULT_EMBEDDING_MODEL` | `embeddinggemma` | Model for creating embeddings |
| `DEFAULT_BASE_URL` | `localhost:11434` | Ollama server URL |
| `DEFAULT_COLLECTION_NAME` | `rag_collection` | Milvus collection name |
| `DEFAULT_MILVUS_URI` | `http://localhost:19530` | Milvus connection URI |

## Usage

### Interactive Mode

```bash
python main.py
```

This starts an interactive session where you can ask questions about the loaded documents.

### Loading Documents

Before asking questions, you need to load documents into the vector store. You can do this using the following code snippet:

```python
rag = CustomRag()
rag.load_text_files(path="<path-to-txt-files>")
rag.load_pdf_files(path="<path-to-pdf-files>", use_semantic=True)
```

### Running Tests

```bash
python test.py
```

The test runner validates answers against:
- **Context-based validation**: Checks if the answer aligns with retrieved documents
- **Expected answer validation**: Compares against predefined correct answers
- **Keyword validation**: Verifies presence of expected keywords

### Test file structure

Test questions are stored in `tests/questions/` as JSON files. Each file contains a list of test cases with the following structure:

```json
[
    {
        "question": "<question>",
        "expected_answer": "<expected-answer-for-asked-question>",
        "keywords": ["<expected-keyword1>","<expected-keyword2>"]
    },
    ...
]
```

## Project Structure

```
RAG/
├── main.py              # Main entry point - interactive CLI
├── CustomRag.py         # Core RAG implementation
├── AdjustedOllama.py    # Ollama LLM wrapper with prompts
├── CustomLogger.py      # Configurable colored logging
├── TestRunner.py        # Implementation of test runner
├── test.py              # Test execution script
├── utils.py             # Utility functions
├── docker-compose.yml   # Milvus infrastructure
├── documents/           # Source documents
│   ├── rfc/             # RFC documents (PDF)
│   └── universe/        # Example files (TXT)
├── tests/
│   ├── questions/       # Test question sets (JSON)
│   └── results/         # Test output files
└── volumes/             # Docker persistent storage
```

## Text Chunking Strategies

### Recursive Character Splitting
Default chunking with configurable parameters:
- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- Separators: `\n\n`, `\n`, `. `

### Semantic Chunking
Uses embedding similarity to create semantically coherent chunks:
- `breakpoint_threshold_type`: percentile
- `breakpoint_threshold_amount`: 85

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| Milvus | 19530 | Vector database API |
| Milvus Health | 9091 | Health check endpoint |
| MinIO API | 9000 | Object storage API |
| MinIO Console | 9001 | MinIO web interface |

