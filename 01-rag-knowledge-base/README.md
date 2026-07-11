# RAG Knowledge Base

A Retrieval-Augmented Generation knowledge base that answers questions about your documents with inline citations — fully offline by default, with pluggable cloud backends.

```
                          ┌──────────────────────────────────────────────┐
                          │                RAGPipeline                   │
                          │                                              │
 .txt / .md / .pdf ──────►│ Ingestion ──► Chunking ──► Embedder ──┐      │
                          │  (loaders)   (fixed /     (hash /     │      │
                          │              sentence)    ST / OpenAI)│      │
                          │                                       ▼      │
                          │                                 VectorStore  │
                          │                                 (numpy +     │
                          │                                  persistence)│
                          │                                       │      │
 question ───────────────►│ HybridRetriever ◄─────────────────────┘      │
                          │  dense cosine ─┐                             │
                          │                ├─► RRF fusion ─► top-k       │
                          │  sparse BM25 ──┘                  │          │
                          │                                   ▼          │
 answer + citations ◄─────│ LLMProvider (offline extractive /            │
                          │              OpenAI / Anthropic)             │
                          └──────────────────────────────────────────────┘
                                   exposed via FastAPI + CLI demo
```

## Features

- **Document ingestion** for `.txt` and `.md` (per-section metadata from Markdown headings), plus optional `.pdf` via lazily imported `pypdf`
- **Two chunking strategies**: fixed-size with overlap, and sentence-aware packing that never splits a sentence
- **Pluggable embeddings** behind an `Embedder` interface — the default is a deterministic hashed bag-of-words embedder (sublinear TF, L2-normalized) that needs no model download or network; SentenceTransformers and OpenAI backends are one env var away
- **Hybrid retrieval**: dense cosine similarity + sparse BM25, combined with weighted reciprocal rank fusion
- **Cited answers** behind an `LLMProvider` interface — the default offline provider extracts the most relevant sentences and tags them `[1]`, `[2]`, each mapped back to its source file and section
- **In-memory vector store** with npz/json persistence, a **FastAPI** service, and a self-contained **CLI demo**
- Fast, offline, deterministic test suite (45 tests) — no API keys required anywhere

## Quickstart

Requires Python 3.11+.

```bash
pip install -r requirements.txt

# Run the offline demo (ingests sample_data/ and answers 3 questions)
python -m rag_kb.demo

# Start the API
uvicorn rag_kb.api:app --host 0.0.0.0 --port 8000

# Use it
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
     -d '{"path": "sample_data"}'
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question": "How do I roll back a bad deployment?"}'
curl localhost:8000/health
```

With Docker:

```bash
docker build -t rag-kb .
docker run --rm -p 8000:8000 rag-kb
```

## Example output

Real output from `python -m rag_kb.demo` against the bundled sample corpus (a fictional company's engineering handbook):

```
Ingested 20 documents into 20 chunks.

Q: How do I roll back a bad deployment?
A: To roll back a bad deployment, run `helios deploy rollback <service>` which
   redeploys the previous known-good release within minutes. [1]
   Anyone can declare an incident by running `/incident declare` in the
   `#eng-incidents` channel. [2]
   ...
Sources:
  [1] 03-deployments.md > Rolling back
  [2] 04-incident-response.md > Declaring an incident
  [3] 02-code-review.md > What reviewers look for
  [4] 02-code-review.md > Approval rules
```

## Configuration

All settings are environment variables with the `RAG_KB_` prefix (see `.env.example`). Defaults run fully offline.

| Variable | Default | Description |
|---|---|---|
| `RAG_KB_CHUNK_STRATEGY` | `sentence` | `fixed` (character windows) or `sentence` (sentence-aware packing) |
| `RAG_KB_CHUNK_SIZE` | `800` | Maximum chunk length in characters |
| `RAG_KB_CHUNK_OVERLAP` | `120` | Overlap between consecutive fixed-size chunks |
| `RAG_KB_EMBEDDER` | `hash` | `hash`, `sentence-transformers`, or `openai` |
| `RAG_KB_EMBEDDING_DIM` | `512` | Dimensionality of the hashed embedder |
| `RAG_KB_TOP_K` | `4` | Chunks retrieved per query |
| `RAG_KB_CANDIDATE_POOL` | `20` | Candidates fetched per retriever before fusion |
| `RAG_KB_FUSION_WEIGHT` | `0.5` | Dense weight in RRF; `0` = BM25 only, `1` = dense only |
| `RAG_KB_RRF_K` | `60` | Rank-offset constant for reciprocal rank fusion |
| `RAG_KB_LLM_PROVIDER` | `offline` | `offline`, `openai`, or `anthropic` |

The cloud backends additionally read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` through their SDKs and require `pip install openai` / `pip install anthropic`; those packages are imported lazily and only when selected.

## Project structure

```
01-rag-knowledge-base/
├── rag_kb/
│   ├── config.py        # pydantic-settings configuration (RAG_KB_* env vars)
│   ├── schemas.py       # shared pydantic models (chunks, requests, citations)
│   ├── ingestion.py     # .txt/.md/.pdf loaders with source/section/page metadata
│   ├── chunking.py      # fixed-size and sentence-aware chunkers
│   ├── embeddings.py    # Embedder interface + hash / ST / OpenAI backends
│   ├── vector_store.py  # numpy cosine store with npz/json persistence
│   ├── retrieval.py     # BM25 + dense hybrid retrieval with RRF
│   ├── llm.py           # LLMProvider interface + offline extractive synthesizer
│   ├── pipeline.py      # RAGPipeline: ingest -> retrieve -> answer
│   ├── api.py           # FastAPI app (/ingest, /query, /health)
│   └── demo.py          # offline CLI demo
├── sample_data/         # fictional engineering handbook (5 markdown docs)
├── tests/               # 45 offline, deterministic pytest tests
├── Dockerfile           # python:3.11-slim, non-root, serves the API
├── Makefile             # install / test / lint / demo / run targets
└── requirements.txt     # CPU-only runtime deps
```

## Design decisions

- **Hybrid retrieval by default.** Dense embeddings capture paraphrases ("roll back" vs "revert") while BM25 nails exact identifiers and rare terms that hashing or small embedding models miss. Reciprocal rank fusion combines the two using only ranks, so no score calibration between the incompatible scales is needed, and one `fusion_weight` knob tunes the balance.
- **Provider abstractions everywhere it hurts.** Embeddings and answer synthesis sit behind small interfaces (`Embedder`, `LLMProvider`). The defaults are deterministic and offline, which makes the entire test suite fast and key-free; swapping in SentenceTransformers or a hosted LLM is a config change, not a refactor. Optional SDKs are imported lazily so the core package has no heavyweight dependencies.
- **Deterministic offline baseline.** The hashed embedder and extractive synthesizer trade quality for reproducibility on purpose: identical inputs always produce identical answers and rankings, which keeps tests meaningful and demos honest.
- **Simple storage on purpose.** At the scale of a knowledge base (thousands of chunks), a normalized numpy matrix and a dot product beat the operational cost of an external vector database. Persistence is plain npz + json, so an index can be inspected with standard tools.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q          # 45 tests, all offline and deterministic
ruff check .       # lints clean
```

The suite covers chunking edge cases (empty text, oversized sentences, invalid overlap), embedder determinism, vector store save/load round-trips, RRF fusion ordering and weight extremes, citation mapping, and all API endpoints via `TestClient`.

## Roadmap

- Cross-encoder reranking stage after fusion for higher precision at small `top_k`
- Streaming answers from the cloud providers via server-sent events
- Incremental BM25 index updates instead of rebuilding after each ingest
- Evaluation harness (retrieval hit-rate and answer faithfulness on a labeled set)
