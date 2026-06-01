<p align="center">
  <h1 align="center">🧬 Clinical Trial RAG</h1>
  <p align="center">
    <strong>Retrieval-Augmented Generation for 23,000+ Clinical Trials</strong>
  </p>
  <p align="center">
    <a href="#-quickstart"><img src="https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="#-quickstart"><img src="https://img.shields.io/badge/FastAPI-0.1.0-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
    <a href="#-quickstart"><img src="https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="#-quickstart"><img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?logo=meta&logoColor=white" alt="LLM"></a>
    <a href="#-quickstart"><img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  </p>
</p>

---

A production-ready **Retrieval-Augmented Generation** system that lets clinicians, researchers, and patients ask natural-language questions against a corpus of **23,000+ clinical trials** (73,012 chunks). The pipeline fuses dense and sparse retrieval, re-ranks with a cross-encoder, and generates grounded, source-cited answers via **Groq Llama 3.3 70B**.

<br>

## ⚡ Highlights

| | |
|---|---|
| 📊 **23,000+ trials** ingested and chunked into **73,012** searchable passages |
| 🔍 **Hybrid retrieval** — FAISS dense search + BM25 sparse search fused with Reciprocal Rank Fusion |
| 🎯 **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` for precision reranking of top-k candidates |
| 🤖 **Llama 3.3 70B** on Groq for fast, grounded answer generation with source citations |
| 📈 **Evaluated** on 735 queries — Recall@5 = **0.73**, MRR = **0.50** |
| 🐳 **Dockerized** — one command to build & run |

<br>

## 🏗️ Architecture

```
                        ┌─────────────────────────────────┐
                        │         Streamlit Frontend       │
                        │    localhost:8501                 │
                        └──────────────┬──────────────────┘
                                       │  POST /ask
                                       ▼
                        ┌─────────────────────────────────┐
                        │         FastAPI Backend          │
                        │    localhost:8000                 │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │        RAG Service               │
                        │                                  │
                        │  ┌───────────┐  ┌────────────┐  │
                        │  │   FAISS   │  │   BM25     │  │
                        │  │  (Dense)  │  │  (Sparse)  │  │
                        │  └─────┬─────┘  └─────┬──────┘  │
                        │        │              │          │
                        │        ▼              ▼          │
                        │  ┌─────────────────────────┐    │
                        │  │  Reciprocal Rank Fusion  │    │
                        │  │       (k = 60)           │    │
                        │  └────────────┬────────────┘    │
                        │               ▼                  │
                        │  ┌─────────────────────────┐    │
                        │  │  Cross-Encoder Reranker  │    │
                        │  │  ms-marco-MiniLM-L-6-v2  │    │
                        │  └────────────┬────────────┘    │
                        │               ▼                  │
                        │  ┌─────────────────────────┐    │
                        │  │  Groq Llama 3.3 70B     │    │
                        │  │  (Answer Generation)     │    │
                        │  └─────────────────────────┘    │
                        └─────────────────────────────────┘
```

<br>

## 📊 Retrieval Evaluation

Evaluated on **735 auto-generated queries** derived from the trial corpus (conditions + interventions).

| Metric | Score | Description |
|:---|:---:|:---|
| **Recall@5** | **0.73** | 73% of relevant documents appear in the top 5 results |
| **MRR** | **0.50** | On average, the first relevant result is at rank ~2 |
| **Eval Queries** | **735** | Auto-generated from conditions & interventions |

> **Retrieval pipeline:** Query → `all-MiniLM-L6-v2` embedding → FAISS top-20 + BM25 top-20 → RRF fusion → Cross-encoder rerank → Top 5 to LLM

<br>

## 📂 Project Structure

```
clinicaltrial_rag/
├── frontend.py                    # Streamlit UI
├── Dockerfile                     # Container setup
├── requirements.txt               # Dependencies
│
├── src/
│   ├── api.py                     # FastAPI endpoints (GET /, POST /ask)
│   ├── rag_service.py             # Orchestrates retrieval → rerank → generate
│   ├── app.py                     # Application config
│   │
│   ├── ingestion/
│   │   ├── data_loader.py         # Load clinical_trials.csv
│   │   └── chunker.py             # Fixed-window chunking (50 words)
│   │
│   ├── retrieval/
│   │   ├── embedder.py            # all-MiniLM-L6-v2 sentence embeddings
│   │   ├── vector_store.py        # FAISS index wrapper
│   │   ├── bm25_store.py          # BM25 sparse retriever
│   │   ├── hybrid_retriever.py    # RRF fusion of FAISS + BM25
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   ├── index_builder.py       # Build & persist indexes
│   │   └── search.py              # Search utilities
│   │
│   ├── generation/
│   │   └── rag_chain.py           # Prompt template + Groq Llama 3.3 70B
│   │
│   └── evaluation/
│       └── retrieval_evaluator.py # Recall@5 & MRR evaluation
│
├── scripts/
│   └── generate_eval_set.py       # Generate evaluation queries from corpus
│
├── data/
│   ├── clinical_trials.csv        # Raw trial data (23,000+ records)
│   ├── evaluation_queries.csv     # 735 eval query-document pairs
│   └── indexes/                   # Persisted FAISS & BM25 indexes
│       └── clinical_trials/
│
└── notebooks/                     # Exploration & analysis notebooks
```

<br>

## 🚀 Quickstart

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com/)

### 1. Clone & install

```bash
git clone https://github.com/rushi-prog/clinical_rag.git
cd clinical_rag
```

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env
```

### 3. Build the index (first time only)

```bash
uv run python -m src.retrieval.index_builder
```

This will:
- Load `data/clinical_trials.csv` (23,000+ trials)
- Chunk into 50-word windows → **73,012 chunks**
- Embed with `all-MiniLM-L6-v2` (384-dim)
- Build & persist FAISS + BM25 indexes

### 4. Start the backend

```bash
uv run uvicorn src.api:app --reload
```

API available at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

### 5. Start the frontend

```bash
uv run streamlit run frontend.py
```

App available at `http://localhost:8501`

### 🐳 Docker

```bash
docker build -t clinical-trial-rag .
docker run -p 8000:8000 --env-file .env clinical-trial-rag
```

<br>

## 🔬 How It Works

### 1. Ingestion

Clinical trial records are loaded from CSV containing fields: **Study Title**, **Conditions**, **Interventions**, and **Brief Summary**. Each trial is concatenated into a structured document and chunked into 50-word fixed windows, producing **73,012 chunks**.

### 2. Indexing

Each chunk is embedded using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dimensional vectors) and stored in a **FAISS** flat index for dense retrieval. Simultaneously, a **BM25** sparse index is built over the raw text for lexical matching.

### 3. Hybrid Retrieval

At query time, both FAISS (semantic) and BM25 (lexical) return their top-k candidates. The results are combined using **Reciprocal Rank Fusion (RRF)** with `k=60`:

```
RRF_score(doc) = Σ 1 / (k + rank_i)
```

This ensures documents ranked highly by either method get promoted, while mitigating the weaknesses of each individual retriever.

### 4. Reranking

The fused candidate set is re-scored by a **cross-encoder** ([`ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)) which attends jointly to the query-document pair for more accurate relevance estimation. The top 5 documents after reranking are passed to the generator.

### 5. Generation

The top 5 reranked documents are injected into a structured prompt and sent to **Groq Llama 3.3 70B** (`temperature=0`) which produces a grounded answer with inline source citations (e.g., `[Source 1]`, `[Source 2]`).

<br>

## 📈 Evaluation

### Generating the eval set

```bash
uv run python scripts/generate_eval_set.py
```

Samples 500 trials and generates condition-based + intervention-based queries → **735 unique query-document pairs**.

### Running the evaluation

```bash
uv run python -m src.evaluation.retrieval_evaluator
```

### Results

```
Recall@5: 0.73
MRR:      0.50
```

| Metric | Value | Interpretation |
|:---|:---:|:---|
| **Recall@5** | 0.73 | The correct document is in the top 5 for 73% of queries |
| **MRR** | 0.50 | When found, the relevant doc averages rank ~2 |

<br>

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Frontend** | Streamlit | Interactive query interface |
| **Backend** | FastAPI | REST API (`GET /`, `POST /ask`) |
| **Dense Retrieval** | FAISS + all-MiniLM-L6-v2 | Semantic vector search |
| **Sparse Retrieval** | BM25 (rank-bm25) | Lexical keyword matching |
| **Fusion** | Reciprocal Rank Fusion | Combine dense + sparse rankings |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-attention reranking |
| **Generation** | Groq (Llama 3.3 70B) | Answer synthesis with citations |
| **Containerization** | Docker | Deployment packaging |

<br>

## 🔗 API Reference

### `GET /`

Health check.

```json
{ "message": "Clinical Trial RAG API" }
```

### `POST /ask`

Ask a clinical question.

**Request:**

```json
{
  "question": "Summarize clinical trial evidence related to insulin therapy."
}
```

**Response:**

```json
{
  "question": "Summarize clinical trial evidence related to insulin therapy.",
  "answer": "Clinical trial evidence related to insulin therapy includes comparisons of biphasic insulin aspart 30 and biphasic human insulin 30 [Source 2]...",
  "sources": [
    "[Source 1] glyburide vs insulin in the treatment of gestational diabetes mellitus...",
    "[Source 2] aim of this clinical trial is to investigate the blood sugar lowering effect..."
  ]
}
```

<br>

## 📜 License

This project is for educational and research purposes.

---

<p align="center">
  Built with ❤️ for clinical research
</p>
