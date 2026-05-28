# Clinical Trial Literature Synthesizer — Claude Code Prompt

## Project overview

Build a production-grade RAG system that answers pharmacovigilance and clinical trial design questions by retrieving and synthesizing evidence from PubMed, ClinicalTrials.gov, and FDA FAERS. Every answer must be grounded with full citation chains (PMID / NCT IDs). The system prioritizes faithfulness over fluency — a hallucinated drug interaction is worse than "I don't know."

---

## Tech stack

- **Language**: Python 3.11+
- **Orchestration**: LangChain (with LCEL chains)
- **Embedding model**: `pritamdeka/S-PubMedBert-MS-MARCO` via `sentence-transformers`
- **Vector store**: Qdrant (local Docker mode for dev, Qdrant Cloud for prod)
- **Sparse retrieval**: `rank_bm25`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- **LLM**: Claude (`claude-sonnet-4-20250514`) via `anthropic` SDK
- **PDF parsing**: GROBID (Docker) for section-aware chunking
- **Evaluation**: RAGAS
- **Data ingestion**: `biopython` for PubMed API, `requests` for ClinicalTrials.gov API

---

## Project structure to scaffold

```
clinical-rag/
├── ingestion/
│   ├── pubmed_fetcher.py        # E-utilities API wrapper, bulk abstract fetch
│   ├── clinicaltrials_fetcher.py # ClinicalTrials.gov v2 API wrapper
│   ├── faers_fetcher.py         # FDA FAERS OpenFDA API wrapper
│   ├── grobid_parser.py         # Section-aware PDF → structured chunks
│   └── normalizer.py            # UMLS/RxNorm drug name normalization
├── indexing/
│   ├── chunker.py               # Section-aware + parent-child chunking
│   ├── embedder.py              # BioMedBERT embedding wrapper
│   ├── qdrant_store.py          # Qdrant collection setup + upsert
│   └── bm25_store.py            # BM25 index build + persist
├── retrieval/
│   ├── hybrid_retriever.py      # RRF fusion of BM25 + dense results
│   ├── reranker.py              # Cross-encoder reranker, top-k
│   └── query_rewriter.py        # HyDE + PICO query decomposition
├── generation/
│   ├── prompt_templates.py      # System + user prompt templates
│   ├── chain.py                 # LCEL chain: retrieve → rerank → generate
│   └── attribution.py           # Post-gen attribution scoring per claim
├── evaluation/
│   ├── ragas_eval.py            # RAGAS faithfulness + relevancy pipeline
│   ├── retrieval_eval.py        # Recall@k, MRR on BioASQ eval set
│   └── contradiction_detector.py # Flag conflicting claims in retrieved chunks
├── api/
│   └── server.py                # FastAPI server exposing /query endpoint
├── config.py                    # Pydantic settings (API keys, model names)
├── requirements.txt
└── README.md
```

---

## Step-by-step build order

### Step 1 — Environment and config

Set up `pyproject.toml` or `requirements.txt` with all dependencies. Create `config.py` using `pydantic-settings` to load:
- `ANTHROPIC_API_KEY`
- `QDRANT_URL`, `QDRANT_API_KEY`
- `NCBI_API_KEY` (free, increases PubMed rate limit from 3 → 10 req/s)
- `UMLS_API_KEY`
- Embedding model name, reranker model name, LLM model name
- Chunking parameters: `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`, `PARENT_CHUNK_SIZE=2048`

### Step 2 — Data ingestion layer

**`pubmed_fetcher.py`**
- Use Biopython `Entrez` to search PubMed by MeSH terms and free text
- Fetch abstracts + metadata: PMID, journal, year, MeSH terms, article type
- Support bulk fetch with pagination (10k records per query)
- Store raw records as JSON lines to disk before processing

**`clinicaltrials_fetcher.py`**
- Use ClinicalTrials.gov v2 REST API (`https://clinicaltrials.gov/api/v2/studies`)
- Fetch fields: NCT ID, phase, status, conditions, interventions, primary outcome, enrollment, sponsor
- Filter: only Interventional studies, phases II–IV

**`faers_fetcher.py`**
- Use OpenFDA API (`https://api.fda.gov/drug/event.json`)
- Fetch adverse event reports by drug name (use RxNorm normalized name)
- Aggregate by outcome (serious, hospitalization, death)

**`normalizer.py`**
- Use RxNorm REST API to normalize drug names to standard concept (RXCUI)
- Build a local synonym dict so "aspirin" / "ASA" / "acetylsalicylic acid" all map to the same entity
- Tag every chunk with normalized drug entities before indexing

### Step 3 — Chunking strategy

**`chunker.py`** — implement parent-child chunking:
1. Split document into large parent chunks (~2048 tokens) by section (Abstract, Methods, Results, Discussion, Conclusion)
2. Split each parent into small child chunks (~512 tokens with 64-token overlap)
3. Store child chunks in Qdrant for retrieval
4. Store parent chunks separately; each child chunk metadata includes `parent_id`
5. At retrieval time: fetch child chunks, then swap in their parent chunks for the LLM context window

For ClinicalTrials records (structured JSON, not free text):
- Flatten to a structured text template: "Trial [NCT ID] is a Phase [X] [status] study of [intervention] in [condition]. Primary outcome: [outcome]. Enrollment: [N]."
- Chunk these as single units (they're already short)

### Step 4 — Indexing

**`qdrant_store.py`**
- Create two collections:
  - `biomedical_chunks` — child chunks, BioMedBERT embeddings (768-dim), HNSW index
  - `biomedical_parents` — parent chunks, stored as payload only (no vector needed)
- Payload fields per chunk: `source` (pubmed/clinicaltrials/faers), `pmid`, `nct_id`, `section`, `year`, `phase`, `drug_entities[]`, `parent_id`
- Implement metadata filtering helpers: filter by source, phase, year range, drug entity

**`bm25_store.py`**
- Build a BM25 index over the same child chunks using `rank_bm25`
- Tokenize with a biomedical-aware tokenizer (remove stopwords, keep MeSH terms)
- Persist index to disk with `pickle` so it doesn't rebuild every run

### Step 5 — Retrieval pipeline

**`hybrid_retriever.py`**
- Take a query string + optional filters (phase, year, source)
- Run BM25 retrieval → top 100 results with scores
- Run dense retrieval via Qdrant → top 100 results with scores
- Fuse using Reciprocal Rank Fusion (RRF): `score = 1 / (k + rank)` where `k=60`
- Return top 20 fused results

**`query_rewriter.py`**
- Implement HyDE: send query to LLM asking it to "write a short passage from a clinical paper that would answer this question." Embed the hypothetical passage and use it as the dense query vector.
- Implement PICO decomposition: parse the query into Population, Intervention, Comparator, Outcome components. Run separate retrievals per component and merge.

**`reranker.py`**
- Take hybrid retriever output (top 20)
- Score each (query, chunk) pair with cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Return top 5 after reranking
- Swap child chunks for parent chunks before passing to generation

### Step 6 — Generation chain

**`prompt_templates.py`**
System prompt:
```
You are a clinical research assistant. Answer the question using ONLY the provided context. 
For every factual claim, cite the source using [PMID:XXXXXXXX] or [NCT:XXXXXXXXXX].
If the context does not contain enough information to answer, say "Insufficient evidence in retrieved sources."
Never speculate beyond what the sources state.
```

User prompt template:
```
Context:
{context}

Question: {question}

Instructions: Answer in structured format. For each claim, add inline citation. 
End with a "Sources" section listing all cited PMIDs and NCT IDs.
```

**`chain.py`** — LCEL chain:
```python
chain = (
    RunnableParallel(question=RunnablePassthrough(), context=retriever)
    | prompt
    | llm
    | attribution_scorer
    | StrOutputParser()
)
```

**`attribution.py`**
- After generation, extract all cited PMIDs/NCT IDs from the answer
- For each claim sentence, check whether the cited chunk actually supports it
- Return an attribution score per claim (0–1) and flag any unsupported claims
- If any claim scores < 0.5 faithfulness, prepend a warning to the answer

### Step 7 — Evaluation harness

**`retrieval_eval.py`**
- Download BioASQ training set (task B, factoid questions)
- For each question, run hybrid retriever and check if gold-standard PubMed articles appear in top-5
- Report Recall@1, Recall@5, MRR

**`ragas_eval.py`**
- Build a small eval set of 50 clinical questions with ground-truth answers (manually written or from BioASQ)
- Run full chain on each question
- Score with RAGAS: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Target thresholds: faithfulness > 0.85, context_precision > 0.70

**`contradiction_detector.py`**
- After retrieval, before generation: check if any two retrieved chunks contain contradictory claims about the same drug+outcome pair
- Use a lightweight NLI model (`cross-encoder/nli-deberta-v3-small`) to score contradiction
- If contradiction detected, include a note in the system prompt: "Note: Retrieved sources contain conflicting evidence on [topic]. Surface both views explicitly."

### Step 8 — FastAPI server

**`server.py`**
```python
POST /query
  body: { question: str, filters: { phase?: str, year_from?: int, source?: str[] } }
  response: { answer: str, sources: [{id, title, year, url}], attribution_scores: {}, warnings: [] }

GET /health
GET /stats  # index size, last updated timestamp
```

---

## Key implementation notes

1. **Never embed full documents** — always chunk first. A 30-page paper embedded as one vector loses all retrieval precision.

2. **Drug name normalization is non-negotiable** — without it, a query about "ibuprofen" will miss chunks that say "Advil" or "NSAIDs." Run the normalizer on both the indexed chunks AND the incoming query.

3. **Parent-child retrieval matters** — embed small chunks for precision, but pass large parent chunks to the LLM for context. This single change typically improves answer quality more than any prompt engineering.

4. **RRF over weighted sum** — Reciprocal Rank Fusion is more robust than trying to tune BM25/dense score weights. Use it unless you have a specific reason not to.

5. **GROBID for PDFs** — if you need full-text papers (not just abstracts), run GROBID as a local Docker service (`docker run -p 8070:8070 grobid/grobid:latest`) and parse PDFs into structured XML before chunking. Do not use naive PyPDF2 — it destroys document structure.

6. **Rate limits** — PubMed allows 10 req/s with an API key, 3 req/s without. ClinicalTrials.gov v2 allows 5 req/s. Build in `asyncio` rate limiting from the start.

7. **Faithfulness gate** — before returning any answer, check attribution score. If faithfulness < 0.7, return the answer with a visible warning rather than silently serving a potentially hallucinated response.

---

## Milestones to hit in order

- [ ] Ingest 10,000 PubMed abstracts for a single drug (e.g. metformin) and index them
- [ ] Hybrid retrieval working, returning top-5 relevant chunks for a test query
- [ ] Reranker integrated, measurably improving Recall@5 vs no reranker
- [ ] End-to-end chain generating a cited answer for one test question
- [ ] Attribution scorer flagging at least one unsupported claim in eval set
- [ ] RAGAS faithfulness > 0.80 on 20-question eval set
- [ ] FastAPI server running locally, answering queries via HTTP
- [ ] Full eval run on BioASQ subset with Recall@5 and MRR reported

---

## Good first query to test the system end-to-end

> "What are the known hepatotoxicity risks of metformin in elderly patients, and which clinical trials have studied this population?"

This requires multi-source retrieval (PubMed abstracts + trial records), drug entity matching, and multi-hop reasoning — a good stress test of the full pipeline.