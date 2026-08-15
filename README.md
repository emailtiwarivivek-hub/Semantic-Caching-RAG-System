# Semantic Caching RAG System

A document Q&A system that adds two things most RAG implementations skip: **hybrid retrieval**
(keyword search alongside vector search, merged with Reciprocal Rank Fusion) and a **semantic
cache** that returns answers for questions it has already seen — even when they're worded
differently.

Upload a PDF, ask questions about it, and see exactly which chunks the answer came from.

---

## Why these two additions

**Vector search alone misses exact tokens.** Embeddings capture meaning but compress it.
"Section 234F" and "Section 234G" look nearly identical to an embedding model — both are
legal section references. Product codes, proper nouns, dates, and figures get blurred in
the same way. BM25 keyword search has the opposite profile: it nails exact tokens and
understands no meaning at all. Running both covers each one's blind spot.

**Exact-match caching barely fires.** "What's the refund policy?" and "How do refunds work?"
are different strings, so a normal cache treats them as unrelated and you pay for the LLM
call twice. A semantic cache embeds the question and looks for a previous question within a
distance threshold, so paraphrases hit.

---

## Architecture

```
                    ┌─ Redis semantic cache ── hit ──→ answer (0 tokens)
question ───────────┤
                    └─ miss
                        │
                        ├──→ FAISS  (dense, by meaning)  → 20 candidates
                        ├──→ BM25   (sparse, by keyword) → 20 candidates
                        │
                        └──→ Reciprocal Rank Fusion → top 5 chunks
                                    │
                                    └──→ Gemini → answer → cached
```

| Component | Choice | Reason |
|---|---|---|
| Vector store | FAISS (in-memory) | One PDF per session; a library, no server to run |
| Keyword search | BM25 (`rank-bm25`) | Recovers exact tokens embeddings blur |
| Fusion | Reciprocal Rank Fusion, hand-written | See below — implemented directly rather than via `EnsembleRetriever` |
| Embeddings | `all-MiniLM-L6-v2`, local CPU | Retrieval and cache lookups cost nothing and run offline |
| Cache | Redis Cloud vector search | Similarity lookup happens in Redis; survives restarts |
| LLM | Gemini 3.1 Flash-Lite | Only the final answer hits an API |

---

## How Reciprocal Rank Fusion works

FAISS returns cosine similarities (roughly 0–1). BM25 returns unbounded scores that depend on
corpus statistics and shift from query to query. You cannot add them — 14.2 and 0.83 are not
on the same scale, and normalizing is fragile because the ranges move.

RRF discards the scores entirely and uses only **positions**. Each list awards a chunk
`1 / (k + rank)` points, summed across lists:

```
rank 1 → 1/61 = 0.0164
rank 2 → 1/62 = 0.0161
rank 3 → 1/63 = 0.0159
```

The gaps between adjacent ranks are tiny, but appearing in a *second* list adds a whole
~0.016 — roughly 60× more valuable. **Agreement beats confidence.**

Worked example, chunk `C` being the correct answer:

```
FAISS:  A, C, D, E, F
BM25:   B, C, G, H, A

fused:  C  0.03226   ← 2nd in both lists
        A  0.03178   ← 1st in one, 5th in the other
        B  0.01639
        D  0.01587
```

`C` wins without topping either list. `A` was FAISS's top pick, but BM25 ranked it 5th — so it
loses to the chunk both methods independently liked. Dense-only search would have returned `A`
first.

Each retriever contributes 20 candidates rather than 5, because the interesting case is a chunk
ranked 9th by one and 12th by the other. Fusing two top-5 lists would discard exactly the
signal RRF exists to find.

`k = 60` is a damping constant from the original paper. Smaller values make rank 1 far more
valuable, letting one confident retriever dominate; larger values flatten the curve so
agreement matters more.

---

## Semantic caching, and two bugs worth documenting

**The measured win.** A repeat question, worded differently:

| | Tokens | Latency |
|---|---|---|
| First ask (`"what is Cutaneous T-cell lymphoma"`) | 505 | 7.72s |
| Paraphrase (`"tell me about Cutaneous T-cell lymphoma"`) | 0 | 0.59s |

**Bug 1 — keying on the full prompt.** The cache key originally included the whole prompt,
which is ~95% retrieved context and only a few words of actual question. Two different
questions about the same topic produced nearly identical prompts and collided. Fixed by keying
on the user query alone.

**Bug 2 — a second cache layer underneath.** `langchain_core.globals.set_llm_cache()` registers
a cache that intercepts *every* LLM call transparently. With that in place there were two
caches: the explicit one keyed on the user query, and a hidden one keyed on the full prompt —
reintroducing bug 1 below the layer that had fixed it.

The symptom was subtle: the UI reported `cached: False`, yet the response returned in 1.2s
instead of 7.7s with a token count *identical* to a previous answer. A genuine call would have
produced different numbers. Fixed by removing the global registration and letting the QA layer
own caching.

**Threshold is a real trade-off.** `distance_threshold` controls how similar a question must be
to hit. Too loose and you get false hits — a cached answer returned for a question it doesn't
actually address, delivered confidently and instantly. Too tight and the cache never fires.
Currently `0.1`.

---

## Setup

```bash
git clone https://github.com/emailtiwarivivek-hub/Semantic-Caching-RAG-System
cd Semantic-Caching-RAG-System

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
REDIS_HOST_KEY=redis://default:password@host:port
```

Run:

```bash
streamlit run app.py
```

First launch downloads the MiniLM embedding model (~90MB).

---

## Using it

Upload a PDF and ask a question. The sidebar switches between **hybrid**, **dense**, and
**bm25** so you can compare retrieval strategies on the same question — try one containing an
exact term (a name, section number, or figure) and compare the Sources panel between modes.

Every answer shows tokens used, latency, whether it was a cache hit, and which retrieval mode
produced it. The Sources expander lists the five chunks that were actually passed to the model.

---

## Project structure

```
rag/
  config.py      LLM + embedding factories, env loading
  ingest.py      PDF loading, chunking, stable chunk IDs, FAISS index
  retrieval.py   BM25, dense, and hybrid retrievers + RRF
  cache.py       Redis semantic cache
  qa.py          cache lookup → retrieve → prompt → answer
app.py           Streamlit UI
```

The LLM sits behind a factory function in `config.py`, so swapping model or provider means
editing one file without touching retrieval or caching.

Chunks carry a `chunk_id` derived from a hash of their own text, so IDs survive an index
rebuild. Random UUIDs would change on every rebuild, which makes offline evaluation impossible.

---

## Configuration

| Setting | Location | Current | Notes |
|---|---|---|---|
| `CHUNK_SIZE` | `rag/ingest.py` | 900 | Smaller chunks split answers across retrieval boundaries |
| `CHUNK_OVERLAP` | `rag/ingest.py` | 150 | Keeps context across chunk edges |
| `CANDIDATE_DEPTH` | `rag/retrieval.py` | 20 | Candidates per retriever before fusion |
| `RRF_K` | `rag/retrieval.py` | 60 | Damping constant |
| `distance_threshold` | `rag/cache.py` | 0.1 | Lower = stricter cache match |

---

## Roadmap

- **Evaluation harness** — a test set of ~100 questions with known correct chunks (generated by
  asking the LLM to write a question for each chunk), measuring `recall@5` across dense, BM25,
  and hybrid retrieval. Retrieval evals must run with the cache disabled, since a cache hit
  skips retrieval and would inflate the numbers.
- **Cross-encoder reranking** — retrieve 20 with hybrid, rerank with `BAAI/bge-reranker-base`
  on CPU, pass the top 3–5 to the LLM.
- **Cache threshold sweep** — hit rate, median latency, and false-hit rate measured across
  0.20 / 0.15 / 0.10 / 0.05, with the threshold chosen from the data rather than by feel.
