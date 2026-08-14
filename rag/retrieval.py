
import re

from rank_bm25 import BM25Okapi

from rag.ingest import iter_documents, make_chunk_id

# How many candidates each retriever contributes before fusion.
# You fuse a deep pool then keep the top k, because a chunk sitting at rank 12
# in BM25 and rank 9 in FAISS should win — but only if both lists go deep
# enough to contain it. Fusing two top-5 lists throws that signal away.
CANDIDATE_DEPTH = 20
RRF_K = 60


def key_of(doc):
    """
    Stable identifier for a chunk.

    Uses the chunk_id stamped at ingest time. If you haven't updated ingest.py
    yet, it computes the same hash on the fly, so this module works either way.
    """
    return doc.metadata.get("chunk_id") or make_chunk_id(doc)


def tokenize(text):
    """
    Split text into lowercase word tokens.

    BM25 is pure keyword matching — it has no idea "car" and "automobile" are
    related. It only counts word overlap. So tokenization is the whole game:
    "GST" and "gst" must become the same token or they'll never match.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_n=None):
    """
    Merge several ranked lists into one.

    The problem: FAISS returns cosine similarities (roughly 0-1), BM25 returns
    unbounded scores that depend on corpus statistics. You cannot add them —
    14.2 and 0.83 are not on the same scale, and normalizing is fragile
    because the ranges shift from query to query.

    RRF's answer: throw the scores away, use only the *positions*.
    Each list gives a chunk 1 / (k + rank) points, and the points are summed.

        rank 1 -> 1/61 = 0.0164
        rank 2 -> 1/62 = 0.0161
        rank 3 -> 1/63 = 0.0159

    The gaps between adjacent ranks are tiny, so a chunk that both retrievers
    rank *decently* (3rd and 4th) beats one that a single retriever ranks 1st
    and the other misses entirely. Agreement beats confidence.

    k is a damping constant. Small k makes rank 1 far more valuable than rank
    2, letting one confident retriever dominate. Large k flattens the curve so
    agreement matters more. 60 is the original paper's default.

    Args:
        ranked_lists: lists of chunk_ids, each already sorted best-first
        k: damping constant
        top_n: how many to return (all if None)
    """
    scores = {}
    first_seen = {}  # deterministic tie-breaking

    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = len(first_seen)

    fused = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    return fused[:top_n] if top_n else fused


class HybridRetriever:
    """
    FAISS + BM25, merged with Reciprocal Rank Fusion.

    Embeddings capture meaning but blur exact tokens; BM25 nails exact tokens
    but understands no meaning. Names, product codes, and numbers are where
    dense search quietly fails, and that's the gap BM25 fills.

    Build this ONCE per document (it tokenizes the whole corpus), then reuse
    it for every query.
    """

    def __init__(self, vectorstore, depth=CANDIDATE_DEPTH, rrf_k=RRF_K):
        self.vs = vectorstore
        self.depth = depth
        self.rrf_k = rrf_k

        docs = list(iter_documents(vectorstore))
        self.by_id = {key_of(d): d for d in docs}

        # BM25Okapi wants a list of token lists, one per document. We keep
        # ids in the same order so a score position maps back to a chunk.
        self.ids = [key_of(d) for d in docs]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def _dense_ids(self, query, k):
        return [key_of(d) for d in self.vs.similarity_search(query, k=k)]

    def _bm25_ids(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        # Sort positions by score descending; the second key keeps ties in
        # stable corpus order so repeated runs give identical results.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.ids[i] for i in order[:k]]

    def search_ids(self, query, k=5):
        fused = reciprocal_rank_fusion(
            [self._dense_ids(query, self.depth),
             self._bm25_ids(query, self.depth)],
            k=self.rrf_k,
            top_n=k,
        )
        return fused

    def search(self, query, k=5):
        return [self.by_id[cid] for cid in self.search_ids(query, k)]


class DenseRetriever:
    """FAISS only — your original behaviour, kept for side-by-side comparison."""

    def __init__(self, vectorstore):
        self.vs = vectorstore

    def search(self, query, k=5):
        return self.vs.similarity_search(query, k=k)

    def search_ids(self, query, k=5):
        return [key_of(d) for d in self.search(query, k)]


class BM25Retriever:
    """BM25 only — keyword search, no embeddings involved."""

    def __init__(self, vectorstore):
        docs = list(iter_documents(vectorstore))
        self.by_id = {key_of(d): d for d in docs}
        self.ids = [key_of(d) for d in docs]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    def search_ids(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.ids[i] for i in order[:k]]

    def search(self, query, k=5):
        return [self.by_id[cid] for cid in self.search_ids(query, k)]


def get_retriever(vectorstore, mode="hybrid"):
    """Factory so the UI can switch strategies by name."""
    return {
        "hybrid": HybridRetriever,
        "dense": DenseRetriever,
        "bm25": BM25Retriever,
    }[mode](vectorstore)
