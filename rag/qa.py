import time

from langchain_core.outputs import Generation

from rag.cache import setup_cache
from rag.config import MODEL, get_llm

semantic_cache = setup_cache()
LLM_KEY = MODEL


def extract_text(content):
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content


def answer_with_rag(vectorstore, query, use_cache=True, retriever=None, k=5):
    """
    Answer a question using retrieved context.

    Args:
        vectorstore: the FAISS store (used only if no retriever is given)
        retriever:   any object with .search(query, k) -> [Document].
                     Pass a HybridRetriever to use BM25 + FAISS + RRF.
                     None keeps the original dense-only behaviour.
        use_cache:   set False to force a real retrieval + LLM call.

    Returns:
        (answer, tokens_used, elapsed_seconds, was_cached, retrieved_docs)
    """
    start = time.time()

    # The cache key must include the retrieval mode. Without it, an answer
    # cached in dense mode comes straight back when you switch to hybrid —
    # retrieval never runs, and hybrid looks like it changed nothing.
    mode_name = type(retriever).__name__ if retriever is not None else "dense"
    cache_key = f"{LLM_KEY}|{mode_name}"

    if use_cache:
        hit = semantic_cache.lookup(query, cache_key)
        if hit:
            # A cache hit skips retrieval entirely — that's the point of the
            # cache, and also why retrieval evals must pass use_cache=False.
            return hit[0].text, 0, time.time() - start, True, []

    if retriever is not None:
        retrieved = retriever.search(query, k)
    else:
        retrieved = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)

    context = "\n".join(doc.page_content for doc in retrieved)

    response = get_llm().invoke(
        f"Answer the question using only the context below. "
        f"If the answer is not in the context, say you don't know.\n\n"
        f"Context: {context}\nQuestion: {query}\n\nAnswer:"
    )
    elapsed = time.time() - start

    result = extract_text(response.content)
    tokens_used = (response.usage_metadata or {}).get("total_tokens", 0)

    if use_cache:
        semantic_cache.update(query, cache_key, [Generation(text=result)])

    return result, tokens_used, elapsed, False, retrieved