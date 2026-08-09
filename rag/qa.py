import time
from langchain_core.outputs import Generation

from rag.config import get_llm, MODEL
from rag.cache import setup_cache

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


def answer_with_rag(vectorstore, query, use_cache=True):
    start = time.time()

    if use_cache:
        hit = semantic_cache.lookup(query, LLM_KEY)
        if hit:
            return hit[0].text, 0, time.time() - start, True

    retrieved = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
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
        semantic_cache.update(query, LLM_KEY, [Generation(text=result)])

    return result, tokens_used, elapsed, False