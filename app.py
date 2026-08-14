import logging

import streamlit as st
import transformers

from rag.cache import setup_cache
from rag.ingest import CHUNK_SIZE, build_vectorstore
from rag.qa import answer_with_rag
from rag.retrieval import get_retriever

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.CRITICAL)

setup_cache()

st.title("RAG with Semantic Caching")

# Switching modes live is the cheapest way to show hybrid search working:
# ask a question containing a name, code, or number and compare the sources.
mode = st.sidebar.radio("Retrieval", ["hybrid", "dense", "bm25"])

if st.sidebar.button("Clear cache"):
    setup_cache().clear()
    st.success("Cache cleared")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # The key includes CHUNK_SIZE, not just the filename. Keying on the
    # filename alone means editing CHUNK_SIZE silently reuses the old chunks —
    # session_state still holds the index built with the previous setting, so
    # the change appears to do nothing.
    doc_key = f"{uploaded.name}:{CHUNK_SIZE}"

    if st.session_state.get("doc_key") != doc_key:
        with st.spinner("Indexing pdf..."):
            st.session_state.vectorstore = build_vectorstore(uploaded)
            st.session_state.doc_key = doc_key
            st.session_state.retrievers = {}  # drop retrievers for the old index

    # setdefault returns the existing dict, or creates it if missing.
    # session_state survives code edits, so a PDF indexed before this file
    # changed could leave doc_key set but retrievers absent.
    retrievers = st.session_state.setdefault("retrievers", {})

    # Build each retriever once per document and keep it. BM25 tokenizes the
    # whole corpus on construction, so rebuilding it per query would be slow.
    if mode not in retrievers:
        retrievers[mode] = get_retriever(st.session_state.vectorstore, mode)
    retriever = retrievers[mode]

    st.sidebar.caption(f"chunk size: {CHUNK_SIZE}")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            result, tokens_used, elapsed, was_cached, docs = answer_with_rag(
                st.session_state.vectorstore, query, retriever=retriever
            )
        st.write(result)
        st.caption(
            f"tokens: {tokens_used} | time: {elapsed:.2f}s | "
            f"cached: {was_cached} | retrieval: {mode}"
        )

        if docs:
            with st.expander(f"Sources ({len(docs)} chunks)"):
                for i, doc in enumerate(docs, start=1):
                    page = doc.metadata.get("page", "?")
                    chars = len(doc.page_content)
                    st.markdown(f"**{i}. page {page}**  ·  {chars} chars")
                    st.text(doc.page_content[:400])