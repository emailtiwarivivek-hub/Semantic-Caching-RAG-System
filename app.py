import streamlit as st

from rag.cache import setup_cache
from rag.ingest import build_vectorstore
from rag.qa import answer_with_rag

setup_cache()

st.title("RAG with Semantic Caching")

if st.sidebar.button("Clear cache"):
    setup_cache().clear()
    st.success("Cache cleared")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("filename") != uploaded.name:
        with st.spinner("Indexing pdf..."):
            st.session_state.vectorstore = build_vectorstore(uploaded)
            st.session_state.filename = uploaded.name

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            result, tokens_used, elapsed, was_cached = answer_with_rag(
                st.session_state.vectorstore, query
            )
        st.write(result)
        st.write(f"tokens: {tokens_used}  time: {elapsed:.2f}s  cached: {was_cached}")