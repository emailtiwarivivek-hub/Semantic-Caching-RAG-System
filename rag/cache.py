import streamlit as st
from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from rag.config import REDIS_URL, get_embeddings


@st.cache_resource
def setup_cache(distance_threshold=0.2):
    """
    Create the semantic cache and register it globally.
    Lower distance_threshold = stricter match.
    """
    cache = RedisSemanticCache(
        embeddings=get_embeddings(),
        redis_url=REDIS_URL,
        distance_threshold=distance_threshold,
    )
    set_llm_cache(cache)
    return cache
