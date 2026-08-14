import streamlit as st
from langchain_redis import RedisSemanticCache

from rag.config import REDIS_URL, get_embeddings


@st.cache_resource
def setup_cache(distance_threshold=0.1):
    """
    Create the semantic cache.

    Deliberately NOT registered via set_llm_cache(): qa.py does its own
    lookup/update keyed on the user query alone. Registering globally would
    add a second cache underneath, keyed on the full prompt — which is mostly
    retrieved context — so different questions about the same topic collide.

    Lower distance_threshold = stricter match.
    """
    return RedisSemanticCache(
        embeddings=get_embeddings(),
        redis_url=REDIS_URL,
        distance_threshold=distance_threshold,
    )