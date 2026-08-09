import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

REDIS_URL = os.getenv("REDIS_HOST_KEY")
MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)