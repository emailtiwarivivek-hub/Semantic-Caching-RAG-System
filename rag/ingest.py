import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_embeddings


def build_vectorstore(uploaded_file):
    """
    Turn an uploaded pdf into a FAISS vector store.

    Args:
    uploaded_file: file object from st.file_uploader

    Returns:
    FAISS vector store
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    pages = PyPDFLoader(path).load()
    os.unlink(path)

    splitter = splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    return FAISS.from_documents(chunks, get_embeddings())
