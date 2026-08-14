import hashlib
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_embeddings

INDEX_PATH = "faiss_index"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def make_chunk_id(chunk) -> str:
    """A stable fingerprint for one chunk."""
    page = chunk.metadata.get("page", "?")
    payload = f"{page}||{chunk.page_content}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def split_documents(pages):
    """Split pages into chunks and stamp each one with a stable chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = make_chunk_id(chunk)

    return chunks


def build_vectorstore(uploaded_file):
    """Turn an uploaded PDF into a FAISS vector store (Streamlit safe)."""
    # delete=False required for Windows compatibility before reading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        return build_vectorstore_from_path(path)
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError:
                pass  # Avoid crashing Streamlit if Windows holds a temporary file lock


def build_vectorstore_from_path(pdf_path):
    """Build FAISS vector store from disk path using stable chunk IDs."""
    pages = PyPDFLoader(pdf_path).load()
    chunks = split_documents(pages)
    
    # Pass stable chunk_ids directly as FAISS primary keys
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    
    return FAISS.from_documents(
        documents=chunks, 
        embedding=get_embeddings(), 
        ids=ids
    )


def save_vectorstore(vectorstore, path=INDEX_PATH):
    vectorstore.save_local(path)


def load_vectorstore(path=INDEX_PATH):
    """Load a previously saved index."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"No index at '{path}'. Build one first:\n"
            f"    python scripts\\build_index.py path\\to\\your.pdf"
        )
    return FAISS.load_local(
        path,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def iter_documents(vectorstore):
    """Yield every Document in the index."""
    for _, doc_id in vectorstore.index_to_docstore_id.items():
        yield vectorstore.docstore.search(doc_id)