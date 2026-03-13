import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_PATH = "vector_db"

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

def load_or_create_vector_db(documents):

    embeddings = load_embeddings()

    if os.path.exists(VECTOR_PATH):
        db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(VECTOR_PATH)

    return db