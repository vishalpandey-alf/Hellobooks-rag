import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_PATH = "vectorstore/db_faiss"


def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )


def build_vector_db(documents, embeddings):

    db = FAISS.from_documents(documents, embeddings)

    os.makedirs(VECTOR_PATH, exist_ok=True)

    db.save_local(VECTOR_PATH)

    return db


def load_or_create_vector_db(documents):

    embeddings = load_embeddings()

    index_file = os.path.join(VECTOR_PATH, "index.faiss")

    # ---------------------------------
    # SAFE LOAD
    # ---------------------------------

    if os.path.exists(index_file):

        try:

            db = FAISS.load_local(
                VECTOR_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

        except Exception:
            # rebuild if index corrupted
            db = build_vector_db(documents, embeddings)

    else:

        db = build_vector_db(documents, embeddings)

    return db