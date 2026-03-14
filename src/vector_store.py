import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader
)

DATA_PATH = "knowledge"
VECTOR_PATH = "vectorstore"

def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_documents():

    documents = []

    for file in os.listdir(DATA_PATH):

        path = os.path.join(DATA_PATH,file)

        if file.endswith(".txt"):
            loader = TextLoader(path)

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)

        elif file.endswith(".csv"):
            loader = CSVLoader(path)

        else:
            continue

        documents.extend(loader.load())

    return documents


def load_or_create_vector_db():

    embeddings = load_embeddings()

    if os.path.exists(VECTOR_PATH):

        return FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_PATH)

    return db