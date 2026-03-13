from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

import os


VECTOR_PATH = "vectorstore"


def load_rag_pipeline():

    # -------------------------
    # EMBEDDINGS (load once)
    # -------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # LOAD EXISTING VECTOR DB
    # -------------------------

    if os.path.exists(VECTOR_PATH):

        vector_db = FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        # -------------------------
        # LOAD DOCUMENTS
        # -------------------------

        loader = DirectoryLoader(
            "knowledge",
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()

        # -------------------------
        # SPLIT DOCUMENTS
        # -------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        # -------------------------
        # CREATE VECTOR DATABASE
        # -------------------------

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        # save for future runs
        vector_db.save_local(VECTOR_PATH)

    # -------------------------
    # RETRIEVER
    # -------------------------

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    # -------------------------
    # LLM (FAST CONFIG)
    # -------------------------

    llm = Ollama(
        model="gemma:2b",
        temperature=0.2
    )

    # -------------------------
    # PROMPT
    # -------------------------

    template = """
You are a professional accounting assistant.

Use the context to answer the question clearly.

Rules:
- Provide 5–7 sentences
- Explain in simple language
- Give examples if useful

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # -------------------------
    # QA CHAIN
    # -------------------------

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        chain_type="stuff",

        retriever=retriever,

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa_chain, vector_db