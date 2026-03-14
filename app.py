import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from src.vector_store import load_or_create_vector_db
from src.hybrid_retriever import HybridRetriever

load_dotenv()


def load_rag_pipeline():

    vector_db = load_or_create_vector_db()

    retriever = HybridRetriever(vector_db)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        streaming=True,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return llm,retriever


def generate_answer(llm,retriever,query):

    docs = retriever.search(query,k=4)

    context = "\n\n".join([d.page_content for d in docs[:2]])

    prompt = f"""

You are an expert accounting AI assistant.

Use the provided context to answer the question.

Context:
{context}

Question:
{query}

Provide a clear and detailed answer.
"""

    response = llm.invoke(prompt)

    return response.content,docs