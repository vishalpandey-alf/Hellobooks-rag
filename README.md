# HelloBooks AI

HelloBooks AI is a lightweight **AI-powered accounting assistant** built using **Retrieval-Augmented Generation (RAG)**. The application allows users to ask accounting-related questions and receive intelligent answers generated from a curated knowledge base.

The system combines **semantic search with a local language model** to retrieve relevant accounting information and generate contextual responses.

---

## Features

* AI-powered accounting question answering
* Retrieval-Augmented Generation (RAG) architecture
* Local LLM inference using Ollama
* FAISS vector database for semantic search
* Knowledge base ingestion from text documents
* Interactive chat interface built with Streamlit
* Real-time response analytics
* Chat export functionality

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM Runtime:** Ollama
* **AI Framework:** LangChain
* **Embeddings:** HuggingFace Sentence Transformers
* **Vector Database:** FAISS
* **Architecture:** Retrieval-Augmented Generation (RAG)

---

## Project Structure

```
Hellobooks-rag
│
├── app.py
├── knowledge/
│   └── accounting documents
│
├── src/
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   └── analytics.py
│
├── vectorstore/
│   └── FAISS index files
│
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/hellobooks-ai.git
cd hellobooks-ai
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit server:

```
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## How It Works

1. Accounting documents are stored in the **knowledge** directory.
2. Documents are split into smaller chunks.
3. Embeddings are generated using **Sentence Transformers**.
4. The embeddings are stored in a **FAISS vector database**.
5. When a user asks a question:

   * Relevant document chunks are retrieved
   * The LLM generates a contextual answer using the retrieved information.

---

## Future Improvements

* Streaming responses
* Multi-document knowledge ingestion
* Advanced analytics dashboard
* Cloud deployment
* API support for external integrations

---

## License

This project is for educational and demonstration purposes.
