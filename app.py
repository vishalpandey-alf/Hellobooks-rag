import streamlit as st
import time
import json
from datetime import datetime

from src.rag_pipeline import load_rag_pipeline

st.set_page_config(
    page_title="Hellobooks AI",
    page_icon="💼",
    layout="wide"
)

# ---------- Dark SaaS Theme ----------
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0E1117;
    color: white;
}

.stChatMessage {
    border-radius: 10px;
    padding: 12px;
}

.block-container {
    padding-top: 2rem;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

h1 {
    font-weight: 700;
}

.footer {
    text-align:center;
    font-size:12px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)


# ---------- Header ----------
st.title("💼 Hellobooks AI Assistant")
st.caption("AI powered accounting support using RAG + Local LLM")


# ---------- Sidebar ----------
with st.sidebar:

    st.markdown("## ⚙️ Control Panel")

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []

    st.markdown("---")

    st.markdown("## 📤 Export Chat")

    if "messages" in st.session_state and len(st.session_state.messages) > 0:

        chat_json = json.dumps(st.session_state.messages, indent=2)

        st.download_button(
            label="Download Chat History",
            data=chat_json,
            file_name=f"hellobooks_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    st.markdown("---")

    st.markdown("## 🧠 System")

    st.markdown("""
Vector DB: FAISS  
LLM Runtime: Ollama  
Model: Mistral  
Embedding: MiniLM
""")

    st.markdown("---")

    st.markdown("## 💡 Example Questions")

    st.markdown("""
• What is bookkeeping?  
• Explain accrual accounting  
• What is a balance sheet?  
• How are invoices recorded?
""")


# ---------- Load RAG Backend ----------
with st.spinner("Initializing AI engine..."):
    qa_chain = load_rag_pipeline()


# ---------- Conversation Memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Display Chat ----------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)


# ---------- Chat Input ----------
prompt = st.chat_input("Ask an accounting question...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Assistant response
    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner("Searching knowledge base..."):

            result = qa_chain.invoke({"query": prompt})

            answer = result["result"]
            sources = []

            for doc in result["source_documents"]:
                sources.append(doc.metadata.get("source", "document"))

        # Streaming typing effect
        full_text = ""
        for word in answer.split():

            full_text += word + " "
            placeholder.markdown(full_text)

            time.sleep(0.02)

        # Show sources
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


# ---------- Footer ----------
st.markdown("""
<div class="footer">
Hellobooks AI • Accounting Assistant Demo
</div>
""", unsafe_allow_html=True)