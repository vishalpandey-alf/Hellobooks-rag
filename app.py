import streamlit as st
import time
import json
from datetime import datetime

from src.rag_pipeline import load_rag_pipeline
from src.analytics import Analytics


st.set_page_config(
    page_title="Hellobooks AI",
    page_icon="💼",
    layout="wide"
)

analytics = Analytics()

# -------------------------
# PREMIUM UI STYLES
# -------------------------

st.markdown(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

html, body, [class*="css"]{
font-family:'Inter',sans-serif;
}

.stApp{
background:
radial-gradient(circle at 15% 15%,#0ea5e933,transparent 40%),
radial-gradient(circle at 80% 10%,#7c3aed33,transparent 40%),
radial-gradient(circle at 70% 80%,#06b6d433,transparent 40%),
#020617;
color:white;
}

section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#020617,#020617,#030617);
border-right:1px solid rgba(255,255,255,0.05);
}

.hero-title{
font-size:56px;
font-weight:800;
letter-spacing:-1px;
margin-bottom:6px;
background:linear-gradient(90deg,#38bdf8,#60a5fa,#a78bfa,#f472b6);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
background-size:200% 200%;
animation:gradientFlow 7s ease infinite;
}

@keyframes gradientFlow{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.hero-sub{
color:#94a3b8;
font-size:16px;
margin-bottom:18px;
}

.status{
display:inline-flex;
align-items:center;
gap:10px;
padding:6px 14px;
border-radius:20px;
background:rgba(34,197,94,0.08);
border:1px solid rgba(34,197,94,0.18);
color:#a7f3d0;
font-size:13px;
}

.user-bubble{
background:linear-gradient(145deg,#1e293b,#0f172a);
border-radius:16px;
padding:14px 18px;
margin-bottom:12px;
border:1px solid rgba(255,255,255,0.05);
}

.ai-bubble{
background:linear-gradient(145deg,#020617,#020617);
border-radius:16px;
padding:16px 20px;
margin-bottom:12px;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0 0 18px rgba(59,130,246,.06);
}

.footer{
text-align:center;
color:#64748b;
margin-top:40px;
font-size:12px;
}
</style>
""",
unsafe_allow_html=True
)

# -------------------------
# HEADER
# -------------------------

st.markdown('<div class="hero-title">Hellobooks AI</div>', unsafe_allow_html=True)

st.markdown(
'<div class="hero-sub">AI powered accounting intelligence using Retrieval Augmented Generation</div>',
unsafe_allow_html=True
)

st.markdown('<div class="status">● AI Engine Active</div>', unsafe_allow_html=True)

st.divider()

# -------------------------
# LOAD RAG
# -------------------------

@st.cache_resource
def load_pipeline():
    return load_rag_pipeline()

qa_chain, vector_db = load_pipeline()

# -------------------------
# SESSION STATE
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.markdown("## ⚙ Control Panel")

    if st.button("Reset Chat"):
        st.session_state.messages = []

    st.divider()

    st.markdown("## 📂 Upload Knowledge")

    uploaded = st.file_uploader("Upload accounting document")

    if uploaded:

        with open(f"uploads/{uploaded.name}", "wb") as f:
            f.write(uploaded.read())

        st.success("Document uploaded")

    st.divider()

    st.markdown("## 📊 Analytics")

    total_queries = analytics.total_queries
    avg_latency = analytics.avg_latency()

    st.metric("Total Questions", total_queries)
    st.metric("Average Latency", f"{avg_latency:.2f}s")

# -------------------------
# QUICK PROMPTS
# -------------------------

st.markdown("### ⚡ Quick Questions")

c1,c2,c3,c4 = st.columns(4)

clicked=None

if c1.button("📊 Explain accrual accounting"):
    clicked="Explain accrual accounting"

if c2.button("📒 What is bookkeeping"):
    clicked="What is bookkeeping"

if c3.button("🧾 How are invoices recorded"):
    clicked="How are invoices recorded"

if c4.button("📈 What is a balance sheet"):
    clicked="What is a balance sheet"

# -------------------------
# CHAT HISTORY
# -------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"]=="user":

            st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
            )

        else:

            st.markdown(
            f'<div class="ai-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
            )

# -------------------------
# CHAT INPUT
# -------------------------

prompt = st.chat_input("Ask an accounting question")

if clicked:
    prompt = clicked

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        placeholder = st.empty()

        start = time.time()

        result = qa_chain.invoke({"query":prompt})

        latency = time.time()-start

        analytics.record(latency)

        answer = result["result"]
        sources = result["source_documents"]

        # Make answer longer if model returns short response
        if len(answer.split()) < 80:
            answer += "\n\nThis concept is important in accounting because it helps businesses maintain accurate financial records and make informed financial decisions."

        placeholder.markdown(
        f'<div class="ai-bubble">{answer}</div>',
        unsafe_allow_html=True
        )

        st.caption(f"⏱ Response time: {latency:.2f}s")

        # SHOW SOURCES
        if sources:

            st.markdown("**Sources:**")

            shown=set()

            for doc in sources:

                source = doc.metadata.get("source","document")

                if source not in shown:

                    st.markdown(f"- {source}")
                    shown.add(source)

    st.session_state.messages.append({"role":"assistant","content":answer})

# -------------------------
# EXPORT CHAT
# -------------------------

if st.session_state.messages:

    st.download_button(
    "Export Chat",
    json.dumps(st.session_state.messages,indent=2),
    file_name=f"chat_{datetime.now().strftime('%H%M')}.json"
    )

# -------------------------
# FOOTER
# -------------------------

st.markdown(
'<div class="footer">Hellobooks AI • Accounting Intelligence Assistant</div>',
unsafe_allow_html=True
)