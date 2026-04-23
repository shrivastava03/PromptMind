import os
import fitz
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.stApp { background: #0f0f0f; color: #e8e4dc; }

.pm-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.pm-subtitle {
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Setup card */
.setup-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.setup-card .card-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

/* Index status */
.index-card {
    background: #1a1a0f;
    border: 1px solid #c8a96e33;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    display: flex;
    gap: 24px;
}
.index-stat .label { font-size: 0.68rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }
.index-stat .value { font-size: 1rem; color: #c8a96e; font-weight: 500; }

/* Chat bubbles */
.user-bubble {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 8px auto;
    max-width: 80%;
    color: #e8e4dc;
    font-size: 0.95rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c8a96e;
    border-radius: 4px 16px 16px 16px;
    padding: 16px 20px;
    margin: 8px auto 8px 0;
    max-width: 88%;
    color: #e8e4dc;
    font-size: 0.95rem;
    line-height: 1.7;
}

.source-pill {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #888;
    margin: 3px 3px 3px 0;
}

/* Buttons — white text always */
.stButton > button {
    background: #c8a96e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
.stButton > button * { color: #ffffff !important; }

/* Inputs */
[data-testid="stFileUploader"] {
    background: #161616;
    border: 1px dashed #333;
    border-radius: 10px;
    padding: 8px;
}
[data-testid="stChatInput"] textarea {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e4dc !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
input[type="password"], input[type="text"] {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e4dc !important;
    border-radius: 8px !important;
}

hr { border-color: #2a2a2a !important; }
[data-testid="stSlider"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stExpander"] {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #888 !important; }

.stSlider > div > div { background: transparent !important; }


.empty-state { text-align: center; padding: 3rem 0; }
.empty-state .icon { font-size: 2rem; margin-bottom: 0.8rem; }
.empty-state .msg  { font-size: 0.95rem; color: #555; }
.empty-state .sub  { font-size: 0.78rem; color: #3a3a3a; margin-top: 0.3rem; }

.divider { border: none; border-top: 1px solid #2a2a2a; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
MODEL_PATH = "sentence_transformer_models/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH, device="cpu")
    m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    m.save(MODEL_PATH)
    return m

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)

model = load_model()


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "index": None,
    "metadata_store": [],
    "messages": [],
    "indexed_files": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Core functions ────────────────────────────────────────────────────────────
def extract_chunks_from_pdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    chunks = []
    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if text and len(text.split()) >= 10:
                chunks.append({
                    "text": text,
                    "page_number": page_number,
                    "document": os.path.basename(file_path)
                })
    doc.close()
    return chunks


def upload_and_index(uploaded_files) -> int:
    total_chunks = 0
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatL2(384)
    os.makedirs("temp", exist_ok=True)
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.indexed_files:
            continue
        if not uploaded_file.name.lower().endswith(".pdf"):
            continue
        temp_path = os.path.join("temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunks = extract_chunks_from_pdf(temp_path)
        if chunks:
            embeddings = model.encode(
                [c["text"] for c in chunks],
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")
            st.session_state.index.add(embeddings)
            st.session_state.metadata_store.extend(chunks)
            st.session_state.indexed_files.append(uploaded_file.name)
            total_chunks += len(chunks)
    return total_chunks


def retrieve_chunks(query: str, top_k: int) -> List[Dict]:
    if st.session_state.index is None or not st.session_state.metadata_store:
        return []
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    _, I = st.session_state.index.search(query_emb, top_k)
    return [st.session_state.metadata_store[i] for i in I[0] if i < len(st.session_state.metadata_store)]


def generate_answer(query: str, top_chunks: List[Dict], groq_client) -> str:
    context = "\n\n".join([
        f"[{c['document']} — Page {c['page_number']}]\n{c['text'][:600]}"
        for c in top_chunks
    ])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Answer using only the provided context. "
                    "Be concise and clear. Always cite document name and page number."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="pm-title">🧠 PromptMind</div>', unsafe_allow_html=True)
st.markdown('<div class="pm-subtitle">Upload PDFs · Ask questions · Get cited answers</div>', unsafe_allow_html=True)

# ── Section 1: Groq API Key ───────────────────────────────────────────────────
st.markdown('<div class="setup-card"><div class="card-label">Step 1 — Groq API Key</div>', unsafe_allow_html=True)

groq_key = os.getenv("GROQ_API_KEY", "")
if not groq_key:
    col1, col2 = st.columns([4, 1])
    with col1:
        groq_key = st.text_input(
            "Groq key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("[Get free key →](https://console.groq.com)")

groq_client = get_groq_client(groq_key) if groq_key else None

if groq_client:
    st.success("✓ Groq connected — Llama 3.3 70B ready")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Upload + Index ─────────────────────────────────────────────────
st.markdown('<div class="setup-card"><div class="card-label">Step 2 — Upload & Index PDFs</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)
top_k = st.slider("Chunks to retrieve", 1, 10, 5)

if uploaded_files:
    if st.button("⚡ Index PDFs"):
        with st.spinner("Indexing..."):
            total = upload_and_index(uploaded_files)
        if total > 0:
            st.success(f"✅ {total} chunks indexed")
        else:
            st.info("All files already indexed")

if st.session_state.index is not None:
    st.markdown(f"""
    <div class="index-card">
        <div class="index-stat">
            <div class="label">Status</div>
            <div class="value">✓ Ready</div>
        </div>
        <div class="index-stat">
            <div class="label">Chunks</div>
            <div class="value">{len(st.session_state.metadata_store):,}</div>
        </div>
        <div class="index-stat">
            <div class="label">Files</div>
            <div class="value">{len(st.session_state.indexed_files)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.indexed_files:
    with st.expander("📄 Indexed files"):
        for fname in st.session_state.indexed_files:
            st.markdown(f"- {fname}")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Chat ───────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col_title, col_clear = st.columns([5, 1])
with col_title:
    st.markdown("**Ask your documents**")
with col_clear:
    if st.button("🗑 Clear"):
        st.session_state.messages = []
        st.rerun()

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("▶ Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<span class="source-pill">📄 {s["document"]} — p.{s["page_number"]}</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(s["text"][:200] + "...")

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💬</div>
        <div class="msg">Your conversation will appear here</div>
        <div class="sub">Index your PDFs above, then ask anything below</div>
    </div>
    """, unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    if st.session_state.index is None or not st.session_state.metadata_store:
        st.error("Please upload and index PDFs first.")
        st.stop()
    if not groq_client:
        st.error("Please add your Groq API key above.")
        st.stop()

    st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Searching and thinking..."):
        top_chunks = retrieve_chunks(query, top_k)
        answer     = generate_answer(query, top_chunks, groq_client)

    st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
    with st.expander("▶ Sources"):
        for s in top_chunks:
            st.markdown(
                f'<span class="source-pill">📄 {s["document"]} — p.{s["page_number"]}</span>',
                unsafe_allow_html=True
            )
            st.caption(s["text"][:200] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": top_chunks
    })

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;font-size:0.7rem;color:#333;margin-top:2rem">'
    'MiniLM-L6-v2 · FAISS · Llama 3.3 70B via Groq'
    '</div>',
    unsafe_allow_html=True
)
