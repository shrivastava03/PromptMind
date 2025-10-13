import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. Load Sentence Transformer Model
# ----------------------------
MODEL_PATH = "sentence_transformer_models/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(MODEL_PATH)
    else:
        model = SentenceTransformer(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# 2. Extract Chunks from PDF
# ----------------------------
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

# ----------------------------
# 3. Initialize FAISS Index & Metadata
# ----------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = []

# ----------------------------
# 4. Upload and Index PDFs
# ----------------------------
def upload_pdfs(uploaded_files):
    total_chunks = 0
    metadata_store = st.session_state.metadata_store
    index = st.session_state.index

    if index is None:
        index = faiss.IndexFlatL2(384)  # 384 dims for all-MiniLM-L6-v2

    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            # Save the uploaded file temporarily
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract and embed chunks
            chunks = extract_chunks_from_pdf(temp_path)
            if chunks:
                embeddings = model.encode([c["text"] for c in chunks], convert_to_numpy=True)
                index.add(embeddings)
                metadata_store.extend(chunks)
                total_chunks += len(chunks)

    st.session_state.index = index
    st.session_state.metadata_store = metadata_store

    return total_chunks

# ----------------------------
# 5. Query the Indexed PDFs
# ----------------------------
def query_docs(query: str, top_k: int = 5):
    index = st.session_state.index
    metadata_store = st.session_state.metadata_store

    if index is None or len(metadata_store) == 0:
        return "⚠️ Please upload and index PDFs first."

    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = []

    for idx in I[0]:
        chunk = metadata_store[idx]
        results.append(f"📄 **{chunk['document']}** — Page {chunk['page_number']}\n\n> {chunk['text']}")

    return "\n\n---\n\n".join(results)

# ----------------------------
# 6. Streamlit App UI
# ----------------------------
st.set_page_config(
    page_title="PromptMind 📚",
    page_icon="📘",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>📚 PromptMind</h1>
    <p style='text-align:center; font-size:18px;'>Fine-tuned Conversations, Every Time.</p>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("⚙️ Control Panel")
top_k = st.sidebar.slider("Top K Results", 1, 20, 5)
st.sidebar.info("Upload multiple PDFs and query them instantly!")

# File uploader
uploaded_files = st.file_uploader("📤 Upload your PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Index PDFs"):
        with st.spinner("Indexing documents... This may take a few seconds ⏳"):
            total_chunks = upload_pdfs(uploaded_files)
            st.success(f"✅ Indexed {total_chunks} text chunks from {len(uploaded_files)} PDFs.")
else:
    st.warning("Please upload at least one PDF to start indexing.")

# Query section
st.markdown("---")
query = st.text_input("🔍 Enter your query", placeholder="Type your question here...")

if st.button("Search"):
    if query.strip():
        with st.spinner("Searching relevant information..."):
            result = query_docs(query, top_k)
        st.markdown(result)
    else:
        st.warning("Please enter a query before searching.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🚀 Powered by FAISS + Sentence Transformers + Streamlit</p>",
    unsafe_allow_html=True
)
