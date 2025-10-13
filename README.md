---
title: PromptMind
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: docker
sdk_version: 5.42.0
app_file: app.py
pinned: false
short_description: RAG based system for top results.
---

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# 📚 PromptMind

**PromptMind** is a semantic search app for multiple PDF documents, built with **Gradio**, **FAISS**, and **Sentence Transformers**.

## 🚀 Features
- Upload multiple PDFs at once
- Automatically chunk text from each page
- Create embeddings with `all-MiniLM-L6-v2`
- Search across all uploaded PDFs using semantic similarity
- Instant results with document and page references

## 🛠️ Tech Stack
- **Python**
- **StreamLit** for the interactive UI
- **PyMuPDF** for PDF text extraction
- **Sentence Transformers** for embeddings
- **FAISS** for vector search

## 📄 How to Use
Click Upload PDFs and select one or more PDF files.

Press Index PDFs to process and embed them.

Type a query and press Search.

See matching results with document names and page numbers.

## 🛠️ Setup Instructions
```bash
git clone https://github.com/<your-username>/PromptMind.git
cd PromptMind
pip install -r requirements.txt
streamlit run app.py
