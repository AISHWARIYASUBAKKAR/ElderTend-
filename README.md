# ElderTend 🏥

A RAG-based medical document assistant that answers questions about patient records.

## What it does
- Upload any medical PDF (discharge summary, lab report, progress notes)
- Ask questions in plain English
- Get grounded answers with source citations
- Full chat history with conversational context

## Tech Stack
- **LLM**: Llama 3.1 via Groq
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Framework**: LangChain + Streamlit
- **Pipeline**: RAG (Retrieval Augmented Generation)

## Live Demo
[Try ElderTend here](https://dcxpwx2rwwacdyynlpw5ng.streamlit.app)

## Run Locally
```bash
git clone https://github.com/AISHWARIYASUBAKKAR/ElderTend-.git
cd ElderTend-
pip install -r requirements.txt
streamlit run app.py
```
