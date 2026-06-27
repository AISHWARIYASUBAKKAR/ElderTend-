import streamlit as st
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chunk_and_store(text, doc_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    client = chromadb.PersistentClient(path="./data/chroma")
    collection = client.get_or_create_collection("eldertend")
    vectors = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc_name} for _ in chunks]
    )
    return len(chunks)

def answer_question(query, chat_history):
    client = chromadb.PersistentClient(path="./data/chroma")
    collection = client.get_or_create_collection("eldertend")
    query_vector = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    chunks = results['documents'][0]
    context = "\n\n".join(chunks)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant. Answer ONLY from the provided context. If the answer is not in the context, say 'I cannot find that information in the document.'"
        }
    ]

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content, chunks

st.title("ElderTend")
st.caption("Ask questions about your medical documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded = st.file_uploader("Upload a medical PDF", type=["pdf"])

if uploaded:
    save_path = f"uploads/{uploaded.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    with st.spinner("Processing document..."):
        text = extract_text(save_path)
        chunks = chunk_and_store(text, uploaded.name)
    st.success(f"Ready! Processed {chunks} chunks from {uploaded.name}")

st.divider()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question, st.session_state.chat_history)
        st.write(answer)
        with st.expander("View source passages"):
            for i, chunk in enumerate(sources):
                st.markdown(f"**Source {i+1}:**")
                st.caption(chunk)
                st.divider()
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })