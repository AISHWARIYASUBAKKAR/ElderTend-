import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks: list, doc_name: str):
    client = chromadb.PersistentClient(path="./data/chroma")
    collection = client.get_or_create_collection("eldertend")
    vectors = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc_name} for _ in chunks]
    )
    print(f"Stored {len(chunks)} chunks from {doc_name}")

if __name__ == "__main__":
    pdf_path = "uploads/sample.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embed_and_store(chunks, "sample.pdf")