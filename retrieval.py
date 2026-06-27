import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve_chunks(query: str, n_results: int = 5) -> list:
    db = chromadb.PersistentClient(path="./data/chroma")
    collection = db.get_or_create_collection("eldertend")
    query_vector = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )
    return results['documents'][0]

def answer_question(query: str) -> str:
    chunks = retrieve_chunks(query)
    context = "\n\n".join(chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Answer ONLY from the provided context. If the answer is not in the context, say 'I cannot find that information in the document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What medications were prescribed?"
    print(f"Question: {question}")
    print(f"\nAnswer: {answer_question(question)}")