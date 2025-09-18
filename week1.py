import os
from dotenv import load_dotenv
import google.generativeai as genai
from ddgs import DDGS
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

def test_gemini():
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content("Hello Gemini, explain Retrieval-Augmented Generation (RAG) in 2 sentences.")
    print("\n🧠 Gemini says:\n", resp.text)

def test_search(query="What is LangGraph?"):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
    print("\n🔎 Search Results:")
    for r in results:
        print(f"- {r['title']} ({r['href']})\n  {r['body']}")
    return results

def test_vector_db():
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = [
        {"id": "1", "text": "LangGraph is a library for building agentic workflows with memory."},
        {"id": "2", "text": "Chroma is a vector database designed for semantic search and embeddings."}
    ]

    for d in docs:
        collection.add(
            ids=[d["id"]],
            documents=[d["text"]],
            embeddings=[embedder.encode(d["text"]).tolist()]
        )

    query = "who is batman?"
    results = collection.query(
        query_embeddings=[embedder.encode(query).tolist()],
        n_results=2
    )
    print("\n📂 Vector DB Query Results:", results)

if __name__ == "__main__":
    test_gemini()
    test_search()
    test_vector_db()
