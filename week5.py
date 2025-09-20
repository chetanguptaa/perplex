import os
import requests
from bs4 import BeautifulSoup
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
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection_name = "web_docs_week5"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)

# Short-term memory (last N turns)
conversation_history = []
MAX_TURNS = 5

def search_web(query, max_results=3):
    with DDGS() as ddgs:
        return [r for r in ddgs.text(query, max_results=max_results)]

def fetch_page_text(url, max_paragraphs=8):
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs[:max_paragraphs])
    except Exception:
        return ""

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def index_results(results):
    ids, docs, embs = [], [], []
    idx = 0
    for r in results:
        page_text = fetch_page_text(r["href"])
        if not page_text:
            continue
        chunks = chunk_text(page_text)
        for chunk in chunks:
            ids.append(str(idx))
            docs.append(chunk)
            embs.append(embedder.encode(chunk).tolist())
            idx += 1
    if docs:
        collection.add(ids=ids, documents=docs, embeddings=embs)

def retrieve(query, n_results=3):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    return results

# =========================
# Gemini with Memory
# =========================
def build_history_context():
    """Format last N conversation turns into context string."""
    history = conversation_history[-MAX_TURNS:]
    formatted = []
    for turn in history:
        formatted.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n".join(formatted)

def ask_gemini(query, retrieved_docs):
    history_context = build_history_context()
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a helpful assistant. Use both the conversation history and retrieved documents to answer.

Conversation history:
{history_context}

Question: {query}

Retrieved context:
{context}

Answer:
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    answer = resp.text.strip()

    # Update memory
    conversation_history.append({"question": query, "answer": answer})

    return answer

# =========================
# Main Loop
# =========================
if __name__ == "__main__":
    print("🤖 Week 5 – RAG + Short-term Memory\n")
    while True:
        query = input("❓ You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Web search + retrieval
        search_results = search_web(query, max_results=3)
        index_results(search_results)
        retrieval = retrieve(query, n_results=3)
        retrieved_docs = retrieval["documents"][0]

        # Answer with memory
        answer = ask_gemini(query, retrieved_docs)
        print(f"🤖 Assistant: {answer}\n")
