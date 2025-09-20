import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from ddgs import DDGS
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# Setup
# =========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()

# Collections
docs_collection = "web_docs_week6"
memory_collection = "chat_memory_week6"

try:
    doc_store = chroma_client.get_collection(docs_collection)
except:
    doc_store = chroma_client.create_collection(docs_collection)

try:
    memory_store = chroma_client.get_collection(memory_collection)
except:
    memory_store = chroma_client.create_collection(memory_collection)

# Short-term memory (rolling buffer)
conversation_history = []
MAX_TURNS = 5

# =========================
# Web Search + Scraping
# =========================
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
            ids.append(f"doc_{idx}")
            docs.append(chunk)
            embs.append(embedder.encode(chunk).tolist())
            idx += 1
    if docs:
        doc_store.add(ids=ids, documents=docs, embeddings=embs)

def retrieve_docs(query, n_results=3):
    query_emb = embedder.encode(query).tolist()
    results = doc_store.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    return results

# =========================
# Long-term Memory
# =========================
def store_memory(question, answer, turn_id):
    """Store Q+A as a single chunk in vector DB."""
    text = f"User: {question}\nAssistant: {answer}"
    emb = embedder.encode(text).tolist()
    memory_store.add(
        ids=[f"turn_{turn_id}"],
        documents=[text],
        embeddings=[emb]
    )

def retrieve_memory(query, n_results=3):
    """Retrieve semantically relevant past chats."""
    query_emb = embedder.encode(query).tolist()
    results = memory_store.query(
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

def ask_gemini(query, retrieved_docs, retrieved_memory):
    history_context = build_history_context()
    docs_context = "\n\n".join(retrieved_docs)
    memory_context = "\n\n".join(retrieved_memory)

    prompt = f"""
You are a helpful assistant. Use the following information to answer:

Conversation history (short-term):
{history_context}

Relevant long-term memory:
{memory_context}

Retrieved documents:
{docs_context}

Question: {query}

Answer:
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    answer = resp.text.strip()

    # Update short-term + long-term memory
    turn_id = len(conversation_history)
    conversation_history.append({"question": query, "answer": answer})
    store_memory(query, answer, turn_id)

    return answer

# =========================
# Main Loop
# =========================
if __name__ == "__main__":
    print("🤖 Week 6 – RAG + Short-term + Long-term Memory\n")
    while True:
        query = input("❓ You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Web docs retrieval
        search_results = search_web(query, max_results=3)
        index_results(search_results)
        retrieval = retrieve_docs(query, n_results=3)
        retrieved_docs = retrieval["documents"][0]

        # Long-term memory retrieval
        retrieved_memories = retrieve_memory(query, n_results=3)
        memory_docs = retrieved_memories["documents"][0] if retrieved_memories["documents"] else []

        # Answer with combined memory
        answer = ask_gemini(query, retrieved_docs, memory_docs)
        print(f"🤖 Assistant: {answer}\n")
