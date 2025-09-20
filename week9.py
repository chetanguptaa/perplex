import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from ddgs import DDGS
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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

docs_collection = "web_docs_week9"
try:
    doc_store = chroma_client.get_collection(docs_collection)
except:
    doc_store = chroma_client.create_collection(docs_collection)

# For BM25
bm25_corpus = []     # list of tokenized docs
bm25_docs = []       # raw docs
bm25_index = None

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
    global bm25_corpus, bm25_docs, bm25_index

    ids, docs, embs = [], [], []
    idx = len(bm25_docs)

    for r in results:
        page_text = fetch_page_text(r["href"])
        if not page_text:
            continue
        chunks = chunk_text(page_text)
        for chunk in chunks:
            ids.append(f"doc_{idx}")
            docs.append(chunk)
            embs.append(embedder.encode(chunk).tolist())

            # BM25
            bm25_docs.append(chunk)
            bm25_corpus.append(chunk.lower().split())
            idx += 1

    if docs:
        doc_store.add(ids=ids, documents=docs, embeddings=embs)

    # Rebuild BM25 index
    if bm25_docs:
        bm25_index = BM25Okapi(bm25_corpus)

# =========================
# Hybrid Retrieval
# =========================
def retrieve_vector(query, n_results=3):
    query_emb = embedder.encode(query).tolist()
    results = doc_store.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    return list(zip(results["documents"][0], results["distances"][0]))

def retrieve_bm25(query, n_results=3):
    if not bm25_index:
        return []
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [(bm25_docs[i], scores[i]) for i in top_idx]

def hybrid_retrieve(query, top_k=5):
    vector_results = retrieve_vector(query, n_results=top_k)
    bm25_results = retrieve_bm25(query, n_results=top_k)

    # Normalize scores
    if vector_results:
        max_v = max([1e-6] + [1 - d for _, d in vector_results])
    else:
        max_v = 1
    if bm25_results:
        max_b = max([1e-6] + [s for _, s in bm25_results])
    else:
        max_b = 1

    scored = []
    for doc, dist in vector_results:
        score = (1 - dist) / max_v
        scored.append((doc, score, "vector"))
    for doc, scr in bm25_results:
        score = scr / max_b
        scored.append((doc, score, "bm25"))

    # Merge + sort
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# =========================
# Gemini Answering
# =========================
def ask_gemini(query, retrieved_docs):
    context = "\n\n".join([doc for doc, _, _ in retrieved_docs])
    prompt = f"""
Answer the following question using the retrieved context.
If the context is insufficient, say "I don't know."

Question: {query}

Retrieved context:
{context}

Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

# =========================
# Main Loop
# =========================
if __name__ == "__main__":
    print("🤖 Week 9 – Hybrid RAG (Vector + BM25)\n")
    while True:
        query = input("❓ You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        search_results = search_web(query, max_results=3)
        index_results(search_results)

        retrieved = hybrid_retrieve(query, top_k=5)
        answer = ask_gemini(query, retrieved)

        print(f"🤖 Assistant: {answer}\n")
        print("📚 Sources:", [src for _, _, src in retrieved], "\n")
