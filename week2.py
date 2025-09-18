import os
import json
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
collection = chroma_client.create_collection(name="web_docs")


# ========== 1. Web Search ==========
def search_web(query, max_results=5):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return results


# ========== 2. Scrape Page Text ==========
def fetch_page_text(url, max_paragraphs=10):
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs[:max_paragraphs])
        return text
    except Exception as e:
        return ""


# ========== 3. Chunk Text ==========
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# ========== 4. Index Search Results into Vector DB ==========
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


# ========== 5. Retrieve Relevant Chunks ==========
def retrieve(query, n_results=3):
    query_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    return results


# ========== 6. Ask Gemini and return JSON ==========
def ask_gemini(query, retrieved):
    context = "\n\n".join(retrieved)
    prompt = f"""
Use the following context from web pages to answer the user question.
Give only the answer, no citations, no explanations, just a clean response.

Question: {query}

Context:
{context}

Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return {"answer": resp.text.strip()}


# ========== Main ==========
if __name__ == "__main__":
    user_query = input("\n❓ Enter your question: ")

    print("\n🔎 Searching and scraping web pages...")
    search_results = search_web(user_query, max_results=3)

    index_results(search_results)

    retrieval = retrieve(user_query, n_results=3)
    retrieved_docs = retrieval["documents"][0]

    result = ask_gemini(user_query, retrieved_docs)

    print(json.dumps(result, indent=2, ensure_ascii=False))
