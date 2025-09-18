import os
import json
import requests
import sqlite3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from ddgs import DDGS
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma DB (default for now)
chroma_client = chromadb.Client()
collection_name = "web_docs_week3"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)



# SQLite logging
conn = sqlite3.connect("conversations.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT
)
""")
conn.commit()


# ========== Helper Functions ==========
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


def ask_gemini(query, retrieved):
    context = "\n\n".join(retrieved)
    prompt = f"""
Use the following context from web pages to answer the user question.
Answer concisely. Do not include citations, only the response.

Question: {query}

Context:
{context}

Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()


def log_conversation(question, answer):
    c.execute("INSERT INTO logs (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()


def get_logs():
    c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 10")
    return c.fetchall()


# ========== Streamlit UI ==========
st.set_page_config(page_title="Week 3", layout="wide")
st.title("🤖 Week 3")

query = st.text_input("Ask a question:", "")

if st.button("Search & Answer") and query.strip():
    st.write("🔎 Searching and scraping...")
    results = search_web(query, max_results=3)
    index_results(results)

    retrieval = retrieve(query, n_results=3)
    retrieved_docs = retrieval["documents"][0]

    st.write("🤖 Generating answer...")
    answer = ask_gemini(query, retrieved_docs)

    # Log to DB
    log_conversation(query, answer)

    # Show result
    st.json({"answer": answer})

st.subheader("📜 Conversation Log (last 10)")
for row in get_logs():
    st.write(f"Q: {row[1]}")
    st.write(f"A: {row[2]}")
    st.write("---")
