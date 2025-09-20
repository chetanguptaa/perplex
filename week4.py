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
from numpy import dot
from numpy.linalg import norm

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection_name = "web_docs_week4"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(name=collection_name)

PROMPTS = {
    "concise": """
Use the context below to answer the question.
Give a short, clear response without citing sources.

Question: {query}
Context:
{context}

Answer:
    """,

    "verbose": """
You are an expert research assistant.
Use the provided context to answer the question thoroughly.
If the context is insufficient, say "I don’t know."

Question: {query}
Context:
{context}

Answer:
    """,

    "with_sources": """
Answer the question using the context.
At the end of your response, list the most relevant sources (URLs).

Question: {query}
Context:
{context}

Answer:
    """
}

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
# Gemini Call
# =========================
def ask_gemini(query, retrieved, style="concise"):
    context = "\n\n".join(retrieved)
    prompt = PROMPTS[style].format(query=query, context=context)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

# =========================
# Evaluation Metrics
# =========================
def score_answer(answer: str, ground_truth: str) -> int:
    """String containment check (strict)."""
    return int(ground_truth.lower() in answer.lower())

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def score_answer_semantic(answer: str, ground_truth: str, threshold: float = 0.75) -> int:
    """Embedding similarity check (robust to paraphrases)."""
    ans_emb = embedder.encode(answer)
    gt_emb = embedder.encode(ground_truth)
    sim = cosine_similarity(ans_emb, gt_emb)
    return int(sim >= threshold)

# =========================
# Evaluation Runner
# =========================
def evaluate(dataset_path="week4_eval_dataset.json", prompt_style="concise"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    string_correct = 0
    semantic_correct = 0

    for item in dataset:
        q = item["question"]
        gt = item["ground_truth"]

        # Search + retrieve context
        results_search = search_web(q, max_results=3)
        index_results(results_search)
        retrieval = retrieve(q, n_results=3)
        retrieved_docs = retrieval["documents"][0]

        # Ask Gemini
        answer = ask_gemini(q, retrieved_docs, style=prompt_style)

        # Scores
        s_score = score_answer(answer, gt)
        sem_score = score_answer_semantic(answer, gt)

        string_correct += s_score
        semantic_correct += sem_score

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "string_score": s_score,
            "semantic_score": sem_score
        })

    string_acc = string_correct / len(dataset)
    semantic_acc = semantic_correct / len(dataset)

    report = {
        "string_accuracy": string_acc,
        "semantic_accuracy": semantic_acc,
        "results": results
    }

    with open("week4_eval_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Evaluation complete.")
    print(f"String Accuracy: {string_acc:.2f}")
    print(f"Semantic Accuracy: {semantic_acc:.2f}")

if __name__ == "__main__":
    print("Running Week 4 Evaluation...\n")
    evaluate(dataset_path="week4_eval_dataset.json", prompt_style="concise")
