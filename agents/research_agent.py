import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FOLDER = "rag/vectorstore"
VECTOR_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 3

_embed_model = None
_cross_encoder = None
_index = None
_chunks = None
_sources = None
_bm25 = None

def _load_once():
    global _embed_model, _cross_encoder, _index, _chunks, _sources, _bm25
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _index = faiss.read_index(os.path.join(INDEX_FOLDER, "index.faiss"))
        with open(os.path.join(INDEX_FOLDER, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)
        _chunks, _sources = data["chunks"], data["sources"]
        with open(os.path.join(INDEX_FOLDER, "bm25.pkl"), "rb") as f:
            _bm25 = pickle.load(f)

def run_research_agent(query):
    _load_once()

    query_embedding = _embed_model.encode([query]).astype("float32")
    _, vector_indices = _index.search(query_embedding, VECTOR_TOP_K)
    vector_candidates = set(vector_indices[0])

    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    bm25_candidates = set(bm25_top_indices)

    combined_indices = list(vector_candidates | bm25_candidates)
    candidates = [{"text": _chunks[i], "source": _sources[i]} for i in combined_indices]

    pairs = [[query, c["text"]] for c in candidates]
    scores = _cross_encoder.predict(pairs)
    for c, score in zip(candidates, scores):
        c["score"] = score
    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:FINAL_TOP_K]

    context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in reranked])
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer (mention which source(s) you used):"""

    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return {"answer": response.text, "sources": [c["source"] for c in reranked]}