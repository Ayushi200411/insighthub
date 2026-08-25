from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FOLDER = "rag/vectorstore"
VECTOR_TOP_K = 10   # candidates from vector search
BM25_TOP_K = 10      # candidates from keyword search
FINAL_TOP_K = 3      # final chunks after re-ranking

def load_index():
    index = faiss.read_index(os.path.join(INDEX_FOLDER, "index.faiss"))
    with open(os.path.join(INDEX_FOLDER, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(INDEX_FOLDER, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    return index, data["chunks"], data["sources"], bm25

def hybrid_retrieve(query, embed_model, index, chunks, sources, bm25):
    # Vector search
    query_embedding = embed_model.encode([query]).astype("float32")
    _, vector_indices = index.search(query_embedding, VECTOR_TOP_K)
    vector_candidates = set(vector_indices[0])

    # BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]
    bm25_candidates = set(bm25_top_indices)

    # Combine (union of both methods' results)
    combined_indices = list(vector_candidates | bm25_candidates)

    candidates = [{"text": chunks[i], "source": sources[i], "idx": i} for i in combined_indices]
    return candidates

def rerank(query, candidates, cross_encoder, top_k=FINAL_TOP_K):
    pairs = [[query, c["text"]] for c in candidates]
    scores = cross_encoder.predict(pairs)

    for c, score in zip(candidates, scores):
        c["score"] = score

    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer (mention which source(s) you used):"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

def main():
    print("Loading models and index...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index, chunks, sources, bm25 = load_index()

    print("\nReady. Ask a question (or type 'exit' to quit).\n")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            break

        candidates = hybrid_retrieve(query, embed_model, index, chunks, sources, bm25)
        reranked = rerank(query, candidates, cross_encoder)

        print("\nTop sources after re-ranking:")
        for r in reranked:
            print(f"  {r['source']} (score: {r['score']:.3f})")

        answer = generate_answer(query, reranked)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()