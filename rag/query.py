from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FOLDER = "rag/vectorstore"
TOP_K = 3  # how many chunks to retrieve per question

def load_index():
    index = faiss.read_index(os.path.join(INDEX_FOLDER, "index.faiss"))
    with open(os.path.join(INDEX_FOLDER, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["sources"]

def retrieve(query, model, index, chunks, sources, k=TOP_K):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        results.append({"text": chunks[idx], "source": sources[idx]})
    return results

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
    print("Loading embedding model and index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks, sources = load_index()

    print("\nReady. Ask a question (or type 'exit' to quit).\n")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            break

        retrieved = retrieve(query, model, index, chunks, sources)
        print("\nRetrieved from:", [r["source"] for r in retrieved])

        answer = generate_answer(query, retrieved)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()