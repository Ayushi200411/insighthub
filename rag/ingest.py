import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

DOCS_FOLDER = "documents"
INDEX_FOLDER = "rag/vectorstore"
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between chunks

def load_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    chunk_sources = []  # tracks which file + chunk number each chunk came from

    print("Reading and chunking PDFs...")
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            text = load_pdf_text(filepath)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_sources.append(f"{filename} (chunk {i+1})")
            print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("Embedding chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_FOLDER, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_FOLDER, "index.faiss"))

    with open(os.path.join(INDEX_FOLDER, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": chunk_sources}, f)

    print(f"\nDone. Index and chunks saved to {INDEX_FOLDER}/")

if __name__ == "__main__":
    main()