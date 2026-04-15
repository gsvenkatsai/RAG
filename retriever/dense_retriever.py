from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from retriever.bm25_retriever import setup_bm25
# load model once (used to convert text into embeddings/vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

# dummy global DB (later you load real one)
index = None          # FAISS index (vector database) — currently not initialized
documents = []        # stores original text corresponding to embeddings

from ingestion import load_all_texts, chunk_text

def setup_vector_db():
    global index, documents

    data = load_all_texts("data")

    documents = []

    for file in data:
        chunks = chunk_text(file["text"])

        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": file["source"],
                "chunk_id": i
            })
    setup_bm25(documents)
    embeddings = model.encode([doc["text"] for doc in documents])

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

def embed_query(query):
    # converts a single query string into embedding vector
    # model expects a list, so we pass [query]
    return model.encode([query])


def vector_db_search(query_embedding, k=5, threshold=1.5):
    distances, indices = index.search(query_embedding, k)
    print(distances)
    results = []
    scores = []
    for i, idx in enumerate(indices[0]):
        dist = distances[0][i]

        if dist <= threshold:
            results.append(documents[idx])
            scores.append(dist)

    if len(results) == 0:
        return None, []
    results = results[:3]
    return results, scores

def retrieve(query, k=5):
    query_embedding = model.encode([query])

    docs, scores = vector_db_search(
        query_embedding,
        k=k
    )

    if docs is None:
        return []

    return docs