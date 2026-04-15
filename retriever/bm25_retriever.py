from rank_bm25 import BM25Okapi

bm25 = None
documents = []

def setup_bm25(all_documents):
    global bm25, documents

    documents = all_documents

    # Each Document is converted into array of words
    tokenized_docs = [
        doc["text"].lower().split()
        for doc in documents
    ]

    # Builds BM25 index using tokenized documents.This is similar to how FAISS builds index from embeddings.
    bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, top_k=5):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query) # Generates scores for each doc

    ranked = sorted(
        enumerate(scores),  # [7.8, 0.2, 2.5] -> [(0, 7.8), (1, 0.2), (2, 2.5)] = (document_index, score)
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:
        results.append({
            "score": score,
            "document": documents[idx]
        })

    return results