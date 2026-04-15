from retriever.dense_retriever import retrieve
from retriever.bm25_retriever import bm25_search
from retriever.query_transformer import rewrite_query

def hybrid_search(query, k=5):
    query = rewrite_query(query)
    dense_results = retrieve(query, k=k)
    bm25_results = bm25_search(query, top_k=k)

    merged_results = []

    for doc in dense_results:
        merged_results.append({
            "text": doc["text"],
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "retrieval_type": "dense",
            "retrieval_score": 0.7
        })
    for item in bm25_results:
        doc = item["document"]

        already_exists = any(
            existing["source"] == doc["source"] and
            existing["chunk_id"] == doc["chunk_id"]
            for existing in merged_results
        )

        if not already_exists:
            merged_results.append({
                "text": doc["text"],
                "source": doc["source"],
                "chunk_id": doc["chunk_id"],
                "retrieval_type": "bm25",
                "retrieval_score": item["score"]
            })
    
    merged_results = sorted(
        merged_results,
        key=lambda x: x["retrieval_score"],
        reverse=True
    )
    return merged_results

def fallback_retrieval(query):
    return hybrid_search(query, k=10)