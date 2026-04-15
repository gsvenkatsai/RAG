from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# reranked_results = rerank_results(query, merged_results)
def rerank_results(query, results):
    pairs = []

    for item in results:
        pairs.append([query, item["text"]])

    scores = reranker_model.predict(pairs)

    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)

        retrieval_score = results[i].get("retrieval_score", 0)

        results[i]["final_score"] = retrieval_score + float(score)

    reranked_results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return reranked_results
