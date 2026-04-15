# retrieval_failure_handler.py

def is_empty_retrieval(results):
    return len(results) == 0


from config import LOW_QUALITY_SCORE_THRESHOLD

def is_low_quality_retrieval(results):
    if not results:
        return True

    best_score = results[0]["rerank_score"]

    return best_score < LOW_QUALITY_SCORE_THRESHOLD

# retrieval_failure_handler.py

def log_retrieval_failure(query, reason):
    print("\n[RETRIEVAL FAILURE]")
    print(f"Query: {query}")
    print(f"Reason: {reason}")


# retrieval_failure_handler.py

def build_safe_response(reason):
    if reason == "empty retrieval":
        return {
            "answer": "I could not find any relevant information for that query.",
            "sources": []
        }

    if reason == "low quality retrieval":
        return {
            "answer": "I found some information, but it was not reliable enough to answer safely.",
            "sources": []
        }

    return {
        "answer": "I do not know.",
        "sources": []
    }