from config import (
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    FINAL_CONTEXT_K
)

def validate_retrieval_parameters():
    if RETRIEVAL_TOP_K < RERANK_TOP_K:
        raise ValueError(
            "RETRIEVAL_TOP_K must be >= RERANK_TOP_K"
        )

    if RERANK_TOP_K < FINAL_CONTEXT_K:
        raise ValueError(
            "RERANK_TOP_K must be >= FINAL_CONTEXT_K"
        )

    if FINAL_CONTEXT_K <= 0:
        raise ValueError(
            "FINAL_CONTEXT_K must be greater than 0"
        )