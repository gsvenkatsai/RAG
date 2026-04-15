from    retrieval_metrics import (
    calculate_precision,
    calculate_recall
)

retrieved_chunks = [
    {"chunk_id": 1},
    {"chunk_id": 2},
    {"chunk_id": 3},
    {"chunk_id": 4},
    {"chunk_id": 5}
]

relevant_chunks = [2, 3, 4, 8]

precision = calculate_precision(
    retrieved_chunks,
    relevant_chunks
)

recall = calculate_recall(
    retrieved_chunks,
    relevant_chunks
)

print("Precision:", precision)
print("Recall:", recall)