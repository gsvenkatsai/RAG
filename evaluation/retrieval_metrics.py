def calculate_precision(retrieved_chunks, relevant_chunks):
    if len(retrieved_chunks) == 0:
        return 0

    relevant_retrieved = 0

    for chunk in retrieved_chunks:
        if chunk["chunk_id"] in relevant_chunks:
            relevant_retrieved += 1

    precision = relevant_retrieved / len(retrieved_chunks)

    return round(precision, 2)


def calculate_recall(retrieved_chunks, relevant_chunks):
    if len(relevant_chunks) == 0:
        return 0

    relevant_retrieved = 0

    for chunk in retrieved_chunks:
        if chunk["chunk_id"] in relevant_chunks:
            relevant_retrieved += 1

    recall = relevant_retrieved / len(relevant_chunks)

    return round(recall, 2)