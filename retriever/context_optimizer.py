def remove_redundant_chunks(results):
    unique_results = []
    seen_texts = set()

    for item in results:
        normalized_text = item["text"].lower().strip()

        if normalized_text not in seen_texts:
            unique_results.append(item)
            seen_texts.add(normalized_text)

    return unique_results

def compress_context(results, max_words=20):
    compressed_results = []

    for item in results:
        words = item["text"].split()

        compressed_text = " ".join(words[:max_words])

        new_item = item.copy()
        new_item["text"] = compressed_text

        compressed_results.append(new_item)

    return compressed_results

def limit_context_size(results, max_total_words=100):
    limited_results = []
    total_words = 0

    for item in results:
        chunk_words = len(item["text"].split())

        if total_words + chunk_words > max_total_words:
            break

        limited_results.append(item)
        total_words += chunk_words

    return limited_results


def prioritize_chunks(results):
    prioritized_results = sorted(
        results,
        key=lambda x: x.get("final_score", 0),
        reverse=True
    )

    return prioritized_results