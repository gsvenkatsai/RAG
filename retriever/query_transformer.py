import re

def normalize_query(query):
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)
    query = re.sub(r"\s+", " ", query)
    return query


def rewrite_query(query):
    replacements = {
        "rag": "retrieval augmented generation",
        "db": "database",
        "ml": "machine learning",
        "ai": "artificial intelligence"
    }

    words = query.split()

    rewritten_words = []

    for word in words:
        if word in replacements:
            rewritten_words.append(replacements[word])
        else:
            rewritten_words.append(word)

    return " ".join(rewritten_words)


def handle_vague_query(query):
    vague_words = ["it", "this", "that", "thing", "stuff"]

    if len(query.split()) <= 2:
        query += " explanation"

    for word in vague_words:
        query = query.replace(word, "")

    return query.strip()


def expand_query(query, context=None):
    if context:
        return f"{query} in {context}"

    return query


def generate_queries(query):
    queries = [query]

    if "rag" in query:
        queries.append(query.replace("rag", "retrieval augmented generation"))
        queries.append("explain rag system")
        queries.append("rag architecture")

    if "faiss" in query:
        queries.append("vector similarity search")
        queries.append("faiss index")
        queries.append("faiss retrieval")

    return list(set(queries))

def process_query(query, context=None):
    query = normalize_query(query)

    generated_queries = generate_queries(query)

    final_queries = []

    for q in generated_queries:
        q = rewrite_query(q)
        q = handle_vague_query(q)
        q = expand_query(q, context)

        final_queries.append(q)

    return list(set(final_queries))

queries = process_query(
    "What is RAG in AI?",
    context="retrieval systems"
)

print(queries)