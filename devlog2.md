# 📘 Dev Log — Week 5 Topic 1: Hybrid Retrieval System

---

# 🎯 Goal

Improve retrieval quality by combining:

1. Dense Retrieval (Embeddings + FAISS)
2. Sparse Retrieval (BM25)

Until now the system used only dense retrieval:

```text id="qkdbw7"
Query → Embedding → FAISS → Top Chunks
```

Problem:

Dense retrieval is good at semantic meaning but weak for:

- Exact keywords
- Error messages
- Acronyms
- Rare technical terms

Example:

```text id="bxt6cv"
postgres connection refused
```

Dense retrieval may return:

```text id="f7v9a7"
database connection issue
backend failed to connect to database
```

BM25 can directly return:

```text id="8cskha"
postgres connection refused on port 5432
```

So hybrid retrieval uses both.

---

# 🛣 Suggested Build Order

```text id="r4c4d5"
1. Keep existing dense retriever
2. Create BM25 retriever
3. Run both retrievers
4. Merge results
5. Remove duplicates
6. Add weighting
```

---

# 📂 Folder Structure

```text id="7w8p04"
rag/
├── retriever/
│   ├── __init__.py
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   └── hybrid_retriever.py
├── prompt_builder.py
├── generator.py
├── pipeline.py
├── main.py
```

---

# 🔹 Subtopic 1: Combine Dense Embeddings with BM25

---

## Dense Retrieval

Already existed in project.

Uses:

- SentenceTransformer
- FAISS
- Embeddings

Example:

```python id="cnzj7z"
def retrieve(query):
    query_embedding = model.encode([query])

    docs = vector_db_search(query_embedding)

    if docs is None:
        return None

    return docs
```

---

## BM25 Retriever

Created new file:

```text id="0m3e6x"
retriever/bm25_retriever.py
```

Code:

```python id="6yjlwm"
from rank_bm25 import BM25Okapi

bm25 = None
documents = []

def setup_bm25(all_documents):
    global bm25, documents

    documents = all_documents

    tokenized_docs = [
        doc["text"].lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, top_k=5):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        enumerate(scores),
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
```

---

## How BM25 Works

Example query:

```python id="syiql8"
query = "redis worker timeout"
```

Tokenized:

```python id="szjlwm"
["redis", "worker", "timeout"]
```

BM25 checks all documents and gives score:

```python id="0hr4qj"
[7.8, 0.2, 2.5]
```

Higher score = better keyword match.

---

## Update Dense Retriever Setup

Inside `dense_retriever.py`:

```python id="5rgm82"
from retriever.bm25_retriever import setup_bm25
```

Inside `setup_vector_db()`:

```python id="e2kux7"
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
```

---

## Flow

```text id="8jlwmw"
load files
→ chunk files
→ create documents
→ setup_bm25(documents)
→ create embeddings
→ build FAISS index
```

---

# 🔹 Subtopic 2: Run Parallel Retrieval Pipelines

Goal:

Run both retrievers from one function.

Created:

```text id="8g4tn2"
retriever/hybrid_retriever.py
```

Code:

```python id="by4apd"
from retriever.dense_retriever import retrieve
from retriever.bm25_retriever import bm25_search

def hybrid_search(query):
    dense_results = retrieve(query)
    bm25_results = bm25_search(query)

    return {
        "dense": dense_results,
        "bm25": bm25_results
    }
```

This is not true threading yet.

It only means both retrieval systems are called from one place.

---

# 🔹 Subtopic 3: Merge and Rank Results

Problem:

Hybrid search returns two separate lists.

Need one merged list.

Updated `hybrid_search()`:

```python id="8yjlwm"
from retriever.dense_retriever import retrieve
from retriever.bm25_retriever import bm25_search

def hybrid_search(query):
    dense_results = retrieve(query)
    bm25_results = bm25_search(query)

    merged_results = []

    for doc in dense_results:
        merged_results.append({
            "text": doc["text"],
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "retrieval_type": "dense"
        })

    for item in bm25_results:
        doc = item["document"]

        merged_results.append({
            "text": doc["text"],
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "retrieval_type": "bm25",
            "score": item["score"]
        })

    return merged_results
```

---

# 🔹 Subtopic 4: Handle Conflicts Between Retrieval Methods

Problem:

Same chunk may appear in both dense and BM25 results.

Need duplicate removal.

Code:

```python id="jlwmzd"
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
            "score": item["score"]
        })
```

Logic:

If same source + chunk_id already exists, skip duplicate BM25 chunk.

Dense is currently given first preference because semantic relevance is usually stronger.

---

# 🔹 Subtopic 5: Tune Weighting Between Sparse and Dense Retrieval

Problem:

Currently dense always comes first.

Need score balancing.

Example weights:

```python id="1vcg2n"
dense_weight = 0.7
bm25_weight = 0.3
```

Dense update:

```python id="jlwm7f"
for doc in dense_results:
    merged_results.append({
        "text": doc["text"],
        "source": doc["source"],
        "chunk_id": doc["chunk_id"],
        "retrieval_type": "dense",
        "final_score": 0.7
    })
```

BM25 update:

```python id="jjlwm1"
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
            "score": item["score"],
            "final_score": item["score"] * 0.3
        })
```

Sort final list:

```python id="tjlwm5"
merged_results = sorted(
    merged_results,
    key=lambda x: x["final_score"],
    reverse=True
)
```

---

# 🧪 Main Test Code

```python id="jlwmc3"
from retriever.dense_retriever import setup_vector_db
from retriever.hybrid_retriever import hybrid_search

setup_vector_db()

query = "What is RAG"

results = hybrid_search(query)

print("\n=== MERGED RESULTS ===")

for item in results:
    print(item["text"])
    print(f"Source: {item['source']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Retrieved By: {item['retrieval_type']}")

    if "score" in item:
        print(f"BM25 Score: {item['score']:.2f}")

    if "final_score" in item:
        print(f"Final Score: {item['final_score']:.2f}")

    print("-" * 50)
```

# Dev Log — Week 5 Topic 2: Query Transformation Layer

# Goal

Improve user queries before retrieval.

Problem:

Users often ask:

```text id="w0b2os"
what is it
db issue
what is rag
```

These are weak queries.

Need to:

1. Normalize query
2. Rewrite query
3. Handle vague queries
4. Expand query with context
5. Generate multiple queries

---

# Folder Structure

```text id="rqbbgm"
rag/
├── retriever/
│   ├── query_transformer.py
```

---

# normalize_query()

Purpose:

Make query format consistent before retrieval.

Code:

```python id="jlwm2a"
import re

def normalize_query(query):
    query = query.lower().strip()

    query = re.sub(r"[^\w\s]", "", query)

    query = re.sub(r"\s+", " ", query)

    return query
```

Example:

```python id="4q8s8s"
normalize_query("  WHAT is RAG?? ")
```

Output:

```python id="kjlwm4"
what is rag
```

---

# rewrite_query()

Purpose:

Replace short forms with full forms.

Code:

```python id="p1puh1"
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
```

Example:

```python id="jlwm3q"
rewrite_query("what is rag in ai")
```

Output:

```python id="jlwm7s"
what is retrieval augmented generation in artificial intelligence
```

---

# handle_vague_query()

Purpose:

Improve very short or vague queries.

Code:

```python id="jlwm8m"
def handle_vague_query(query):
    vague_words = ["it", "this", "that", "thing", "stuff"]

    if len(query.split()) <= 2:
        query += " explanation"

    for word in vague_words:
        query = query.replace(word, "")

    return query.strip()
```

Example:

```python id="jlwm2t"
handle_vague_query("what is it")
```

Output:

```python id="jlwm6w"
what is explanation
```

Example:

```python id="jlwm1v"
handle_vague_query("connection issue")
```

Output:

```python id="jlwm4u"
connection issue explanation
```

---

# expand_query()

Purpose:

Use known context to make query more specific.

Code:

```python id="jlwm0g"
def expand_query(query, context=None):
    if context:
        return f"{query} in {context}"

    return query
```

Example:

```python id="jlwm6u"
expand_query("what is chunking", "retrieval augmented generation")
```

Output:

```python id="jlwm9r"
what is chunking in retrieval augmented generation
```

---

# generate_queries()

Purpose:

Generate multiple versions of same query.

Code:

```python id="jlwm3o"
def generate_queries(query):
    query = query.lower().strip()

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
```

Example:

```python id="jlwm9u"
generate_queries("what is rag")
```

Output:

```python id="jlwm7m"
[
    "what is rag",
    "what is retrieval augmented generation",
    "explain rag system",
    "rag architecture"
]
```

---

# process_query()

Purpose:

Apply all query transformation functions in correct order.

Code:

```python id="jlwm5v"
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
```

Example:

```python id="jlwm0u"
queries = process_query(
    "What is RAG in AI?",
    context="retrieval systems"
)

print(queries)
```

Output:

```python id="jlwm3v"
[
    "what is retrieval augmented generation in artificial intelligence in retrieval systems",
    "what is retrieval augmented generation in retrieval systems",
    "explain rag system in retrieval systems",
    "rag architecture in retrieval systems"
]
```

---

# Flow

```text id="jlwm8j"
raw query
→ normalize query
→ generate multiple queries
→ rewrite query
→ handle vague query
→ expand query with context
→ final query list
```

---

# Future Usage

Later this final query list can be used like:

```python id="jlwm1q"
queries = process_query(query)

all_results = []

for q in queries:
    results = hybrid_search(q)
    all_results.extend(results)
```

Then results can be:

- merged
- deduplicated
- reranked

# Dev Log — Week 5 Topic 3: Reranking Layer

# Goal

Improve ranking quality after retrieval.

Problem:

Dense retrieval and BM25 can retrieve useful chunks, but ordering is often weak.

Need:

1. Better ranking after retrieval
2. Better relevance scoring
3. Compare before vs after reranking
4. Limit reranking cost
5. Integrate reranking into pipeline

---

# Flow

```text id="jlwm4k"
query
→ retrieval
→ merged results
→ reranker
→ better ordered results
```

---

# Folder Structure

```text id="6jlwm7"
rag/
├── retriever/
│   ├── reranker.py
```

---

# Cross Encoder

Created:

```python id="9jlwm6"
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
```

Dense retrieval compares embeddings separately:

```text id="9jlwm4"
embedding(query)
vs
embedding(chunk)
```

Cross-encoder reads query and chunk together:

```text id="9jlwm1"
query + chunk
```

This gives more accurate relevance scoring.

---

# rerank_results()

Code:

```python id="7jlwm0"
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
```

---

# How It Works

Example:

```python id="7jlwm7"
query = "What is RAG"
```

Input chunks:

```python id="6jlwm8"
[
    {"text": "RAG combines retrieval and generation"},
    {"text": "Django is a backend framework"}
]
```

Cross-encoder gives scores:

```python id="4jlwm8"
[9.7, -10.6]
```

Meaning:

- First chunk is highly relevant
- Second chunk is irrelevant

---

# Improve Relevance Scoring

Before reranking, retrieval score already exists.

Dense:

```python id="2jlwm9"
for doc in dense_results:
    merged_results.append({
        "text": doc["text"],
        "source": doc["source"],
        "chunk_id": doc["chunk_id"],
        "retrieval_type": "dense",
        "retrieval_score": 0.7
    })
```

BM25:

```python id="1jlwm3"
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
```

Final score:

```python id="5jlwm3"
final_score = retrieval_score + rerank_score
```

---

# Before vs After Reranking

Before:

```python id="8jlwm1"
print("\n=== BEFORE RERANKING ===")

for item in merged_results:
    print(item["text"])
    print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
    print("-" * 50)
```

After:

```python id="9jlwm8"
print("\n=== AFTER RERANKING ===")

for item in reranked_results:
    print(item["text"])
    print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
    print(f"Rerank Score: {item.get('rerank_score', 0)}")
    print(f"Final Score: {item.get('final_score', 0)}")
    print("-" * 50)
```

Observation:

Before reranking, dense chunks had similar scores:

```text id="9jlwm9"
0.7
0.7
0.7
```

After reranking:

```text id="3jlwm7"
10.44
9.58
7.87
```

Irrelevant chunks dropped:

```text id="7jlwm5"
-10.45
-10.60
```

---

# Limit Top-N Before Reranking

Problem:

Reranking every chunk is expensive.

Need to rerank only best few chunks.

Code:

```python id="0jlwm5"
top_n = 5

merged_results = hybrid_search(query)

merged_results = merged_results[:top_n]

reranked_results = rerank_results(query, merged_results)
```

This reduces:

- latency
- compute cost
- unnecessary reranking

---

# Final Pipeline Integration

Updated `pipeline.py`:

```python id="8jlwm3"
from retriever.query_transformer import process_query
from retriever.hybrid_retriever import hybrid_search
from retriever.reranker import rerank_results
from prompt_builder import build_prompt
from generator import generate
```

Final pipeline:

```python id="1jlwm8"
def run_rag(query):
    queries = process_query(query)

    all_results = []

    for q in queries:
        results = hybrid_search(q)
        all_results.extend(results)

    unique_results = []

    for item in all_results:
        already_exists = any(
            existing["source"] == item["source"] and
            existing["chunk_id"] == item["chunk_id"]
            for existing in unique_results
        )

        if not already_exists:
            unique_results.append(item)

    if len(unique_results) == 0:
        return {
            "answer": "I don't know",
            "sources": []
        }

    top_results = unique_results[:5]

    reranked_results = rerank_results(query, top_results)

    final_context = reranked_results[:3]

    prompt = build_prompt(query, final_context)

    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": final_context
    }
```

---

# Main Test Code

```python id="2jlwm5"
from retriever.dense_retriever import setup_vector_db
from pipeline import run_rag

setup_vector_db()

query = "What is RAG"

result = run_rag(query)

print("\n=== ANSWER ===")
print(result["answer"])

print("\n=== SOURCES ===")

for item in result["sources"]:
    print(item["text"])
    print(f"Source: {item['source']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Retrieved By: {item['retrieval_type']}")
    print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
    print(f"Rerank Score: {item.get('rerank_score', 0)}")
    print(f"Final Score: {item.get('final_score', 0)}")
    print("-" * 50)
```

---

# Final Flow

```text id="6jlwm0"
query
→ query transformation
→ hybrid retrieval
→ merge results
→ remove duplicates
→ take top 5
→ rerank
→ take top 3
→ prompt
→ LLM
→ answer
```

---

# Key Points

- Dense retrieval gives initial candidates
- BM25 gives exact keyword matches
- Cross-encoder reranks retrieved chunks
- Final score combines retrieval score + rerank score
- Only top few chunks are reranked
- Final context becomes cleaner and more relevant

# Dev Log — Week 5 Topic 4: Context Optimization

# Goal

Improve quality of retrieved context before sending it to LLM.

Problem:

Retrieved chunks may be:

- duplicate
- repetitive
- too long
- too many
- low quality

Need to:

1. Remove redundant chunks
2. Compress long chunks
3. Limit total context size
4. Prioritize high quality chunks

---

# Folder Structure

```text id="0jlwm0"
rag/
├── retriever/
│   ├── context_optimizer.py
```

---

# remove_redundant_chunks()

Purpose:

Remove duplicate chunks.

Code:

```python id="4jlwm7"
def remove_redundant_chunks(results):
    unique_results = []
    seen_texts = set()

    for item in results:
        normalized_text = item["text"].lower().strip()

        if normalized_text not in seen_texts:
            unique_results.append(item)
            seen_texts.add(normalized_text)

    return unique_results
```

Example input:

```python id="7jlwm2"
[
    {"text": "RAG combines retrieval and generation"},
    {"text": "RAG combines retrieval and generation"},
    {"text": "FAISS is a vector search library"}
]
```

Output:

```python id="5jlwm4"
[
    {"text": "RAG combines retrieval and generation"},
    {"text": "FAISS is a vector search library"}
]
```

`seen_texts` is a set storing already seen chunk texts.

Set is used because lookup is very fast.

---

# compress_context()

Purpose:

Reduce chunk length before sending to LLM.

Code:

```python id="1jlwm9"
def compress_context(results, max_words=20):
    compressed_results = []

    for item in results:
        words = item["text"].split()

        compressed_text = " ".join(words[:max_words])

        new_item = item.copy()
        new_item["text"] = compressed_text

        compressed_results.append(new_item)

    return compressed_results
```

Example:

Original:

```text id="8jlwm9"
RAG is a technique that combines retrieval of relevant documents with text generation. It improves factual accuracy.
```

Compressed:

```text id="6jlwm5"
RAG is a technique that combines retrieval of relevant documents with text generation.
```

---

# limit_context_size()

Purpose:

Prevent prompt from becoming too large.

Code:

```python id="2jlwm7"
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
```

Example:

```python id="3jlwm4"
final_results = limit_context_size(results, max_total_words=80)
```

This stops adding chunks once total word count crosses limit.

---

# prioritize_chunks()

Purpose:

Sort chunks by quality.

Code:

```python id="9jlwm0"
def prioritize_chunks(results):
    prioritized_results = sorted(
        results,
        key=lambda x: x.get("final_score", 0),
        reverse=True
    )

    return prioritized_results
```

This function is optional because reranking already sorts by `final_score`.

Still useful for modularity.

---

# Suggested Optimization Flow

```python id="1jlwm7"
results = remove_redundant_chunks(results)

results = prioritize_chunks(results)

results = compress_context(results, max_words=20)

results = limit_context_size(results, max_total_words=100)
```

---

# Pipeline Integration

Example:

```python id="8jlwm7"
from retriever.context_optimizer import (
    remove_redundant_chunks,
    compress_context,
    limit_context_size,
    prioritize_chunks
)
```

Inside pipeline:

```python id="5jlwm7"
reranked_results = rerank_results(query, top_results)

optimized_results = remove_redundant_chunks(reranked_results)

optimized_results = prioritize_chunks(optimized_results)

optimized_results = compress_context(optimized_results, max_words=20)

optimized_results = limit_context_size(
    optimized_results,
    max_total_words=100
)

final_context = optimized_results[:3]
```

---

# Final Flow

```text id="7jlwm0"
query
→ retrieval
→ reranking
→ remove duplicates
→ prioritize best chunks
→ compress chunks
→ limit total context
→ final prompt context
```

---

# Key Points

- Duplicate chunks waste context space
- Long chunks increase token usage
- Too many chunks overflow prompt
- Best chunks should appear first
- Context optimization improves final answer quality

# Dev Log — Week 5 Topic 5: Retrieval Failure Handling

# Goal

Handle cases where retrieval fails.

Problem:

Retrieval may:

- return no chunks
- return weak chunks
- return irrelevant chunks
- cause unsafe answers

Need to:

1. Detect empty retrieval
2. Detect low-quality retrieval
3. Use fallback retrieval
4. Log failures
5. Return safe responses

---

# Folder Structure

```text id="gfj7d1"
rag/
├── retriever/
│   ├── retrieval_failure_handler.py
```

---

# is_empty_retrieval()

Purpose:

Detect when no chunks are retrieved.

Code:

```python id="zl0m71"
def is_empty_retrieval(results):
    return len(results) == 0
```

Example:

```python id="c2ofk3"
is_empty_retrieval([])
```

Output:

```python id="x2e7gh"
True
```

---

# is_low_quality_retrieval()

Purpose:

Detect when retrieved chunks are weak.

Code:

```python id="d08mrt"
def is_low_quality_retrieval(results, min_score=0.3):
    if len(results) == 0:
        return True

    top_score = results[0].get("score", 0)

    return top_score < min_score
```

Logic:

- If no results exist → low quality
- If top chunk score is below threshold → low quality

Example:

```python id="t7d0lz"
[
    {"score": 0.1},
    {"score": 0.2}
]
```

Output:

```python id="w3r9m1"
True
```

---

# log_retrieval_failure()

Purpose:

Print retrieval failures for debugging.

Code:

```python id="uy6y8s"
def log_retrieval_failure(query, reason):
    print("\n[RETRIEVAL FAILURE]")
    print(f"Query: {query}")
    print(f"Reason: {reason}")
```

Example output:

```text id="glz4aq"
[RETRIEVAL FAILURE]
Query: What is quantum gravity
Reason: empty retrieval
```

---

# build_safe_response()

Purpose:

Return safe answer when retrieval is unreliable.

Code:

```python id="n4j8k0"
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
```

---

# Fallback Retrieval

Purpose:

Try retrieval again when first retrieval is weak.

Flow:

```text id="ewvw5t"
normal retrieval
→ low quality
→ fallback retrieval
→ rerank again
→ safe response if still weak
```

Example:

```python id="ig7o0w"
fallback_results = fallback_retrieval(query)

fallback_top_results = fallback_results[:10]

reranked_results = rerank_results(
    query,
    fallback_top_results
)
```

Fallback can:

- increase top-k
- use BM25 more
- loosen thresholds
- retrieve more chunks

---

# Pipeline Integration

Updated imports:

```python id="txw2lj"
from retriever.query_transformer import process_query
from retriever.hybrid_retriever import fallback_retrieval, hybrid_search
from retriever.reranker import rerank_results
from prompt_builder import build_prompt
from generator import generate
from retriever.retrieval_failure_handler import (
    build_safe_response,
    is_empty_retrieval,
    is_low_quality_retrieval,
    log_retrieval_failure
)
```

Updated `run_rag()`:

```python id="sdj2ij"
def run_rag(query):
    queries = process_query(query)

    all_results = []

    for q in queries:
        results = hybrid_search(q)
        all_results.extend(results)

    unique_results = []

    for item in all_results:
        already_exists = any(
            existing["source"] == item["source"] and
            existing["chunk_id"] == item["chunk_id"]
            for existing in unique_results
        )

        if not already_exists:
            unique_results.append(item)

    if is_empty_retrieval(unique_results):
        log_retrieval_failure(
            query,
            reason="empty retrieval"
        )

        return build_safe_response("empty retrieval")

    top_results = unique_results[:5]

    reranked_results = rerank_results(query, top_results)

    if is_low_quality_retrieval(reranked_results):
        fallback_results = fallback_retrieval(query)

        fallback_top_results = fallback_results[:10]

        reranked_results = rerank_results(
            query,
            fallback_top_results
        )

        if is_low_quality_retrieval(reranked_results):
            log_retrieval_failure(
                query,
                reason="low quality retrieval"
            )

            return build_safe_response("low quality retrieval")

    final_context = reranked_results[:3]

    prompt = build_prompt(query, final_context)

    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": final_context
    }
```

---

# Final Flow

```text id="y9e1sn"
query
→ process query
→ hybrid retrieval
→ remove duplicates
→ check empty retrieval
→ rerank
→ check low quality retrieval
→ fallback retrieval if needed
→ rerank again
→ safe response if still weak
→ build prompt
→ generate answer
```

---

# Key Points

- Empty retrieval should not go to LLM
- Weak retrieval should trigger fallback
- Retrieval failures should be logged
- Unsafe retrieval should return safe response
- Fallback retrieval improves robustness

# Dev Log — Week 5 Topic 6: Parameter Tuning

# Goal

Tune retrieval parameters to improve:

- retrieval quality
- reranking quality
- recall
- precision
- latency

Need to:

1. Tune top-k values
2. Tune low-quality threshold
3. Measure precision and recall
4. Prevent over retrieval and under retrieval
5. Measure pipeline timing

---

# Folder Structure

```text
rag/
├── config.py
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── test_retrieval_metrics.py
├── retriever/
│   ├── parameter_validator.py
├── utils/
│   ├── timer.py
```

---

# config.py

Purpose:

Store all retrieval tuning values in one place.

Code:

```python
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
FINAL_CONTEXT_K = 3

LOW_QUALITY_SCORE_THRESHOLD = 0.35
```

Meaning:

- `RETRIEVAL_TOP_K` = number of retrieved chunks kept before reranking
- `RERANK_TOP_K` = number of chunks sent to reranker
- `FINAL_CONTEXT_K` = final chunks used in prompt
- `LOW_QUALITY_SCORE_THRESHOLD` = minimum rerank score needed

Recommended values:

```python
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
FINAL_CONTEXT_K = 3
LOW_QUALITY_SCORE_THRESHOLD = 0.35
```

---

# is_low_quality_retrieval()

Purpose:

Detect weak reranked results.

Code:

```python
from config import LOW_QUALITY_SCORE_THRESHOLD

def is_low_quality_retrieval(results):
    if not results:
        return True

    best_score = results[0]["rerank_score"]

    return best_score < LOW_QUALITY_SCORE_THRESHOLD
```

This uses rerank score threshold instead of only checking empty list.

---

# calculate_precision()

Purpose:

Measure how many retrieved chunks are relevant.

Code:

```python
def calculate_precision(retrieved_chunks, relevant_chunks):
    if len(retrieved_chunks) == 0:
        return 0

    relevant_retrieved = 0

    for chunk in retrieved_chunks:
        if chunk["chunk_id"] in relevant_chunks:
            relevant_retrieved += 1

    precision = relevant_retrieved / len(retrieved_chunks)

    return round(precision, 2)
```

Formula:

```text
precision =
relevant retrieved chunks
÷
total retrieved chunks
```

Example:

```python
retrieved_chunks = [
    {"chunk_id": 1},
    {"chunk_id": 2},
    {"chunk_id": 3},
    {"chunk_id": 4},
    {"chunk_id": 5}
]

relevant_chunks = [2, 3, 4, 8]
```

Precision:

```text
3 / 5 = 0.6
```

---

# calculate_recall()

Purpose:

Measure how many useful chunks were successfully retrieved.

Code:

```python
def calculate_recall(retrieved_chunks, relevant_chunks):
    if len(relevant_chunks) == 0:
        return 0

    relevant_retrieved = 0

    for chunk in retrieved_chunks:
        if chunk["chunk_id"] in relevant_chunks:
            relevant_retrieved += 1

    recall = relevant_retrieved / len(relevant_chunks)

    return round(recall, 2)
```

Formula:

```text
recall =
relevant retrieved chunks
÷
all possible relevant chunks
```

Example:

```text
3 / 4 = 0.75
```

---

# test_retrieval_metrics.py

Purpose:

Test precision and recall calculations.

Code:

```python
from evaluation.retrieval_metrics import (
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
```

Expected output:

```text
Precision: 0.6
Recall: 0.75
```

---

# validate_retrieval_parameters()

Purpose:

Prevent invalid top-k values.

Code:

```python
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
```

Bad example:

```python
RETRIEVAL_TOP_K = 3
RERANK_TOP_K = 10
FINAL_CONTEXT_K = 5
```

This is invalid because reranker cannot rerank more chunks than retrieval returns.

---

# measure_time()

Purpose:

Measure execution time of pipeline stages.

Code:

```python
import time

def measure_time(function, *args, **kwargs):
    start_time = time.time()

    result = function(*args, **kwargs)

    end_time = time.time()

    execution_time = round(end_time - start_time, 3)

    return result, execution_time
```

This can be used for:

- retrieval time
- rerank time
- fallback time
- generation time
- total pipeline time

---

# Suggested Parameter Values

Balanced setup:

```python
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
FINAL_CONTEXT_K = 3
```

Fast setup:

```python
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3
FINAL_CONTEXT_K = 2
```

High accuracy setup:

```python
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 10
FINAL_CONTEXT_K = 5
```

---

# Pipeline Integration

Imports:

```python
from config import (
    RETRIEVAL_TOP_K,
    FINAL_CONTEXT_K
)

from retriever.parameter_validator import (
    validate_retrieval_parameters
)

from utils.timer import measure_time
```

Inside pipeline:

```python
def run_rag(query):
    validate_retrieval_parameters()

    total_start = time.time()

    queries = process_query(query)

    all_results = []

    for q in queries:
        results, retrieval_time = measure_time(
            hybrid_search,
            q
        )

        print(f"Retrieval Time for '{q}': {retrieval_time}s")

        all_results.extend(results)

    top_results = unique_results[:RETRIEVAL_TOP_K]

    reranked_results, rerank_time = measure_time(
        rerank_results,
        query,
        top_results
    )

    print(f"Rerank Time: {rerank_time}s")

    final_context = reranked_results[:FINAL_CONTEXT_K]

    answer, generation_time = measure_time(
        generate,
        prompt
    )

    print(f"Generation Time: {generation_time}s")

    total_end = time.time()

    total_time = round(total_end - total_start, 3)

    print(f"Total Pipeline Time: {total_time}s")
```

---

# Final Flow

```text
query
→ retrieval
→ top-k filtering
→ reranking
→ threshold check
→ precision / recall evaluation
→ context selection
→ prompt building
→ answer generation
→ timing measurement
```

---

# Key Points

- Low top-k hurts recall
- High top-k hurts precision
- Rerank top-k should always be lower than retrieval top-k
- Final context size should always be smallest
- Precision checks noise
- Recall checks missing chunks
- Timing helps balance speed and answer quality

# Dev Log — Week 5 Topic 7: Advanced Patterns

# Goal

Improve retrieval quality for more complex queries.

Need to:

1. Support multi-hop retrieval
2. Support parent-child chunking
3. Filter by metadata
4. Support time-aware retrieval
5. Tune retrieval based on domain

---

# Folder Structure

```text
rag/
├── retriever/
│   ├── multi_hop_retriever.py
│   ├── parent_child_chunking.py
│   ├── metadata_filter.py
│   ├── domain_config.py
```

---

# split_multi_hop_query()

Purpose:

Break complex query into multiple smaller queries.

Code:

```python
def split_multi_hop_query(query):
    separators = [" and ", ",", " then ", " also "]

    sub_queries = [query]

    for separator in separators:
        new_sub_queries = []

        for item in sub_queries:
            parts = item.split(separator)

            for part in parts:
                cleaned_part = part.strip()

                if cleaned_part:
                    new_sub_queries.append(cleaned_part)

        sub_queries = new_sub_queries

    return sub_queries
```

Example:

```python
query = "Who created FAISS and what is it used for?"
```

Output:

```python
[
    "Who created FAISS",
    "what is it used for?"
]
```

This improves retrieval for complex questions.

---

# create_parent_child_chunks()

Purpose:

Split large parent chunks into smaller child chunks.

Code:

```python
def create_parent_child_chunks(documents, child_size=20):
    parent_child_pairs = []

    for parent_id, document in enumerate(documents):
        words = document.split()

        child_chunks = []

        for i in range(0, len(words), child_size):
            child_text = " ".join(words[i:i + child_size])

            child_chunks.append({
                "parent_id": parent_id,
                "child_text": child_text
            })

        parent_child_pairs.append({
            "parent_id": parent_id,
            "parent_text": document,
            "children": child_chunks
        })

    return parent_child_pairs
```

Example:

```python
documents = [
    "FAISS is a vector database library created by Meta. It is used for semantic search and similarity search."
]
```

This improves retrieval precision while still keeping full context.

---

# filter_by_metadata()

Purpose:

Remove chunks that do not match required metadata.

Code:

```python
def filter_by_metadata(results, required_topic=None, required_source=None):
    filtered_results = []

    for item in results:
        topic_match = True
        source_match = True

        if required_topic:
            topic_match = item.get("topic") == required_topic

        if required_source:
            source_match = item.get("source") == required_source

        if topic_match and source_match:
            filtered_results.append(item)

    return filtered_results
```

Example:

```python
results = filter_by_metadata(
    results,
    required_topic="python"
)
```

This reduces noisy chunks before reranking.

---

# DOMAIN_CONFIGS

Purpose:

Store different retrieval settings for different domains.

Code:

```python
DOMAIN_CONFIGS = {
    "code": {
        "retrieval_top_k": 5,
        "rerank_top_k": 3,
        "final_context_k": 2,
        "chunk_size": 100
    },
    "legal": {
        "retrieval_top_k": 15,
        "rerank_top_k": 8,
        "final_context_k": 5,
        "chunk_size": 300
    },
    "research": {
        "retrieval_top_k": 20,
        "rerank_top_k": 10,
        "final_context_k": 5,
        "chunk_size": 400
    }
}
```

Different domains need different chunk sizes and top-k values.

---

# get_domain_config()

Purpose:

Load retrieval config for selected domain.

Code:

```python
from retriever.domain_config import DOMAIN_CONFIGS

def get_domain_config(domain):
    return DOMAIN_CONFIGS.get(
        domain,
        DOMAIN_CONFIGS["code"]
    )
```

Example:

```python
config = get_domain_config("legal")
```

Output:

```python
{
    "retrieval_top_k": 15,
    "rerank_top_k": 8,
    "final_context_k": 5,
    "chunk_size": 300
}
```

---

# Pipeline Integration

Imports:

```python
from retriever.multi_hop_retriever import split_multi_hop_query
from retriever.metadata_filter import filter_by_metadata
from retriever.domain_config import DOMAIN_CONFIGS
```

Helper function:

```python
def get_domain_config(domain):
    return DOMAIN_CONFIGS.get(
        domain,
        DOMAIN_CONFIGS["code"]
    )
```

Inside pipeline:

```python
def run_rag(
    query,
    domain="code",
    required_topic=None,
    required_source=None
):
    validate_retrieval_parameters()

    domain_config = get_domain_config(domain)

    retrieval_top_k = domain_config["retrieval_top_k"]
    final_context_k = domain_config["final_context_k"]

    transformed_queries = process_query(query)

    multi_hop_queries = []

    for transformed_query in transformed_queries:
        split_queries = split_multi_hop_query(
            transformed_query
        )

        multi_hop_queries.extend(split_queries)

    all_results = []

    for q in multi_hop_queries:
        results, retrieval_time = measure_time(
            hybrid_search,
            q
        )

        print(f"Retrieval Time for '{q}': {retrieval_time}s")

        all_results.extend(results)

    unique_results = []

    for item in all_results:
        already_exists = any(
            existing["source"] == item["source"] and
            existing["chunk_id"] == item["chunk_id"]
            for existing in unique_results
        )

        if not already_exists:
            unique_results.append(item)

    filtered_results = filter_by_metadata(
        unique_results,
        required_topic=required_topic,
        required_source=required_source
    )

    top_results = filtered_results[:retrieval_top_k]

    reranked_results, rerank_time = measure_time(
        rerank_results,
        query,
        top_results
    )

    final_context = reranked_results[:final_context_k]
```

---

# Final Flow

```text
query
→ process_query()
→ split multi-hop query
→ retrieval for each sub-query
→ merge results
→ remove duplicates
→ metadata filtering
→ domain-based top-k
→ reranking
→ final context
→ answer
```

---

# Key Points

- Multi-hop retrieval improves complex question handling
- Parent-child chunking improves precision and context
- Metadata filtering reduces noisy chunks
- Different domains need different retrieval settings
- Domain tuning improves overall retrieval quality

# Dev Log — Week 5 Topic 8: Evaluation and Benchmarking

# Goal

Measure retrieval quality and answer quality.

Need to test:

- whether correct source was retrieved
- whether answer is correct
- whether different configurations improve performance

---

# Folder Structure

```text id="xjlwm1"
rag/
├── evaluation/
│   ├── __init__.py
│   ├── test_data.py
│   └── evaluator.py
```

---

# test_data.py

Purpose:

Store evaluation dataset.

Code:

```python id="7jlwm8"
test_queries = [
    {
        "query": "What is RAG?",
        "expected_answer": "RAG is a system that combines retrieval with generation.",
        "expected_source": "rag.pdf"
    },
    {
        "query": "What is FAISS?",
        "expected_answer": "FAISS is a library for similarity search.",
        "expected_source": "sample.txt"
    },
    {
        "query": "What is Django?",
        "expected_answer": "Django is a backend framework used for web development.",
        "expected_source": "backend.txt"
    },
    {
        "query": "What is quantum gravity?",
        "expected_answer": "I don't know",
        "expected_source": None
    }
]
```

Need:

- expected answer
- expected source

---

# evaluate_retrieval()

Purpose:

Check whether correct document source was retrieved.

Code:

```python id="0jlwm4"
from evaluation.test_data import test_queries
from pipeline import run_rag

def evaluate_retrieval():
    for item in test_queries:
        result = run_rag(item["query"])

        retrieved_sources = [
            source["source"]
            for source in result["sources"]
        ]

        correct_source_found = (
            item["expected_source"] in retrieved_sources
        )

        print("\nQuery:", item["query"])
        print("Expected Source:", item["expected_source"])
        print("Retrieved Sources:", retrieved_sources)
        print("Correct Source Found:", correct_source_found)
        print("-" * 50)
```

---

# compare_configurations()

Purpose:

Compare different retrieval settings.

Code:

```python id="5jlwm8"
def compare_configurations():
    ks = [3, 5, 7]

    for k in ks:
        print(f"\n=== Testing k={k} ===")

        for item in test_queries:
            result = run_rag(item["query"], k=k)

            retrieved_sources = [
                source["source"]
                for source in result["sources"]
            ]

            correct_source_found = (
                item["expected_source"] in retrieved_sources
            )

            print("Query:", item["query"])
            print("Correct Source Found:", correct_source_found)
            print("Retrieved Sources:", retrieved_sources)
            print("-" * 50)
```

Need update in `pipeline.py`:

```python id="4jlwm4"
def run_rag(
    query,
    k=5,
    domain="code",
    required_topic=None,
    required_source=None
):
```

Need update in `hybrid_retriever.py`:

```python id="6jlwm1"
def hybrid_search(query, k=5):
    dense_results = retrieve(query, k=k)
    bm25_results = bm25_search(query, top_k=k)
```

Observation:

- k=3 is fast but may miss context
- k=7 increases rerank time
- k=5 gives best balance

---

# evaluate_answers()

Purpose:

Check answer quality.

Initial version was too strict:

```python id="3jlwm1"
is_correct = expected in actual
```

Problem:

Correct answers with different wording failed.

Example:

```text id="9jlwm6"
Expected:
RAG is a system that combines retrieval with generation.

Actual:
RAG is a system that combines retrieval (search) with generation (LLM).
```

Need softer evaluation.

Updated code:

```python id="2jlwm3"
def evaluate_answers():
    total_queries = len(test_queries)
    correct_answers = 0

    for item in test_queries:
        result = run_rag(item["query"])

        expected = item["expected_answer"].lower()
        actual = result["answer"].lower()

        expected_words = set(expected.split())
        actual_words = set(actual.split())

        overlap = expected_words.intersection(actual_words)

        score = len(overlap) / len(expected_words)

        if expected == "i don't know":
            is_correct = (
                "not reliable enough" in actual or
                "don't know" in actual
            )
        else:
            is_correct = score >= 0.5

        if is_correct:
            correct_answers += 1

        print("\nQuery:", item["query"])
        print("Expected:", item["expected_answer"])
        print("Actual:", result["answer"])
        print("Match Score:", round(score, 2))
        print("Correct:", is_correct)
        print("-" * 50)

    accuracy = correct_answers / total_queries

    print("\nFinal Accuracy:", round(accuracy, 2))
```

---

# Running Evaluation

In `main.py`:

```python id="1jlwm2"
from retriever.dense_retriever import setup_vector_db
from evaluation.evaluator import (
    evaluate_retrieval,
    compare_configurations,
    evaluate_answers
)

setup_vector_db()

evaluate_retrieval()
compare_configurations()
evaluate_answers()
```

Run:

```bash id="8jlwm4"
python main.py
```

---

# Final Flow

```text id="5jlwm5"
test queries
→ run pipeline
→ check retrieved sources
→ compare k values
→ compare answers
→ calculate accuracy
```

---

# Key Points

- Need fixed test dataset
- Retrieval evaluation checks sources
- Answer evaluation checks wording overlap
- Different k values affect quality and speed
- k=5 gave best balance
- Safe fallback responses should count as correct for unknown questions

# pipeline.py

```python
import time

from retriever.query_transformer import process_query
from retriever.hybrid_retriever import fallback_retrieval, hybrid_search
from retriever.reranker import rerank_results
from retriever.multi_hop_retriever import split_multi_hop_query
from retriever.metadata_filter import filter_by_metadata
from retriever.parameter_validator import validate_retrieval_parameters
from retriever.domain_config import DOMAIN_CONFIGS
from retriever.retrieval_failure_handler import (
    build_safe_response,
    is_empty_retrieval,
    is_low_quality_retrieval,
    log_retrieval_failure
)

from prompt_builder import build_prompt
from generator import generate
from utils.timer import measure_time


def get_domain_config(domain):
    return DOMAIN_CONFIGS.get(
        domain,
        DOMAIN_CONFIGS["code"]
    )


def run_rag(
    query,
    k=5,
    domain="code",
    required_topic=None,
    required_source=None
):
    validate_retrieval_parameters()

    total_start = time.time()

    domain_config = get_domain_config(domain)

    retrieval_top_k = k
    final_context_k = domain_config["final_context_k"]

    transformed_queries = process_query(query)

    multi_hop_queries = []

    for transformed_query in transformed_queries:
        split_queries = split_multi_hop_query(
            transformed_query
        )

        multi_hop_queries.extend(split_queries)

    all_results = []

    for q in multi_hop_queries:
        results, retrieval_time = measure_time(
            hybrid_search,
            q,
            k
        )

        print(f"Retrieval Time for '{q}': {retrieval_time}s")

        all_results.extend(results)

    unique_results = []

    for item in all_results:
        already_exists = any(
            existing["source"] == item["source"] and
            existing["chunk_id"] == item["chunk_id"]
            for existing in unique_results
        )

        if not already_exists:
            unique_results.append(item)

    filtered_results = filter_by_metadata(
        unique_results,
        required_topic=required_topic,
        required_source=required_source
    )

    if is_empty_retrieval(filtered_results):
        log_retrieval_failure(
            query,
            reason="empty retrieval"
        )

        return build_safe_response("empty retrieval")

    top_results = filtered_results[:retrieval_top_k]

    reranked_results, rerank_time = measure_time(
        rerank_results,
        query,
        top_results
    )

    print(f"Rerank Time: {rerank_time}s")

    if is_low_quality_retrieval(reranked_results):
        fallback_results, fallback_time = measure_time(
            fallback_retrieval,
            query
        )

        print(f"Fallback Retrieval Time: {fallback_time}s")

        fallback_results = filter_by_metadata(
            fallback_results,
            required_topic=required_topic,
            required_source=required_source
        )

        fallback_top_results = fallback_results[:retrieval_top_k]

        reranked_results, rerank_time = measure_time(
            rerank_results,
            query,
            fallback_top_results
        )

        print(f"Fallback Rerank Time: {rerank_time}s")

        if is_low_quality_retrieval(reranked_results):
            log_retrieval_failure(
                query,
                reason="low quality retrieval"
            )

            return build_safe_response("low quality retrieval")

    final_context = reranked_results[:final_context_k]

    prompt = build_prompt(query, final_context)

    answer, generation_time = measure_time(
        generate,
        prompt
    )

    print(f"Generation Time: {generation_time}s")

    total_end = time.time()

    total_time = round(total_end - total_start, 3)

    print(f"Total Pipeline Time: {total_time}s")

    return {
        "answer": answer,
        "sources": final_context
    }
```

# main.py

```python
from retriever.dense_retriever import setup_vector_db
from pipeline import run_rag
from evaluation.evaluator import (
    evaluate_retrieval,
    compare_configurations,
    evaluate_answers
)

setup_vector_db()

query = "What is RAG"

result = run_rag(query)

print("\n=== ANSWER ===")
print(result["answer"])

print("\n=== SOURCES ===")

for item in result["sources"]:
    print(item["text"])
    print(f"Source: {item['source']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Retrieved By: {item['retrieval_type']}")
    print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
    print(f"Rerank Score: {item.get('rerank_score', 0)}")
    print(f"Final Score: {item.get('final_score', 0)}")
    print("-" * 50)

print("\n=== RETRIEVAL EVALUATION ===")
evaluate_retrieval()

print("\n=== CONFIGURATION COMPARISON ===")
compare_configurations()

print("\n=== ANSWER EVALUATION ===")
evaluate_answers()
```

# Final Output

```text
=== ANSWER ===
Retrieval Augmented Generation (RAG) is a system that combines retrieval (search) with generation (LLM).

=== SOURCES ===
RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation. FAISS is a library developed by Facebook for efficient similarity search.
Source: sample.txt
Chunk ID: 0
Retrieved By: dense
Retrieval Score: 0.7
Rerank Score: 9.745859146118164
Final Score: 10.445859146118163
--------------------------------------------------
Retrieval Augmented Generation (RAG) - Simple Guide What is RAG? RAG is a system that combines retrieval (search) with generation (LLM).
Source: rag.pdf
Chunk ID: 0
Retrieved By: dense
Retrieval Score: 0.7
Rerank Score: 8.888425827026367
Final Score: 9.588425827026366
--------------------------------------------------

Final Accuracy: 0.75
```

:::
