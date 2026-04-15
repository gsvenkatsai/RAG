# 📘 Dev Log — RAG System (Subtopic 1 & 2)

---

# 🔹 Project: Retrieval-Augmented Generation (RAG)

## Phase: Core Pipeline + Data Ingestion

---

# 🚀 Subtopic 1: End-to-End RAG System Assembly

## 🎯 Goal

Build a complete pipeline:

```
User Query → Retrieval → Prompt → LLM → Answer
```

---

## 🧠 Step 1: Defined Architecture

We structured system into modular components:

```
rag/
 ├── retriever.py
 ├── prompt_builder.py
 ├── generator.py
 ├── pipeline.py
```

---

## 🧱 Step 2: Core Pipeline

### `pipeline.py`

```python
from retriever import retrieve, setup_vector_db
from prompt_builder import build_prompt
from generator import generate

setup_vector_db()

def run_rag(query):
    context = retrieve(query)

    if not context:
        return {
            "answer": "I don't know. No relevant information found.",
            "sources": []
        }

    prompt = build_prompt(query, context)
    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": context
    }
```

---

## 🧱 Step 3: Retriever (FAISS + Embeddings)

### Key logic:

```python
def retrieve(query):
    query_embedding = model.encode([query])
    results = vector_db_search(query_embedding)
    return results
```

---

## 🧱 Step 4: Prompt Engineering

```python
def build_prompt(query, context_chunks):
    context_text = "\n".join([chunk["text"] for chunk in context_chunks])

    prompt = f"""
You are a strict assistant.

Rules:
- Answer ONLY from the provided context
- Do NOT repeat the question
- Give a complete sentence answer
- If answer is not clearly in context, say "I don't know"

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt
```

---

## 🧱 Step 5: Generator (LLM Swap to Groq)

```python
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate(prompt):
    return call_llm(prompt)
```

---

## 🧠 Key Learnings (Subtopic 1)

- RAG = system, not just LLM
- Modular design is critical
- LLM should be replaceable (Gemini → Groq)
- Guardrails prevent hallucination
- Retrieval quality defines system quality

---

# 🚀 Subtopic 2: Data Ingestion Pipeline

---

## 🎯 Goal

Build pipeline:

```
Files → Clean → Chunk → Embed → Store → Retrieve
```

---

## 🧱 Step 1: Load Data (Multi-file)

```python
def load_all_texts(folder_path):
    all_data = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text = ""

        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        text = clean_text(text)

        all_data.append({
            "text": text,
            "source": file
        })

    return all_data
```

---

## 🧱 Step 2: Text Cleaning

```python
def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

## 🧱 Step 3: Chunking (Sentence-aware)

```python
def chunk_text(text, chunk_size=2):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        current_chunk.append(sentence)

        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

---

## 🧱 Step 4: Metadata Integration

```python
documents.append({
    "text": chunk,
    "source": file["source"],
    "chunk_id": i
})
```

---

## 🧱 Step 5: Vector DB Setup

```python
embeddings = model.encode([doc["text"] for doc in documents])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
```

---

## 🧱 Step 6: Retrieval Optimization

```python
def vector_db_search(query_embedding, k=1, threshold=2.0):
    distances, indices = index.search(query_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < threshold:
            results.append(documents[idx])

    return results
```

---

## 🧠 Key Learnings (Subtopic 2)

- Chunking quality > model quality
- PDF ingestion is messy but essential
- Metadata enables traceability
- Cleaning improves embeddings significantly
- Retrieval tuning (k + threshold) is critical

---

# 🏁 Final System Capabilities

✔ Multi-file ingestion (.txt + .pdf)
✔ Clean text preprocessing
✔ Sentence-aware chunking
✔ Metadata tracking
✔ FAISS vector search
✔ Tuned retrieval (k=1)
✔ LLM integration (Groq)
✔ Guardrails + structured output

---

# 🔥 Final Insight

> RAG is not about calling an LLM
> It is about controlling **what the LLM sees**

---

# 📊 Status

| Component   | Status  |
| ----------- | ------- |
| Pipeline    | ✅ Done |
| Retrieval   | ✅ Done |
| Prompting   | ✅ Done |
| Ingestion   | ✅ Done |
| PDF Support | ✅ Done |
| Cleaning    | ✅ Done |

---

# Dev Log — Subtopic 3 (Retrieval Optimization)

## Changes

### 1. Increased K

```python
k = 5
```

More chunks retrieved → better context coverage.

---

### 2. Added Distance Threshold

```python
if dist <= 1.5:
```

Filters out weak matches (FAISS L2 → lower is better).

---

### 3. Empty Retrieval Handling

```python
if len(results) == 0:
    return None
```

Prevents passing irrelevant context to LLM.

---

### 4. Fixed Return Structure

```python
docs = vector_db_search(query_embedding)
return docs
```

Ensures only documents (not tuple) are passed to prompt builder.

---

## Final Code

### `retriever.py`

```python
def vector_db_search(query_embedding, k=5, threshold=1.5):
    distances, indices = index.search(query_embedding, k)

    results = []

    for i, idx in enumerate(indices[0]):
        dist = distances[0][i]

        if dist <= threshold:
            results.append(documents[idx])

    if len(results) == 0:
        return None

    return results


def retrieve(query):
    query_embedding = model.encode([query])

    docs = vector_db_search(query_embedding)

    if docs is None:
        return None

    return docs
```

# 📘 Dev Log — RAG System (Subtopics 4–6)

---

# 🔹 Subtopic 4: Prompt Engineering (Production-Level)

## Changes

### 1. Structured Prompt + Strict Instructions

```python
def build_prompt(query, context_chunks):
    context_text = "\n".join([
        f"[Source: {chunk['source']}] {chunk['text']}"
        for chunk in context_chunks
    ])

    prompt = f"""
You are a factual assistant.

Instructions:
- Answer ONLY from the provided context
- If answer is not in context, respond EXACTLY with: I don't know
- Do NOT make up information
- Do NOT infer beyond context
- Output only the final answer

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt
```

**Why:** forces grounding, deterministic refusal, clean output.

---

# 🔹 Subtopic 5: Response Validation & Guardrails

## Changes

### 1. Post-response Validation

```python
def validate_response(answer, context_chunks):
    context_text = " ".join([chunk["text"] for chunk in context_chunks])

    if answer.strip() == "I don't know":
        return answer

    if len(answer.strip()) < 5:
        return "I don't know"

    if answer.lower() not in context_text.lower():
        return "I don't know"

    return answer
```

**Why:** rejects hallucinated / weak answers.

---

### 2. Confidence Signal

```python
confidence = len(context)
```

**Why:** simple proxy → more chunks = more support (not perfect).

---

### 3. Refusal Strategy (3 layers)

- Retrieval:

```python
if not context:
    return "I don't know"
```

- Prompt:

```text
respond EXACTLY with: I don't know
```

- Validation:

```python
invalid → "I don't know"
```

**Why:** ensures system refuses instead of hallucinating.

---

# 🔹 Subtopic 6: Evaluation Framework

## Changes

### 1. Test Dataset

```python
test_queries = [
    {"query": "What is RAG?", "expected": "RAG is a system that combines retrieval with generation."},
    {"query": "What is FAISS?", "expected": "FAISS is a library for similarity search."},
    {"query": "What is quantum gravity?", "expected": "I don't know"}
]
```

---

### 2. Evaluation Loop

```python
def evaluate():
    for item in test_queries:
        result = run_rag(item["query"])

        is_correct = item["expected"].lower() in result["answer"].lower()

        print("Q:", item["query"])
        print("Expected:", item["expected"])
        print("Got:", result["answer"])
        print("Correct:", is_correct)
        print("-" * 40)
```

---

### 3. Retrieval Latency

```python
import time

start = time.time()
context = retrieve(query)
retrieval_time = time.time() - start
```

---

### 4. Compare Configurations (manual)

```python
# try different values
k = 3, 5, 7
threshold = 1.3, 1.5, 1.7
```

---

### 5. Manual + Automated Evaluation

- Automated:

```python
is_correct = expected in answer
```

- Manual:
  → read answers, check correctness + grounding

---

## Final Pipeline

```python
def run_rag(query):
    import time

    start = time.time()
    context = retrieve(query)
    retrieval_time = time.time() - start

    if not context:
        return {
            "answer": "I don't know",
            "sources": [],
            "confidence": 0,
            "latency": retrieval_time
        }

    prompt = build_prompt(query, context)
    answer = generate(prompt)
    answer = validate_response(answer, context)

    return {
        "answer": answer,
        "sources": context,
        "confidence": len(context),
        "latency": retrieval_time
    }
```

---

# 🏁 Summary

- Prompt → controls generation
- Guardrails → enforce correctness
- Evaluation → measures performance

---

# 📘 Dev Log — RAG System (Subtopics 7–9)

---

# 🔹 Subtopic 7: Backend API Development

## Changes

### 1. FastAPI Integration

```python
from fastapi import FastAPI
from pipeline import run_rag, init

app = FastAPI()

init()

@app.get("/ask")
def ask(query: str):
    return run_rag(query)
```

**Why:** exposes RAG pipeline as an API service.

---

### 2. Response Optimization

```python
sources = [
    {"source": c["source"], "chunk_id": c["chunk_id"]}
    for c in context
]
```

```python
results = results[:3]
```

**Why:** reduces noise, improves latency, cleaner output.

---

### 3. Latency Tracking

```python
start = time.time()
context = retrieve(query)
retrieval_time = time.time() - start
```

**Why:** measure performance per request.

---

### 4. Error Handling

```python
if not context:
    return {
        "answer": "I don't know",
        "sources": [],
        "confidence": 0,
        "latency": retrieval_time
    }
```

**Why:** prevents crashes and enforces consistent responses.

---

# 🔹 Subtopic 8: Simple Frontend

## Changes

### 1. Streamlit UI

```python
import streamlit as st
import requests

st.title("RAG Q&A System")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        response = requests.get(
            "http://127.0.0.1:8000/ask",
            params={"query": query}
        )

        data = response.json()

        st.write("### Answer")
        st.write(data["answer"])

        st.write("### Confidence")
        st.write(data["confidence"])

        st.write("### Sources")
        for s in data["sources"]:
            st.write(f"{s['source']} (chunk {s['chunk_id']})")
```

**Why:** quick UI for interacting with API without frontend complexity.

---

### 2. Frontend–Backend Separation

- Frontend → handles UI
- Backend → handles logic/API

**Why:** modular system, easier deployment and scaling.

---

# 🔹 Subtopic 9: Deployment

## Changes

### 1. Requirements Cleanup

```text
fastapi
uvicorn
sentence-transformers
faiss-cpu
numpy
python-dotenv
groq
```

**Why:** removed dependency conflicts, allowed successful build.

---

### 2. Backend Deployment (Render)

- Build:

```bash
pip install -r requirements.txt
```

- Start:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Why:** makes API accessible over internet.

---

### 3. Environment Variables

```text
GROQ_API_KEY=...
```

**Why:** secure handling of secrets.

---

### 4. Frontend Deployment (optional)

- Streamlit service:

```bash
streamlit run app.py --server.port 10000 --server.address 0.0.0.0
```

- API URL updated to deployed backend

**Why:** connect UI to live backend.

---

## 🏁 Summary

- Backend → API layer over RAG
- Frontend → simple user interface
- Deployment → public access system

---
