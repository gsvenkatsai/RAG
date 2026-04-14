from pipeline import run_rag

# print(run_rag("What is machine learning?"))
# print(run_rag("What is Django?"))
# print(run_rag("What is RAG?"))
# # print(run_rag("What is quantum gravity?"))
# test_queries = [
#     {"query": "What is RAG?", "expected": "RAG is a system that combines retrieval with generation."},
#     {"query": "What is FAISS?", "expected": "FAISS is a library for similarity search."},
#     {"query": "What is quantum gravity?", "expected": "I don't know"}
# ]
# def evaluate():
#     for item in test_queries:
#         result = run_rag(item["query"])
#         print("Q:", item["query"])
#         print("Expected:", item["expected"])
#         print("Got:", result["answer"])
#         print("-" * 40)
# evaluate()

from fastapi import FastAPI
from pipeline import run_rag, init

app = FastAPI()

init()  # initialize once

@app.get("/")
def home():
    return {"message": "RAG API running"}

@app.get("/ask")
def ask(query: str):
    return run_rag(query)