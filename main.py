# from pipeline import run_rag

# # print(run_rag("What is machine learning?"))
# # print(run_rag("What is Django?"))
# # print(run_rag("What is RAG?"))
# # # print(run_rag("What is quantum gravity?"))
# # test_queries = [
# #     {"query": "What is RAG?", "expected": "RAG is a system that combines retrieval with generation."},
# #     {"query": "What is FAISS?", "expected": "FAISS is a library for similarity search."},
# #     {"query": "What is quantum gravity?", "expected": "I don't know"}
# # ]
# # def evaluate():
# #     for item in test_queries:
# #         result = run_rag(item["query"])
# #         print("Q:", item["query"])
# #         print("Expected:", item["expected"])
# #         print("Got:", result["answer"])
# #         print("-" * 40)
# # evaluate()

# from fastapi import FastAPI
# from pipeline import run_rag, init

# app = FastAPI()

# init()  # initialize once

# @app.get("/")
# def home():
#     return {"message": "RAG API running"}

# @app.get("/ask")
# def ask(query: str):
#     return run_rag(query)

# from retriever.hybrid_retriever import hybrid_search
# from retriever.dense_retriever import setup_vector_db

# setup_vector_db()

# query = "What is RAG"

# results = hybrid_search(query)

# print("\n=== DENSE RESULTS ===")

# for doc in results["dense"]:
#     print(doc["text"])
#     print(f"Source: {doc['source']}")
#     print(f"Chunk ID: {doc['chunk_id']}")
#     print("-" * 50)

# print("\n=== BM25 RESULTS ===")

# for item in results["bm25"]:
#     doc = item["document"]

#     print(doc["text"])
#     print(f"BM25 Score: {item['score']:.2f}")
#     print(f"Source: {doc['source']}")
#     print(f"Chunk ID: {doc['chunk_id']}")
#     print("-" * 50)

# from retriever.dense_retriever import setup_vector_db
# from retriever.hybrid_retriever import hybrid_search

# setup_vector_db()

# query = "What is RAG in AI"

# results = hybrid_search(query)

# print("\n=== MERGED RESULTS ===")

# for item in results:
#     print(item["text"])
#     print(f"Source: {item['source']}")
#     print(f"Chunk ID: {item['chunk_id']}")
#     print(f"Retrieved By: {item['retrieval_type']}")

#     if "score" in item:
#         print(f"BM25 Score: {item['score']:.2f}")

#     print("-" * 50)
# from retriever.dense_retriever import setup_vector_db
# from retriever.hybrid_retriever import hybrid_search
# from retriever.reranker import rerank_results

# top_n = 5
# setup_vector_db()

# query = "What is RAG"

# merged_results = hybrid_search(query)

# print("\n=== BEFORE RERANKING ===")

# for item in merged_results:
#     print(item["text"])
#     print(f"Source: {item['source']}")
#     print(f"Chunk ID: {item['chunk_id']}")
#     print(f"Retrieved By: {item['retrieval_type']}")
#     print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
#     print("-" * 50)
# merged_results = merged_results[:top_n]
# reranked_results = rerank_results(query, merged_results)

# print("\n=== AFTER RERANKING ===")

# for item in reranked_results:
#     print(item["text"])
#     print(f"Source: {item['source']}")
#     print(f"Chunk ID: {item['chunk_id']}")
#     print(f"Retrieved By: {item['retrieval_type']}")
#     print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
#     print(f"Rerank Score: {item.get('rerank_score', 0)}")
#     print(f"Final Score: {item.get('final_score', 0)}")
#     print("-" * 50) 

# from retriever.dense_retriever import setup_vector_db
# from pipeline import run_rag

# setup_vector_db()

# query = "What is RAG"

# result = run_rag(query)

# print("\n=== ANSWER ===")
# print(result["answer"])

# print("\n=== SOURCES ===")

# for item in result["sources"]:
#     print(item["text"])
#     print(f"Source: {item['source']}")
#     print(f"Chunk ID: {item['chunk_id']}")
#     print(f"Retrieved By: {item['retrieval_type']}")
#     print(f"Retrieval Score: {item.get('retrieval_score', 0)}")
#     print(f"Rerank Score: {item.get('rerank_score', 0)}")
#     print(f"Final Score: {item.get('final_score', 0)}")
#     print("-" * 50)
# from retriever.dense_retriever import setup_vector_db
# from evaluation.evaluator import evaluate_retrieval

# setup_vector_db()

# evaluate_retrieval()
# from retriever.dense_retriever import setup_vector_db
# from evaluation.evaluator import compare_configurations

# setup_vector_db()

# compare_configurations()
# from retriever.dense_retriever import setup_vector_db
# from evaluation.evaluator import evaluate_answers

# setup_vector_db()

# evaluate_answers()
from retriever.dense_retriever import setup_vector_db
from pipeline import run_rag
from evaluation.evaluator import (
    evaluate_retrieval,
    compare_configurations,
    evaluate_answers
)

setup_vector_db()

# Single Query Test
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

# Retrieval Evaluation
print("\n=== RETRIEVAL EVALUATION ===")
evaluate_retrieval()

# Configuration Comparison
print("\n=== CONFIGURATION COMPARISON ===")
compare_configurations()

# Answer Evaluation
print("\n=== ANSWER EVALUATION ===")
evaluate_answers()