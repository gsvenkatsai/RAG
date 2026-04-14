from prompt_builder import build_prompt
from generator import generate, validate_response
from retriever import retrieve, setup_vector_db
import time
# initialize DB once
def init():
    setup_vector_db()


def run_rag(query):
    start = time.time()
    context = retrieve(query)
    retrieval_time = time.time() - start
    # 🚨 Handle empty retrieval
    if not context or len(context) == 0:
        return {
            "answer": "I don't know.",
            "sources": [],
            "confidence": 0
        }

    prompt = build_prompt(query, context)
    answer = generate(prompt)
    answer = validate_response(answer, context)
    confidence = len(context)
    return {
        "answer": answer,
        "sources": context,
        "confidence": confidence,
        "retrieval_time": retrieval_time
    }