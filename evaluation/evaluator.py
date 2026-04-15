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