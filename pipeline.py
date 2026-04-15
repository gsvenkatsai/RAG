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
    query,k=5,
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
            q,k
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