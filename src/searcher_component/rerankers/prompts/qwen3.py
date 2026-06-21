"""Qwen3 prompt template for listwise reranking (single-turn).

Designed for Qwen3 models served via vLLM. Supports ``enable_thinking`` mode
where the model emits ``<think>...</think>`` reasoning before the ranking.
"""

from .base import PromptTemplate

QWEN3_TEMPLATE = PromptTemplate(
    name="qwen3",
    system_message=(
        "You are an intelligent assistant that can rank passages based on "
        "their relevancy to a query."
    ),
    prefix=(
        "I will provide you with {num} passages, each indicated by a numerical "
        "identifier []. Rank the passages based on their relevance to the "
        "search query: {query}.\n"
    ),
    body="[{rank}] {candidate}\n",
    suffix=(
        "Search Query: {query}.\n"
        "Rank the {num} passages above based on their relevance to the search "
        "query. All the passages should be included and listed using "
        "identifiers, in descending order of relevance. The output format "
        "should be [] > [], e.g., [2] > [1]. "
        "Only respond with the ranking results, do not say any word or explain."
    ),
)
