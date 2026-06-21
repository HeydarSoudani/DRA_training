"""RankGPT prompt template for listwise reranking (multi-turn)."""

from .base import PromptTemplate

RANKGPT_TEMPLATE = PromptTemplate(
    name="rankgpt",
    system_message=(
        "You are RankGPT, an intelligent assistant that can rank passages "
        "based on their relevancy to the query."
    ),
    prefix=(
        "I will provide you with {num} passages, each indicated by number "
        "identifier []. Rank the passages based on their relevance to query: "
        "{query}."
    ),
    body="[{rank}] {candidate}",
    suffix=(
        "Search Query: {query}.\n"
        "Rank the {num} passages above based on their relevance to the search "
        "query. The passages should be listed in descending order using "
        "identifiers. The most relevant passages should be listed first. "
        "The output format should be [] > [], e.g., [1] > [2]. "
        "Only response the ranking results, do not say any word or explain."
    ),
    multi_turn=True,
    prefix_ack="Okay, please provide the passages.",
    body_ack="Received passage [{rank}].",
)
