"""Prompt template for Rank1 pointwise reranking (jhu-clsp/rank1)."""

# Matches the original Rank1 implementation from orionw/rank1
RANK1_RELEVANCE_PROMPT = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'.\n"
    "Query: {query}\n"
    "Passage: {passage}\n"
    "<think>"  # forces the model to begin its reasoning chain
)

# Score returned when true/false logits cannot be extracted
RANK1_INCOMPLETE_SCORE = 0.5
