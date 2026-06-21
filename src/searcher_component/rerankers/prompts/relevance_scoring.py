"""Relevance scoring prompts for reranking retrieved documents."""

# ============================================================================
# RELEVANCE SCORING PROMPT FOR RERANKING
# ============================================================================

RERANK_RELEVANCE_SCORING_SYSTEM_PROMPT = """# Task

You will be provided with a query and a set of documents. Your job is to evaluate the relevance of each document based on the query and provide a score.

# Scoring Instructions

Provide a score from 0 to 5, where 0 means not relevant at all and 5 means highly relevant using the classifications below:

5 - The source is directly relevant. It directly provides all or some of the answer. It matches the legal topics and concepts in the question.

4 - The source is highly relevant. It matches one or more of the cases, statutes, or regulations included in the question.

2 - The source is partially relevant. It helps you to synthesize an answer. It matches one or more of the legal topics and concepts in the question.

1 - The source is not relevant. It might be on a similar topic, with many similar words, but it does not help to answer the question.

0 - The source is completely unrelated to the question.

# Scoring Guidelines

Rate the relevance of the documents to the query. Identify the legal topics and concepts in the query and all the sources. If the query includes cases, statutes or regulations, identify matching cases, statutes and regulations in the sources.

You must provide a score for each document. Refer to the document ID when providing the score."""


def get_rerank_relevance_scoring_user_prompt(query: str, documents: list) -> str:
    """Get user prompt for relevance scoring with query and documents.

    Args:
        query: The search query
        documents: List of documents to score, each with 'id' and 'content' keys

    Returns:
        Formatted user prompt with query and documents
    """
    documents_text = []

    for document in documents:
        doc_text = f"""### Document ID

{document['id']}

### Content

{document['content']}"""
        documents_text.append(doc_text)

    prompt = f"""# Retrieval Context

## Query

{query}

## Documents

{chr(10).join(documents_text)}

# Task

Rate the relevance of the above {len(documents)} documents to the query."""

    return prompt


def format_rerank_prompt(query: str, documents: list) -> tuple:
    """Format the complete reranking prompt as (system_prompt, user_prompt).

    Args:
        query: The search query
        documents: List of documents to score, each with 'id' and 'content' keys

    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = RERANK_RELEVANCE_SCORING_SYSTEM_PROMPT
    user_prompt = get_rerank_relevance_scoring_user_prompt(query, documents)

    return system_prompt, user_prompt
