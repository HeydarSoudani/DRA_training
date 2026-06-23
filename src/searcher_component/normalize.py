"""Normalize retriever responses into a consistent document schema.

Local retrievers (``retriever.py``) return documents keyed by ``id``/``contents``.
Downstream code — fusion, reranking, evaluation, agent plan tools — expects a
consistent ``doc_id`` / ``relevant_text`` / ``title`` schema. This module is the
single place that bridges the two.
"""


def normalize_retrieval_response(response):
    """
    Normalize retrieval response to handle local retrievers.

    Local retrievers (from retriever.py) return a list directly: [...]

    This function also normalizes the document key from 'id' to 'doc_id' for consistency
    across the pipeline (fusion, reranking, evaluation all expect 'doc_id').

    Args:
        response: Response from retriever (either list or dict)

    Returns:
        List of retrieved documents with normalized 'doc_id' key
    """
    # Extract the list of documents
    if isinstance(response, dict):
        docs = response.get("results", [])
    elif isinstance(response, list):
        docs = response
    else:
        # Fallback for unexpected types - return empty list
        return []

    # Normalize key names so downstream code (passages2string, plan tools,
    # evaluators) can rely on a consistent schema regardless of retriever type.
    #
    # Local retrievers use:  id, contents, title (top-level)
    normalized_docs = []
    for doc in docs:
        if isinstance(doc, dict):
            doc = doc.copy()  # Don't modify the original

            # id → doc_id
            if 'id' in doc and 'doc_id' not in doc:
                doc['doc_id'] = doc['id']

            # contents → relevant_text
            if 'contents' in doc and 'relevant_text' not in doc:
                doc['relevant_text'] = doc['contents']

            # Promote metadata.title to top-level when present
            if 'title' not in doc and 'metadata' in doc:
                title = doc['metadata'].get('title')
                if title:
                    doc['title'] = title

            # Ensure metadata.title exists for code that reads metadata dict
            if 'title' in doc:
                metadata = doc.get('metadata', {})
                if 'title' not in metadata:
                    doc['metadata'] = {**metadata, 'title': doc['title']}

            normalized_docs.append(doc)
        else:
            # Keep non-dict items as-is (shouldn't happen, but be safe)
            normalized_docs.append(doc)

    return normalized_docs
