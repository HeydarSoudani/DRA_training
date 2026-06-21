"""Reranker Selector Component for filtering documents based on relevance scores.

This module implements a standalone reranker selector component that truncates a ranked
list of documents based on their relevance scores (rank_score). The selector removes
documents with low relevance scores (typically 0-2) and keeps only those with higher
scores (3-5), making the returned list length dynamic based on quality.

Usage:
    The reranker selector is designed to work after a reranker component that assigns
    rank_score values from 0 to 5 to each document.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RerankerSelector:
    """Reranker Selector for filtering documents based on relevance scores.

    This component truncates a ranked list based on the rank_score field of each
    document. Documents with scores below the threshold are filtered out, resulting
    in a variable-length output list based on document quality.

    Attributes:
        score_threshold: Minimum score (inclusive) for a document to be kept.
                        Documents with rank_score < threshold are filtered out.
                        Default is 3 (keeps scores 3, 4, 5).
        verbose: Whether to print detailed logs.
    """

    def __init__( self, score_threshold: float = 3.0, verbose: bool = True, ):
        """Initialize the reranker selector.

        Args:
            score_threshold: Minimum score (inclusive) to keep a document.
                           Default is 3.0 (keeps documents with scores >= 3).
            verbose: If True, print detailed logs; if False, print minimal logs.
        """
        self.score_threshold = score_threshold
        self.verbose = verbose

    def select( self, documents: List[Dict[str, Any]], score_field: str = "rank_score", ) -> List[Dict[str, Any]]:
        """Filter documents based on their relevance scores.

        This method filters the input document list, keeping only documents with
        a score >= score_threshold. The order of documents is preserved.

        Args:
            documents: List of documents to filter. Each document should have a
                      score field (default: "rank_score").
            score_field: Name of the field containing the relevance score.
                        Default is "rank_score".

        Returns:
            Filtered list of documents with scores >= score_threshold.
            The list length is variable based on how many documents pass the threshold.

        Examples:
            >>> selector = RerankerSelector(score_threshold=3.0)
            >>> docs = [
            ...     {"doc_id": "1", "rank_score": 5},
            ...     {"doc_id": "2", "rank_score": 2},
            ...     {"doc_id": "3", "rank_score": 4},
            ...     {"doc_id": "4", "rank_score": 1},
            ... ]
            >>> filtered = selector.select(docs)
            >>> len(filtered)
            2
            >>> [d["doc_id"] for d in filtered]
            ['1', '3']
        """
        if not documents:
            if self.verbose:
                logger.info("Reranker Selector: No documents to filter")
            return []

        # Filter documents based on score threshold
        filtered_documents = []
        filtered_count = 0

        for doc in documents:
            score = doc.get(score_field, 0.0)

            # Keep documents with score >= threshold
            if score >= self.score_threshold:
                filtered_documents.append(doc)
            else:
                filtered_count += 1

        if self.verbose:
            logger.info(
                f"Reranker Selector: Filtered {len(documents)} documents → "
                f"{len(filtered_documents)} kept (threshold >= {self.score_threshold}), "
                f"{filtered_count} removed"
            )

        return filtered_documents


def setup_reranker_selector( score_threshold: float = 3.0, verbose: bool = True, ) -> RerankerSelector:
    """Setup and return a reranker selector instance.

    Args:
        score_threshold: Minimum score (inclusive) to keep a document.
                        Default is 3.0.
        verbose: If True, print detailed logs.

    Returns:
        Configured RerankerSelector instance.

    Example:
        >>> selector = setup_reranker_selector(score_threshold=3.0)
        >>> filtered_docs = selector.select(documents)
    """
    return RerankerSelector(
        score_threshold=score_threshold,
        verbose=verbose,
    )
