"""Simple concatenation fusion class for combining multiple retrieval results."""

from typing import List, Dict, Any

from .base import BaseFusion


def _score(result: Dict[str, Any]) -> float:
    """Return the best available score from a result dict."""
    return float(result.get("rank_score", result.get("score", 0)))


class SimpleConcatenation(BaseFusion):
    """Combine multiple ranked lists by concatenation with deduplication.

    Iterates through all lists in order, keeping the first occurrence of each
    document. If a document appears more than once, the copy with the higher
    score is retained. The final list is sorted by score descending.
    Accepts both 'rank_score' and 'score' fields.
    """

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        seen_docs: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for result in ranked_list:
                doc_id = self._get_doc_id(result)
                if not doc_id:
                    continue

                if doc_id not in seen_docs:
                    seen_docs[doc_id] = result
                elif _score(result) > _score(seen_docs[doc_id]):
                    seen_docs[doc_id] = result

        return sorted(seen_docs.values(), key=_score, reverse=True)
