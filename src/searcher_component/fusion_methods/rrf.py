"""Reciprocal Rank Fusion (RRF) class for combining multiple retrieval results."""

from typing import List, Dict, Any

from .base import BaseFusion


class ReciprocalRankFusion(BaseFusion):
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Computes a score for each document based on its rank in every list:

        RRF_score(d) = Σ  1 / (k + rank(d))

    where k is a smoothing constant (default 60) and rank is 1-based.
    Documents are returned sorted by their RRF score in descending order.
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: Smoothing constant. Higher values reduce the impact of top-ranked
               documents. Typical value is 60.
        """
        self.k = k

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        # {doc_id: {"score": cumulative_rrf_score, "doc": result_dict}}
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                doc_id = self._get_doc_id(result)
                if not doc_id:
                    continue

                contribution = 1.0 / (self.k + rank)
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"score": contribution, "doc": result}
                else:
                    rrf_scores[doc_id]["score"] += contribution

        fused_results = []
        for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True):
            doc = item["doc"].copy()
            doc["rank_score"] = item["score"]
            fused_results.append(doc)

        return fused_results
