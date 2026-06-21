"""Interleaving fusion classes for combining multiple retrieval results."""

from typing import List, Dict, Any

from .base import BaseFusion


class InterleavingFusion(BaseFusion):
    """Combine multiple ranked lists using round-robin interleaving.

    Selects documents from each list in blocks. With window=1 (default) it takes
    one item from each list per round; with window=k it takes k consecutive items
    from each list per round.

    Example:
        window=1: [A_0, B_0, C_0, A_1, B_1, C_1, ...]
        window=3: [A_0, A_1, A_2, B_0, B_1, B_2, C_0, C_1, C_2, A_3, ...]
    """

    def __init__(self, window: int = 1):
        """
        Args:
            window: Items taken from each list per round. Values <= 0 are treated as 1.
        """
        self.window = window if window and window > 0 else 1

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        fused_results = []
        seen_doc_ids = set()
        positions = [0] * len(ranked_lists)

        while any(pos < len(lst) for pos, lst in zip(positions, ranked_lists)):
            for list_idx, ranked_list in enumerate(ranked_lists):
                items_taken = 0
                while items_taken < self.window and positions[list_idx] < len(ranked_list):
                    result = ranked_list[positions[list_idx]]
                    doc_id = self._get_doc_id(result)

                    if doc_id and doc_id not in seen_doc_ids:
                        fused_results.append(result)
                        seen_doc_ids.add(doc_id)

                    positions[list_idx] += 1
                    items_taken += 1

        return fused_results


class NestedInterleavingFusion(BaseFusion):
    """Combine nested ranked lists using two-level interleaving.

    Applies interleaving at two levels:
    1. WITHIN each Search call: interleave documents from multiple Retrieve actions.
    2. ACROSS Search calls: interleave the per-Search results.

    Use case: ReAct model where each Search tool call contains multiple Retrieve actions.
    - Outer list : multiple Search tool calls
    - Middle list: multiple Retrieve actions within a Search call
    - Inner list : documents from one Retrieve action

    Example:
        Search 1: [[doc1_a, doc1_b], [doc1_c, doc1_d]]
        Search 2: [[doc2_a, doc2_b], [doc2_c, doc2_d]]

        Step 1 – interleave within each Search:
            Search 1 → [doc1_a, doc1_c, doc1_b, doc1_d]
            Search 2 → [doc2_a, doc2_c, doc2_b, doc2_d]

        Step 2 – interleave across Searches:
            Final → [doc1_a, doc2_a, doc1_c, doc2_c, doc1_b, doc2_b, doc1_d, doc2_d]
    """

    def __init__(self, window: int = 1):
        """
        Args:
            window: Window size forwarded to the inner InterleavingFusion.
        """
        self.window = window if window and window > 0 else 1
        self._interleaver = InterleavingFusion(window=self.window)

    def fuse(self, nested_ranked_lists: List[List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        # Step 1: interleave Retrieve actions within each Search call
        search_level_results = [
            interleaved
            for search_call_lists in nested_ranked_lists
            if (interleaved := self._interleaver.fuse(search_call_lists))
        ]

        # Step 2: interleave across Search calls
        return self._interleaver.fuse(search_level_results)
