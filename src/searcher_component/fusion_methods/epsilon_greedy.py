"""Epsilon-greedy fusion: exploit a randomly chosen subquery until it hits
a zero-score document, then pick another subquery at random."""

from typing import Any, Dict, List

import numpy as np

from .base import BaseFusion
from ._bandit_base import (
    _build_doc_lookup,
    _extract_y_and_scores,
    _obs_to_ranked_list,
)


class EpsilonGreedyFusion(BaseFusion):
    """Fuse ranked lists using an ε-greedy exploit-until-irrelevant strategy.

    A subquery is picked at random.  Its documents are retrieved sequentially
    until a document whose score equals ``stop_score`` is encountered (the
    "irrelevant" signal).  A new subquery is then chosen at random, and the
    process repeats until ``total_budget`` documents have been collected.

    This mirrors the ``epsilon_greedy`` function in subquery_.py.

    Args:
        stop_score: Score value that signals the end of a subquery's useful
                    region.  Defaults to 0.  Set to ``None`` to disable early
                    stopping (always exhaust each chosen subquery).
    """

    def __init__(self, stop_score: float = 0.0):
        self.stop_score = stop_score

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return list(ranked_lists[0])

        y, search_scores = _extract_y_and_scores(ranked_lists)
        doc_lookup = _build_doc_lookup(ranked_lists)

        no_queries = len(ranked_lists)
        total_budget = sum(len(rl) for rl in ranked_lists)

        seen_idx = {i: 0 for i in range(no_queries)}
        seen_n = 0
        observations: Dict[int, list] = {}

        while seen_n < total_budget:
            available = [i for i in range(no_queries) if seen_idx[i] < len(y[i])]
            if not available:
                break

            idx_q = int(np.random.choice(available))

            # Exhaust this subquery until stop_score or end of list
            while seen_idx[idx_q] < len(y[idx_q]):
                idx_doc = seen_idx[idx_q]
                obs = y[idx_q][idx_doc]
                search_score = search_scores[idx_q][idx_doc]

                observations.setdefault(idx_q, []).append((seen_n, obs))
                seen_idx[idx_q] += 1
                seen_n += 1

                if seen_n >= total_budget:
                    break

                if self.stop_score is not None and search_score <= self.stop_score:
                    break

        return _obs_to_ranked_list(observations, doc_lookup)
