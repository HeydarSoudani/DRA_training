"""Random fusion: selects documents from subqueries in a uniformly random order."""

import copy
from typing import Any, Dict, List

import numpy as np

from .base import BaseFusion
from ._bandit_base import (
    SubqueryRandom,
    _build_doc_lookup,
    _extract_y_and_scores,
    _obs_to_ranked_list,
)


class RandomFusion(BaseFusion):
    """Fuse ranked lists by randomly selecting documents from subqueries.

    At each step, a random noise score is drawn per arm and the arm with the
    highest noise is chosen.  Within the chosen arm a document is sampled
    uniformly at random from the remaining (unseen) documents.

    This mirrors the ``sample_random`` function in subquery_.py.

    Args:
        rank_aware: If True, the reward for position n is discounted by
                    log₂(n + 2) (not used for selection here, only carried
                    through from the original interface).
    """

    def __init__(self, rank_aware: bool = False):
        self.rank_aware = rank_aware

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return list(ranked_lists[0])

        y, search_scores = _extract_y_and_scores(ranked_lists)
        doc_lookup = _build_doc_lookup(ranked_lists)

        actions = len(ranked_lists)
        total_budget = sum(len(rl) for rl in ranked_lists)
        ratio_documents = {0: total_budget}

        Subqueries = [SubqueryRandom(search_scores[i], y[i], self.rank_aware) for i in range(actions)]
        observations: Dict[int, list] = {}

        value_to_key = {v: k for k, v in ratio_documents.items()}
        steps_it = list(ratio_documents.values())
        last_step = max(steps_it)
        observations_per_step: Dict[Any, Dict] = {}

        for step in range(last_step):
            # Assign a random score to each arm and rank them
            random_samps = [q.get_sample_from_random() for q in Subqueries]
            ranked_indices = sorted(range(actions), key=lambda i: random_samps[i], reverse=True)

            # Pull from the highest-ranked available arm
            chosen_idx = None
            o = None
            for idx in ranked_indices:
                try:
                    _, o = Subqueries[idx].get_satisfaction_from_true_distribution()
                    chosen_idx = idx
                    break
                except (IndexError, ValueError):
                    continue

            if chosen_idx is None:
                break  # all arms exhausted

            observations.setdefault(chosen_idx, []).append((step, o))

            if (step + 1) in steps_it:
                observations_per_step[value_to_key[step + 1]] = copy.deepcopy(observations)

        if observations:
            observations_per_step[value_to_key[last_step]] = copy.deepcopy(observations)

        final_obs = observations_per_step.get(0, observations)
        return _obs_to_ranked_list(final_obs, doc_lookup)
