"""Bernoulli Thompson sampling fusion.

Models each subquery arm's relevance as a Bernoulli(p) variable with a
Beta(α, β) prior.  Scores are normalised to [0, 1] so they can be used
directly as soft-Bernoulli rewards.

All Bernoulli-bandit variants are unified here via constructor parameters:

    ThompsonBernoulliFusion()                       # plain Bernoulli TS
    ThompsonBernoulliFusion(ucb=True)               # + UCB exploration bonus
    ThompsonBernoulliFusion(k=5)                    # + top-k pulling
    ThompsonBernoulliFusion(diversity="linear")     # + linear diversity penalty
    ThompsonBernoulliFusion(diversity="concave")    # + concave diversity penalty
    ThompsonBernoulliFusion(k=5, ucb=True,
                            diversity="concave")    # all three combined
"""

import copy
from typing import Any, Dict, List, Optional

from .base import BaseFusion
from ._bandit_base import (
    SubqueryBeta,
    _build_doc_lookup,
    _embeddings_from_ranked_lists,
    _extract_y_and_scores,
    _obs_to_ranked_list,
    normalize_unit,
)


class ThompsonBernoulliFusion(BaseFusion):
    """Fuse ranked lists via Bernoulli / Beta Thompson sampling.

    At each step:
    1. Sample p_i ~ Beta(α_i, β_i) [+ UCB bonus if ucb=True] for every arm i.
    2. Pull k documents from the arm with the highest sample.
    3. Update that arm's posterior with the (diversity-modulated) reward.

    Args:
        k:                     Number of documents to pull per step (default 1).
        ucb:                   If True, add a UCB exploration bonus to Beta samples:
                               ``0.1 * sqrt(log(n+1) / |arm|)``.
        diversity:             Diversity penalty applied to each reward:
                               ``None`` (off), ``"linear"`` (1 - (max_cos+1)/2),
                               or ``"concave"`` (exp(-5 * max_cos^15)).
        rank_aware:            If True, discount reward at position n by log2(n+2).
        embeddings_dictionary: Doc-id -> embedding mapping (shape ``(1, d)``).
                               When *diversity* is not None and this is omitted,
                               embeddings are extracted automatically from the
                               ``'_emb'`` field attached by ``DenseRetriever``.
    """

    def __init__(self, k: int = 1, ucb: bool = False, diversity: Optional[str] = None, rank_aware: bool = False, embeddings_dictionary: Optional[Dict[str, Any]] = None):
        if diversity not in (None, "linear", "concave"):
            raise ValueError(
                f"diversity must be None, 'linear', or 'concave'; got {diversity!r}"
            )
        self.k = k
        self.ucb = ucb
        self.diversity = diversity
        self.rank_aware = rank_aware
        self.embeddings_dictionary: Dict[str, Any] = embeddings_dictionary or {}

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return list(ranked_lists[0])

        y, raw_scores = _extract_y_and_scores(ranked_lists)
        search_scores = normalize_unit(raw_scores)
        doc_lookup = _build_doc_lookup(ranked_lists)

        actions = len(ranked_lists)
        total_budget = sum(len(rl) for rl in ranked_lists)
        ratio_documents = {0: total_budget}

        Subqueries = [
            SubqueryBeta(search_scores[i], y[i], self.rank_aware) for i in range(actions)
        ]

        value_to_key = {v: k for k, v in ratio_documents.items()}
        steps_it = list(ratio_documents.values())
        last_step = max(steps_it)
        observations_per_step: Dict[Any, Dict] = {}

        diversity_kwargs: Dict[str, Any] = {}
        if self.diversity is not None:
            emb_dict = self.embeddings_dictionary or _embeddings_from_ranked_lists(ranked_lists)
            diversity_kwargs["embeddings_dictionary"] = emb_dict
            diversity_kwargs["diversity_concave"] = (self.diversity == "concave")

        for step in range(last_step):
            post_samps = [
                q.get_sample_from_current_distribution(ucb=self.ucb) for q in Subqueries
            ]
            ranked_indices = sorted(range(actions), key=lambda i: post_samps[i], reverse=True)

            chosen_idx = None
            o = None
            for idx in ranked_indices:
                try:
                    seen_obs = Subqueries[idx].seen_observations
                    _, o = Subqueries[idx].get_satisfaction_from_true_distribution(
                        Subqueries, seen_obs, k=self.k, **diversity_kwargs
                    )
                    chosen_idx = idx
                    break
                except IndexError:
                    continue

            if chosen_idx is None:
                break

            Subqueries[chosen_idx].update_current_distribution()
            Subqueries[chosen_idx].update_observations(step, o)

            if (step + 1) in steps_it:
                observations_per_step[value_to_key[step + 1]] = copy.deepcopy(
                    {k: Subqueries[k].seen_observations for k in range(actions)}
                )

        final_key = value_to_key.get(last_step, 0)
        final_obs = observations_per_step.get(
            final_key,
            {k: Subqueries[k].seen_observations for k in range(actions)},
        )
        return _obs_to_ranked_list(final_obs, doc_lookup)
