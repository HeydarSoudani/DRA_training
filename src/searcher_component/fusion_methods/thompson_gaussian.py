"""Gaussian Thompson sampling fusion.

Models each subquery arm's mean reward using either:
- known variance (default): mu_i ~ N(mu_n, sigma_n^2)
- Normal-Inverse-Gamma prior (nig=True): (mu_i, sigma^2_i) ~ NIG(mu0, kappa0, alpha0, beta0)

Scores are normalised to [-1, 1] before use.

Both Gaussian variants are unified here via the ``nig`` constructor flag:

    ThompsonGaussianFusion()          # known-variance Gaussian TS
    ThompsonGaussianFusion(nig=True)  # NIG prior (jointly models mean + variance)
"""

import copy
from typing import Any, Dict, List

from .base import BaseFusion
from ._bandit_base import (
    SubqueryGaussian,
    SubqueryGaussianNIG,
    _build_doc_lookup,
    _extract_y_and_scores,
    _obs_to_ranked_list,
    normalize_gaussian,
)


class ThompsonGaussianFusion(BaseFusion):
    """Fuse ranked lists via Gaussian Thompson sampling.

    At each step:
    1. Sample mu_i from the arm's posterior for every arm i.
    2. Pull from the arm with the highest sample.
    3. Update that arm's posterior with the observed reward.

    Args:
        nig:    If True, use a Normal-Inverse-Gamma (NIG) prior that jointly
                models (mu, sigma^2).  If False (default), use a Gaussian
                prior with known variance.
        mu0:    NIG prior mean of mu (only used when nig=True).
        kappa0: NIG prior pseudo-count for mu (only used when nig=True).
        alpha0: NIG shape of the Inverse-Gamma prior on sigma^2 (nig=True only).
        beta0:  NIG scale of the Inverse-Gamma prior on sigma^2 (nig=True only).
    """

    def __init__(self, nig: bool = False, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        self.nig = nig
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fuse(self, ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return list(ranked_lists[0])

        y, raw_scores = _extract_y_and_scores(ranked_lists)
        search_scores = normalize_gaussian(raw_scores)
        doc_lookup = _build_doc_lookup(ranked_lists)

        actions = len(ranked_lists)
        total_budget = sum(len(rl) for rl in ranked_lists)
        ratio_documents = {0: total_budget}

        if self.nig:
            Subqueries = [
                SubqueryGaussianNIG(
                    search_scores[i], y[i],
                    mu0=self.mu0, kappa0=self.kappa0,
                    alpha0=self.alpha0, beta0=self.beta0,
                )
                for i in range(actions)
            ]
        else:
            Subqueries = [SubqueryGaussian(search_scores[i], y[i]) for i in range(actions)]

        value_to_key = {v: k for k, v in ratio_documents.items()}
        steps_it = list(ratio_documents.values())
        last_step = max(steps_it)
        observations_per_step: Dict[Any, Dict] = {}

        for step in range(last_step):
            post_samps = [q.get_mu_from_current_distribution() for q in Subqueries]
            ranked_indices = sorted(range(actions), key=lambda i: post_samps[i], reverse=True)

            chosen_idx = None
            s = None
            o = None
            for idx in ranked_indices:
                try:
                    seen_obs = Subqueries[idx].seen_observations
                    s, o = Subqueries[idx].get_satisfaction_from_true_distribution(
                        Subqueries, seen_obs
                    )
                    chosen_idx = idx
                    break
                except IndexError:
                    continue

            if chosen_idx is None:
                break

            if self.nig:
                Subqueries[chosen_idx].update_current_distribution(s)
            else:
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
