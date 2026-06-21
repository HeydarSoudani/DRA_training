"""Shared utilities and subquery arm classes for bandit-based fusion methods.

All bandit fusion classes share:
- I/O helpers that convert ranked_lists ↔ bandit format
- Subquery arm classes (Random, Gaussian, GaussianNIG, Beta)
- Diversity scoring helpers (linear and concave)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import invgamma, norm
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _get_doc_id(result: Dict[str, Any]) -> Optional[str]:
    return result.get("doc_id") or result.get("id")


def _extract_y_and_scores(ranked_lists: List[List[Dict[str, Any]]]) -> Tuple[List[List[Any]], List[List[float]]]:
    """Return (y, search_scores) from ranked_lists.

    y[i]             – list of doc ids for subquery i
    search_scores[i] – list of float scores for subquery i

    If a result dict has no 'score' or 'rank_score', falls back to 1/(rank+1).
    """
    y: List[List[Any]] = []
    search_scores: List[List[float]] = []
    for rl in ranked_lists:
        docs, scores = [], []
        for i, result in enumerate(rl):
            docs.append(_get_doc_id(result))
            score = result.get("score", result.get("rank_score", 1.0 / (i + 1)))
            scores.append(float(score))
        y.append(docs)
        search_scores.append(scores)
    return y, search_scores


def _embeddings_from_ranked_lists(ranked_lists: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Build an embeddings_dictionary from result dicts that carry a ``'_emb'`` field.

    Returns a mapping ``{doc_id: embedding}`` where each embedding has shape
    ``(1, dim)`` (compatible with sklearn ``cosine_similarity``).  Only docs that
    have an ``'_emb'`` key are included; first occurrence per doc_id wins.
    """
    emb_dict: Dict[str, Any] = {}
    for rl in ranked_lists:
        for result in rl:
            doc_id = _get_doc_id(result)
            if doc_id and "_emb" in result and doc_id not in emb_dict:
                emb_dict[doc_id] = result["_emb"]
    return emb_dict


def _build_doc_lookup(ranked_lists: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Map doc_id → result dict (first occurrence wins)."""
    doc_lookup: Dict[str, Dict[str, Any]] = {}
    for rl in ranked_lists:
        for result in rl:
            doc_id = _get_doc_id(result)
            if doc_id and doc_id not in doc_lookup:
                doc_lookup[doc_id] = result
    return doc_lookup


def _obs_to_ranked_list(final_obs: Dict[int, List[Tuple[int, Any]]], doc_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert bandit observations to a deduplicated ranked list.

    final_obs: {query_idx: [(step, doc_id), ...]}
    Documents are ordered by selection step (ascending); rank_score = 1/rank.
    """
    all_obs: List[Tuple[int, Any]] = []
    for obs_list in final_obs.values():
        for step, doc_id in obs_list:
            all_obs.append((step, doc_id))
    all_obs.sort(key=lambda x: x[0])

    seen: set = set()
    fused: List[Dict[str, Any]] = []
    for _, doc_id in all_obs:
        if doc_id is not None and doc_id not in seen:
            seen.add(doc_id)
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id].copy()
                doc["rank_score"] = 1.0 / (len(fused) + 1)
                fused.append(doc)
    return fused


# ---------------------------------------------------------------------------
# Score normalisation helpers
# ---------------------------------------------------------------------------

def normalize_gaussian(score_lists: List[List[float]]) -> List[List[float]]:
    """Map all scores to [-1, 1] via min-max normalisation."""
    flat = [s for sublist in score_lists for s in sublist]
    min_s, max_s = min(flat), max(flat)
    if max_s == min_s:
        return [[0.0 for _ in sublist] for sublist in score_lists]
    return [
        [2.0 * (s - min_s) / (max_s - min_s) - 1.0 for s in sublist]
        for sublist in score_lists
    ]


def normalize_unit(score_lists: List[List[float]]) -> List[List[float]]:
    """Map all scores to [0, 1] via min-max normalisation."""
    flat = [s for sublist in score_lists for s in sublist]
    min_s, max_s = min(flat), max(flat)
    if max_s == min_s:
        return [[0.5 for _ in sublist] for sublist in score_lists]
    return [
        [(s - min_s) / (max_s - min_s) for s in sublist]
        for sublist in score_lists
    ]


# ---------------------------------------------------------------------------
# Diversity metrics (from subquery_.py, lines 386-430)
# ---------------------------------------------------------------------------

def compute_diversity_metric(embeddings_dictionary: Dict[str, Any], new_observation: Any, previous_observations: List[Any]) -> float:
    """Linear diversity: 1 - (max cosine similarity + 1) / 2."""
    try:
        new_embedding = embeddings_dictionary[new_observation]
    except KeyError:
        return 1.0

    cosine_sims = []
    for doc_key in previous_observations:
        try:
            cosine_sims.append(
                cosine_similarity(new_embedding, embeddings_dictionary[doc_key])[0][0]
            )
        except KeyError:
            return 1.0

    if cosine_sims:
        return 1.0 - (max(cosine_sims) + 1.0) / 2.0
    return 1.0


def compute_diversity_metric_concave(embeddings_dictionary: Dict[str, Any], new_observation: Any, previous_observations: List[Any]) -> float:
    """Concave diversity: exp(-a * max_cosine^b) with a=5, b=15."""
    a, b = 5, 15
    try:
        new_embedding = embeddings_dictionary[new_observation]
    except KeyError:
        return 1.0

    cosine_sims = []
    for doc_key in previous_observations:
        try:
            cosine_sims.append(
                cosine_similarity(new_embedding, embeddings_dictionary[doc_key])[0][0]
            )
        except KeyError:
            return 1.0

    if cosine_sims:
        max_cosine = max(cosine_sims)
        if max_cosine < 0:
            return 1.0
        return float(np.exp(-a * (max_cosine ** b)))
    return 1.0


# ---------------------------------------------------------------------------
# Subquery arm classes
# ---------------------------------------------------------------------------

class _SubqueryBase:
    """Base arm carrying scores and observations for one subquery.

    Attributes:
        scores:        list of relevance scores for each document in this arm
        observations:  list of doc ids corresponding to scores
        rank_aware:    if True, discount reward by log2(n+2) at step n
        n:             number of documents pulled so far
        sum_satisfaction: cumulative reward
        last:          most recent reward
    """

    def __init__(self, scores: List[float], observations: List[Any], rank_aware: bool = False):
        self.scores = scores
        self.observations = observations
        self.rank_aware = rank_aware

    def get_satisfaction_from_true_distribution(self, Subqueries: List["_SubqueryBase"], seen_observations: List[Tuple[int, Any]], k: int = 1, embeddings_dictionary: Optional[Dict[str, Any]] = None, diversity_concave: bool = False) -> Tuple[Any, Any]:
        """Pull the next document from this arm and return (score, doc_id).

        Raises IndexError when the arm is exhausted.
        """
        if embeddings_dictionary is None:
            embeddings_dictionary = {}

        rank = 1.0 if not self.rank_aware else math.log2(self.n + 2)

        if k == 1:
            s = self.scores[self.n] / rank   # raises IndexError when exhausted
            o = self.observations[self.n]
            if embeddings_dictionary:
                all_docs = [obs[1] for sq in Subqueries for obs in sq.seen_observations]
                if diversity_concave:
                    s = s * compute_diversity_metric_concave(embeddings_dictionary, o, all_docs)
                else:
                    s = s * compute_diversity_metric(embeddings_dictionary, o, all_docs)
            self.n += 1
            self.sum_satisfaction += s
            self.last = s
        else:
            if self.n >= len(self.scores):
                raise IndexError("Arm exhausted.")
            end = min(self.n + k, len(self.scores))
            s = list(self.scores[self.n:end])
            o_list = list(self.observations[self.n:end])
            if embeddings_dictionary:
                all_docs = [obs[1] for sq in Subqueries for obs in sq.seen_observations]
                for i in range(len(s)):
                    if diversity_concave:
                        s[i] = s[i] * compute_diversity_metric_concave(
                            embeddings_dictionary, o_list[i], all_docs
                        )
                    else:
                        s[i] = s[i] * compute_diversity_metric(
                            embeddings_dictionary, o_list[i], all_docs
                        )
            self.n = end
            self.sum_satisfaction += sum(s) / k
            self.last = sum(s) / k
            o = o_list[0] if o_list else None
            return s, o

        return s, o


class SubqueryRandom:
    """Arm that selects documents uniformly at random (no bandit learning)."""

    def __init__(self, scores: List[float], observations: List[Any], rank_aware: bool = False):
        self.scores = scores
        self.observations = observations
        self.rank_aware = rank_aware
        self.seen: List[int] = []
        self.n: int = 0

    def get_sample_from_random(self) -> float:
        return float(np.random.normal(0.0, 1.0))

    def get_satisfaction_from_true_distribution(self, k: int = 1) -> Tuple[float, Any]:
        if self.rank_aware:
            idx = self.n
        else:
            candidates = [i for i in range(len(self.scores)) if i not in self.seen]
            if not candidates:
                raise ValueError("Arm exhausted.")
            idx = int(np.random.choice(candidates))
        self.n += 1
        self.seen.append(idx)
        return self.scores[idx], self.observations[idx]


class SubqueryGaussian(_SubqueryBase):
    """Arm with Gaussian (known variance) Thompson sampling.

    Prior: μ ~ N(0, 100²).  Posterior updated via conjugate update.
    """

    def __init__(self, scores: List[float], observations: List[Any]):
        self.prior_sigma: float = 100.0
        self.post_mu: float = 0.0
        self.post_sigma: float = 100.0
        self.n: int = 0
        self.sum_satisfaction: float = 0.0
        self.last: Optional[float] = None
        self.seen_observations: List[Tuple[int, Any]] = []
        super().__init__(scores, observations)

    def get_mu_from_current_distribution(self) -> float:
        return float(np.random.normal(self.post_mu, self.post_sigma))

    def update_current_distribution(self) -> None:
        self.post_sigma = np.sqrt((1.0 / self.prior_sigma ** 2 + self.n) ** -1)
        self.post_mu = self.post_sigma ** 2 * self.sum_satisfaction

    def update_observations(self, step: int, o: Any) -> None:
        self.seen_observations.append((step, o))


class SubqueryGaussianNIG(_SubqueryBase):
    """Arm with Normal-Inverse-Gamma (NIG) Thompson sampling.

    Prior: (μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀).
    """

    def __init__(self, scores: List[float], observations: List[Any], mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.mu_n, self.kappa_n = mu0, kappa0
        self.alpha_n, self.beta_n = alpha0, beta0
        self.data: List[float] = []
        self.last: Optional[float] = None
        self.seen_observations: List[Tuple[int, Any]] = []
        self.n: int = 0
        self.sum_satisfaction: float = 0.0
        super().__init__(scores, observations)

    def get_mu_from_current_distribution(self) -> float:
        sigma2 = float(invgamma.rvs(a=self.alpha_n, scale=self.beta_n))
        return float(norm.rvs(loc=self.mu_n, scale=np.sqrt(sigma2 / self.kappa_n)))

    def update_current_distribution(self, new_reward: float) -> None:
        self.data.append(new_reward)
        n = len(self.data)
        mean_x = float(np.mean(self.data))
        S = float(np.sum((np.array(self.data) - mean_x) ** 2))
        self.kappa_n = self.kappa0 + n
        self.mu_n = (self.kappa0 * self.mu0 + n * mean_x) / self.kappa_n
        self.alpha_n = self.alpha0 + n / 2.0
        self.beta_n = (
            self.beta0
            + 0.5 * S
            + (self.kappa0 * n * (mean_x - self.mu0) ** 2) / (2.0 * self.kappa_n)
        )

    def update_observations(self, step: int, o: Any) -> None:
        self.seen_observations.append((step, o))


class SubqueryBeta(_SubqueryBase):
    """Arm with Beta / Bernoulli Thompson sampling.

    Prior: p ~ Beta(1, 1) (uniform).  Posterior updated via conjugate update.
    Optionally adds a UCB exploration bonus.
    """

    def __init__(self, scores: List[float], observations: List[Any], rank_aware: bool = False):
        self.post_alpha: float = 1.0
        self.post_beta: float = 1.0
        self.n: int = 0
        self.sum_satisfaction: float = 0.0
        self.last: Optional[float] = None
        self.seen_observations: List[Tuple[int, Any]] = []
        super().__init__(scores, observations, rank_aware)

    def get_sample_from_current_distribution(self, ucb: bool = False) -> float:
        sample = float(np.random.beta(self.post_alpha, self.post_beta))
        if ucb:
            sample += 0.1 * np.sqrt(np.log(self.n + 1) / max(len(self.scores), 1))
        return sample

    def update_current_distribution(self) -> None:
        if self.last is not None:
            self.post_alpha += self.last
            self.post_beta += 1.0 - self.last

    def update_observations(self, step: int, o: Any) -> None:
        self.seen_observations.append((step, o))
