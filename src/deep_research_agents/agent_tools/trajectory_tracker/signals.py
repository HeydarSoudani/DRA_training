"""Signal classes for the trajectory tracker.

Provides three analysis-aligned signals and the aspect coverage signal:

- ``DocNoveltySignal``: fraction of retrieved docs not seen before.
- ``ConsecQuerySimilaritySignal``: cosine similarity between consecutive
  iterations' mean subquery embeddings.
- ``OrigQuerySimilaritySignal``: cosine similarity between current
  iteration and the original query.
- ``AspectCoverageSignal``: LLM-based query-aspect coverage tracking.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from prompts.trajectory_tracker.aspect_coverage import (
    ASPECT_INIT_SYSTEM,
    ASPECT_UPDATE_SYSTEM,
    ASPECT_INIT_STATIC_SYSTEM,
    ASPECT_UPDATE_STATIC_SYSTEM,
    ASPECT_INIT_USER_TEMPLATE,
    ASPECT_UPDATE_USER_TEMPLATE,
    ASPECT_INIT_STATIC_USER_TEMPLATE,
    ASPECT_UPDATE_STATIC_USER_TEMPLATE,
    FROZEN_INSTRUCTION,
    UNFROZEN_INSTRUCTION,
    Aspect,
    AspectCoverageSummary,
    extract_aspect_coverage,
    extract_aspect_actions,
    apply_aspect_actions,
    format_aspects_for_prompt,
    format_doc_snippets,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalised_mean(embeddings: List[np.ndarray]) -> np.ndarray:
    """L2-normalised mean — mirrors ``_mean_embedding`` in the analysis."""
    mean = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


def encode_fn_from_retriever(retriever) -> Optional[Callable]:
    """Extract an ``encode_fn`` compatible with TrajectoryTracker from a retriever.

    Works with ``DenseRetriever`` (has ``.encoder.encode``).  Returns ``None``
    for retriever types that don't expose a local encoder (BM25, SPLADE,
    endpoint-based retrievers).
    """
    encoder = getattr(retriever, "encoder", None)
    if encoder is None or not callable(getattr(encoder, "encode", None)):
        return None

    def _encode(texts: List[str]) -> np.ndarray:
        return encoder.encode(texts, is_query=True)

    return _encode


# ---------------------------------------------------------------------------
# Lightweight signals (pure computation, no LLM calls)
# ---------------------------------------------------------------------------

class DocNoveltySignal:
    """Measures what fraction of retrieved docs are *new* (not seen before).

    Tracks all doc_ids seen so far across iterations.  Returns a score in
    [0, 1] where 1 means all docs are novel and 0 means every doc was
    already retrieved in a previous step.
    """

    def __init__(self) -> None:
        self._seen_ids: Set[str] = set()

    def reset(self) -> None:
        self._seen_ids.clear()

    def score(self, docs: List[Dict[str, Any]]) -> tuple:
        """Return ``(novelty_fraction, num_novel_docs)``."""
        current_ids = {
            doc.get("doc_id") or doc.get("id") or ""
            for doc in docs
        }
        current_ids.discard("")

        if not current_ids:
            return 0.0, 0

        new_ids = current_ids - self._seen_ids
        novelty = len(new_ids) / len(current_ids)

        # Update state *after* computing novelty
        self._seen_ids.update(current_ids)
        return novelty, len(new_ids)


class ConsecQuerySimilaritySignal:
    """Cosine similarity between consecutive iterations' mean subquery embeddings.

    Matches ``compute_consecutive_similarity`` in the analysis code.
    Returns raw cosine similarity (typically in [0, 1] for text).
    ``None`` for the first iteration (no previous to compare against) or
    when no encoder is available.

    Accepts a list of subqueries per iteration (e.g. AgentCPM, webWeaver).
    All subqueries are encoded, their mean embedding represents the
    iteration, and that mean is compared with the previous iteration's mean.
    """

    def __init__(self, encode_fn: Optional[Callable] = None) -> None:
        self._encode_fn = encode_fn
        self._iter_embeddings: Dict[int, List[np.ndarray]] = {}
        self._iter_mean: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._iter_embeddings.clear()
        self._iter_mean.clear()

    def score(self, subqueries: List[str], iter_num: int) -> Optional[float]:
        """Return cosine similarity with the previous iteration, or ``None``."""
        if self._encode_fn is None:
            return None

        subqueries = [sq for sq in subqueries if sq.strip()]
        if not subqueries:
            return None

        embeddings = self._encode_fn(subqueries)  # (N, D)
        emb_list = [embeddings[i] for i in range(embeddings.shape[0])]

        self._iter_embeddings[iter_num] = emb_list
        self._iter_mean[iter_num] = _l2_normalised_mean(emb_list)

        prev_iters = [k for k in self._iter_mean if k < iter_num]
        if not prev_iters:
            return None
        prev = max(prev_iters)

        return float(np.dot(self._iter_mean[iter_num], self._iter_mean[prev]))


class OrigQuerySimilaritySignal:
    """Cosine similarity between current iteration and the original query.

    Matches ``compute_original_query_similarity`` in the analysis code.
    Returns raw cosine similarity (typically in [0, 1] for text).
    ``None`` when no encoder is available.

    Accepts a list of subqueries per iteration. All subqueries are encoded,
    their mean embedding represents the iteration, and that mean is compared
    with the original query embedding.
    """

    def __init__(self, encode_fn: Optional[Callable] = None) -> None:
        self._encode_fn = encode_fn
        self._orig_emb: Optional[np.ndarray] = None
        self._iter_embeddings: Dict[int, List[np.ndarray]] = {}

    def reset(self) -> None:
        self._orig_emb = None
        self._iter_embeddings.clear()

    def score(
        self, subqueries: List[str], original_query: str, iter_num: int,
    ) -> Optional[float]:
        """Return cosine similarity with the original query, or ``None``."""
        if self._encode_fn is None or not original_query.strip():
            return None

        subqueries = [sq for sq in subqueries if sq.strip()]
        if not subqueries:
            return None

        if self._orig_emb is None:
            emb = self._encode_fn([original_query]).squeeze()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._orig_emb = emb

        embeddings = self._encode_fn(subqueries)  # (N, D)
        emb_list = [embeddings[i] for i in range(embeddings.shape[0])]

        self._iter_embeddings[iter_num] = emb_list
        mean = _l2_normalised_mean(emb_list)
        return float(np.dot(mean, self._orig_emb))


# ---------------------------------------------------------------------------
# Aspect coverage signal (LLM-based)
# ---------------------------------------------------------------------------

class AspectCoverageSignal:
    """Tracks query-aspect coverage across search iterations.

    Maintains a list of information-need aspects for the query and updates
    their coverage status as new evidence is retrieved.  Produces a
    structured summary for logging and decision-maker consumption.

    Two modes:
    - ``"static"``: aspects are extracted from the query text directly
      (e.g. BrowseCompPlus where the query enumerates required aspects).
      Only tick operations are performed; the list is always frozen.
    - ``"dynamic"``: an LLM decomposes the query and updates the list
      each iteration.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        mode: ``"static"`` or ``"dynamic"``.
        max_aspects: Soft cap on the number of aspects.
        stabilization_window: After this many consecutive iterations with
            no structural changes (add/remove), the list is frozen.
        max_tokens: Max tokens for LLM generation.
        temperature: LLM temperature (default 0.0 for determinism).
    """

    def __init__(
        self,
        llm_client: Any = None,
        mode: str = "dynamic",
        max_aspects: int = 8,
        stabilization_window: int = 15,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self._mode = mode
        self._max_aspects = max_aspects
        self._stabilization_window = stabilization_window
        self._max_tokens = max_tokens
        self._temperature = temperature

        # State (cleared on reset)
        self._aspects: List[Aspect] = []
        self._initialized: bool = False
        self._frozen: bool = mode == "static"
        self._last_structure_change_iter: int = 0

    def reset(self) -> None:
        """Clear all state for a new query."""
        self._aspects.clear()
        self._initialized = False
        self._frozen = self._mode == "static"
        self._last_structure_change_iter = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, query: str) -> AspectCoverageSummary:
        """Decompose/extract the query into initial aspects."""
        if self._mode == "static" and self._aspects:
            self._initialized = True
            return self._build_summary(iter_num=0)

        if self._llm is None:
            logger.warning("AspectCoverageSignal: no LLM client, cannot initialize")
            self._initialized = True
            return self._build_summary(iter_num=0)

        min_aspects = 2
        if self._mode == "static":
            system_prompt = ASPECT_INIT_STATIC_SYSTEM.format(
                min_aspects=min_aspects,
                max_aspects=self._max_aspects,
            )
            user_msg = ASPECT_INIT_STATIC_USER_TEMPLATE.format(
                query=query,
                min_aspects=min_aspects,
                max_aspects=self._max_aspects,
            )
        else:
            system_prompt = ASPECT_INIT_SYSTEM.format(
                min_aspects=min_aspects,
                max_aspects=self._max_aspects,
            )
            user_msg = ASPECT_INIT_USER_TEMPLATE.format(
                query=query,
                min_aspects=min_aspects,
                max_aspects=self._max_aspects,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._llm.complete(
                messages, max_tokens=self._max_tokens, temperature=self._temperature,
            )
        except Exception:
            logger.warning("AspectCoverageSignal: init LLM call failed", exc_info=True)
            self._initialized = True
            return self._build_summary(iter_num=0)

        summary = extract_aspect_coverage(raw)
        if summary is None:
            logger.warning("AspectCoverageSignal: init parsing failed")
            self._initialized = True
            return self._build_summary(iter_num=0)

        self._aspects = summary.aspects[:self._max_aspects]
        for a in self._aspects:
            a.status = "not_covered"
            a.evidence = ""
        self._initialized = True

        result = self._build_summary(iter_num=0)
        logger.info(
            "AspectCoverageSignal initialized with %d aspects: %s",
            len(self._aspects), [a.name for a in self._aspects],
        )
        return result

    def set_static_aspects(self, aspect_names: List[str]) -> None:
        """Set the aspect list directly for static mode."""
        self._aspects = [Aspect(name=n, status="not_covered") for n in aspect_names]
        self._initialized = True
        self._frozen = True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        iter_num: int,
        docs: List[Dict[str, Any]],
        subqueries: List[str],
        query: str,
    ) -> AspectCoverageSummary:
        """Update aspect coverage based on new evidence.

        Auto-initializes on first call if not already initialized.
        """
        if not self._initialized:
            self.initialize(query)

        if self._llm is None:
            return self._build_summary(iter_num=iter_num)

        if not self._aspects:
            return self._build_summary(iter_num=iter_num)

        doc_snippets = format_doc_snippets(docs, top_k=10, max_text_length=200)
        current_aspects_formatted = format_aspects_for_prompt(self._aspects)

        if self._mode == "static":
            system_prompt = ASPECT_UPDATE_STATIC_SYSTEM.format(
                max_aspects=self._max_aspects,
            )
            user_msg = ASPECT_UPDATE_STATIC_USER_TEMPLATE.format(
                query=query,
                current_aspects_formatted=current_aspects_formatted,
                subqueries="; ".join(subqueries),
                doc_snippets=doc_snippets,
            )
        else:
            frozen_instruction = FROZEN_INSTRUCTION if self._frozen else UNFROZEN_INSTRUCTION
            system_prompt = ASPECT_UPDATE_SYSTEM.format(
                max_aspects=self._max_aspects,
                frozen_instruction=frozen_instruction,
            )
            user_msg = ASPECT_UPDATE_USER_TEMPLATE.format(
                query=query,
                current_aspects_formatted=current_aspects_formatted,
                subqueries="; ".join(subqueries),
                doc_snippets=doc_snippets,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._llm.complete(
                messages, max_tokens=self._max_tokens, temperature=self._temperature,
            )
        except Exception:
            logger.warning("AspectCoverageSignal: update LLM call failed", exc_info=True)
            return self._build_summary(iter_num=iter_num)

        action_result = extract_aspect_actions(raw)
        if action_result is None:
            logger.warning("AspectCoverageSignal: update parsing failed")
            return self._build_summary(iter_num=iter_num)

        self._aspects, added, removed = apply_aspect_actions(
            current_aspects=self._aspects,
            action_result=action_result,
            max_aspects=self._max_aspects,
            frozen=self._frozen,
        )

        structure_changed = bool(added or removed)
        if structure_changed:
            self._last_structure_change_iter = iter_num

        if (
            not self._frozen
            and self._mode == "dynamic"
            and iter_num - self._last_structure_change_iter >= self._stabilization_window
            and iter_num > 0
        ):
            self._frozen = True
            logger.info(
                "AspectCoverageSignal: list frozen after %d iterations with no structural changes",
                self._stabilization_window,
            )

        result = self._build_summary(iter_num=iter_num)
        result.new_aspects_this_iter = sorted(added)
        result.removed_aspects_this_iter = sorted(removed)
        result.reasoning = action_result.reasoning

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_summary(self, iter_num: int) -> AspectCoverageSummary:
        """Build a summary from the current internal state."""
        num_covered = sum(1 for a in self._aspects if a.status == "covered")
        num_partial = sum(1 for a in self._aspects if a.status == "partial")
        num_not_covered = sum(1 for a in self._aspects if a.status == "not_covered")

        critical_gaps = [
            a.name for a in self._aspects if a.status == "not_covered"
        ]
        minor_gaps = [
            a.name for a in self._aspects if a.status == "partial"
        ]

        stable_since = None
        if self._frozen and self._mode == "dynamic":
            stable_since = self._last_structure_change_iter

        return AspectCoverageSummary(
            aspects=[Aspect(name=a.name, status=a.status, evidence=a.evidence) for a in self._aspects],
            num_covered=num_covered,
            num_partial=num_partial,
            num_not_covered=num_not_covered,
            total=len(self._aspects),
            critical_gaps=critical_gaps,
            minor_gaps=minor_gaps,
            stable_since=stable_since,
            frozen=self._frozen,
        )
